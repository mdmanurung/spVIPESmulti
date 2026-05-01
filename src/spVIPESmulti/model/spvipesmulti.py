import logging
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass
from scvi.utils import setup_anndata_dsp

from spVIPESmulti.data import AnnDataManager
from spVIPESmulti.dataloaders._concat_dataloader import ConcatDataLoader
from spVIPESmulti.model._disentangle_presets import DISENTANGLE_PRESETS
from spVIPESmulti.model.base.training_mixin import MultiGroupTrainingMixin
from spVIPESmulti.module.spVIPESmultimodule import spVIPESmultimodule

logger = logging.getLogger(__name__)


class spVIPESmulti(MultiGroupTrainingMixin, BaseModelClass):
    """
    Implementation of the spVIPESmulti model.

    spVIPESmulti (shared-private Variational Inference with Product of Experts and Supervision)
    is a method for integrating multi-group single-cell datasets using a shared-private
    latent space approach. The model learns both shared representations (common across
    groups) and private representations (group-specific) through a label-based Product
    of Experts (PoE) framework.

    Parameters
    ----------
    adata : AnnData
        AnnData object that has been registered via :func:`~spVIPESmulti.model.spVIPESmulti.setup_anndata`.
    n_hidden : int, default=128
        Number of nodes per hidden layer in the neural networks.
    n_dimensions_shared : int, default=25
        Dimensionality of the shared latent space. This space captures features
        common across all groups/datasets.
    n_dimensions_private : int, default=10
        Dimensionality of the private latent spaces. Each group gets its own
        private latent space of this dimensionality.
    dropout_rate : float, default=0.1
        Dropout rate for neural networks to prevent overfitting.
    **model_kwargs
        Additional keyword arguments passed to the underlying module.

    Examples
    --------
    Basic usage with cell type labels:

    >>> import spVIPESmulti
    >>> adata = spVIPESmulti.data.prepare_adatas({"dataset1": dataset1, "dataset2": dataset2})
    >>> spVIPESmulti.model.spVIPESmulti.setup_anndata(adata, groups_key="groups", label_key="cell_type")
    >>> model = spVIPESmulti.model.spVIPESmulti(adata)
    >>> model.train()
    >>> latents = model.get_latent_representation()

    Notes
    -----
    - We recommend setting n_dimensions_private < n_dimensions_shared for optimal performance
    - GPU acceleration is strongly recommended for large datasets
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_dimensions_shared: int = 25,
        n_dimensions_private: int = 10,
        dropout_rate: float = 0.1,
        use_nf_prior: bool = False,
        nf_type: str = "NSF",
        nf_transforms: int = 3,
        nf_target: str = "shared",
        disentangle_preset: str = "off",
        disentangle_group_shared_weight: Optional[float] = None,
        disentangle_label_shared_weight: Optional[float] = None,
        disentangle_group_private_weight: Optional[float] = None,
        disentangle_label_private_weight: Optional[float] = None,
        contrastive_weight: Optional[float] = None,
        contrastive_temperature: float = 0.1,
        modality_loss_weights: Optional[dict] = None,
        use_jeffreys_integ: bool = False,
        jeffreys_integ_weight: float = 1.0,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.adata = adata
        self.n_dimensions_private = n_dimensions_private
        self.n_dimensions_shared = n_dimensions_shared

        n_batch = self.summary_stats.n_batch

        groups_lengths = adata.uns["groups_lengths"]
        groups_obs_names = adata.uns["groups_obs_names"]
        groups_var_names = adata.uns["groups_var_names"]
        groups_obs_indices = adata.uns["groups_obs_indices"]
        groups_var_indices = adata.uns["groups_var_indices"]

        use_labels = "labels" in self.adata_manager.data_registry
        n_labels = self.summary_stats.n_labels if use_labels else None

        # Multimodal parameters (if available)
        groups_modality_lengths = adata.uns.get("groups_modality_lengths")
        groups_modality_var_indices = adata.uns.get("groups_modality_var_indices")
        modality_likelihoods = adata.uns.get("modality_likelihoods")
        modality_names = adata.uns.get("modality_names")
        groups_modality_masks = adata.uns.get("groups_modality_masks")

        # Resolve disentanglement preset + per-component overrides
        if disentangle_preset not in DISENTANGLE_PRESETS:
            raise ValueError(
                f"Unknown disentangle_preset={disentangle_preset!r}. Available: {list(DISENTANGLE_PRESETS)}"
            )
        _disentangle_weights = dict(DISENTANGLE_PRESETS[disentangle_preset])
        for _name, _override in (
            ("disentangle_group_shared_weight", disentangle_group_shared_weight),
            ("disentangle_label_shared_weight", disentangle_label_shared_weight),
            ("disentangle_group_private_weight", disentangle_group_private_weight),
            ("disentangle_label_private_weight", disentangle_label_private_weight),
            ("contrastive_weight", contrastive_weight),
        ):
            if _override is not None:
                _disentangle_weights[_name] = _override

        # Validate that no disentangle weight is negative
        for _name, _val in _disentangle_weights.items():
            if _val < 0:
                raise ValueError(
                    f"Disentangle weight '{_name}' must be >= 0, got {_val}. "
                    "Negative weights reverse the loss and will produce unexpected results."
                )

        self.module = spVIPESmultimodule(
            groups_lengths=groups_lengths,
            groups_obs_names=groups_obs_names,
            groups_var_names=groups_var_names,
            groups_var_indices=groups_var_indices,
            groups_obs_indices=groups_obs_indices,
            use_labels=use_labels,
            n_labels=n_labels,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_dimensions_shared=n_dimensions_shared,
            n_dimensions_private=n_dimensions_private,
            dropout_rate=dropout_rate,
            groups_modality_lengths=groups_modality_lengths,
            groups_modality_var_indices=groups_modality_var_indices,
            modality_likelihoods=modality_likelihoods,
            modality_names=modality_names,
            groups_modality_masks=groups_modality_masks,
            modality_loss_weights=modality_loss_weights,
            use_jeffreys_integ=use_jeffreys_integ,
            jeffreys_integ_weight=jeffreys_integ_weight,
            use_nf_prior=use_nf_prior,
            nf_type=nf_type,
            nf_transforms=nf_transforms,
            nf_target=nf_target,
            **_disentangle_weights,
            contrastive_temperature=contrastive_temperature,
            **model_kwargs,
        )

        is_multimodal = groups_modality_lengths is not None
        self._model_summary_string = (
            "spVIPESmulti Model with the following params: \nn_hidden: {}, n_dimensions_shared: {}, "
            "n_dimensions_private: {}, dropout_rate: {}, multimodal: {}, "
            "nf_prior: {}"
        ).format(
            n_hidden,
            n_dimensions_shared,
            n_dimensions_private,
            dropout_rate,
            "Yes" if is_multimodal else "No",
            f"{nf_type}({nf_transforms} transforms, target={nf_target})" if use_nf_prior else "No",
        )
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        groups_key: str,
        label_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        modality_likelihoods: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> None:
        """
        Set up AnnData object for spVIPESmulti model.

        Parameters
        ----------
        adata : AnnData
            Annotated data object containing the single-cell data to be integrated.
        groups_key : str
            Key in `adata.obs` that defines the grouping of cells.
        label_key : str, optional
            Key in `adata.obs` containing cell type labels for label-based PoE.
        batch_key : str, optional
            Key in `adata.obs` for batch information.
        layer : str, optional
            Key in `adata.layers` to use. If None, uses `adata.X`.
        modality_likelihoods : dict[str, str], optional
            Mapping from modality name to likelihood type for multimodal data.
            Supported values: ``"nb"`` and ``"gaussian"``.
        **kwargs
            Additional keyword arguments passed to the parent setup method.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField("groups", groups_key),
        ]

        print("=== spVIPESmulti AnnData Setup ===")
        print(f"Setting up with groups_key: '{groups_key}'")

        if label_key is not None:
            print(f"✓ Labels: Using '{label_key}' from adata.obs")
            anndata_fields.append(CategoricalObsField("labels", label_key))
            anndata_fields.append(CategoricalObsField("indices", "indices"))

        print("\n--- Product of Experts (PoE) Configuration ---")
        if label_key is not None:
            print("🎯 Will use: Label-based PoE")
        else:
            print("⚠️  No labels configured — provide label_key for PoE-based integration")

        if modality_likelihoods is not None:
            adata.uns["modality_likelihoods"] = modality_likelihoods
            print(f"✓ Multimodal: Configured with likelihoods {modality_likelihoods}")

        print("=" * 45)

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
        self,
        group_indices_list: list[list[int]],
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        normalized: bool = False,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        drop_last: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Return the latent representation for each cell.

        Parameters
        ----------
        group_indices_list
            List of lists containing the indices of cells in each of the groups used as input for spVIPESmulti.
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        normalized
            Whether to return the normalized cell embedding (softmaxed) or not
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        drop_last
            Whether to drop the last incomplete batch. If None, defaults to False.

        Returns
        -------
        Low-dimensional topic for each cell.
        """
        adata = self._validate_anndata(adata)
        n_groups = len(group_indices_list)
        n_per_group = [len(group) for group in group_indices_list]

        # Automatically determine drop_last
        if drop_last is None:
            drop_last = False

        # Standard processing
        scdl = ConcatDataLoader(
            self.adata_manager,
            indices_list=group_indices_list,
            shuffle=False,
            drop_last=drop_last,
            batch_size=batch_size,
        )

        results = self._process_batches(scdl, normalized, give_mean, mc_samples, n_groups)
        return self._format_results(results, n_per_group)

    def _process_batches(self, dataloader, normalized, give_mean, mc_samples, n_groups=None):
        """Process batches and return intermediate results for N groups."""
        if n_groups is None:
            # Infer from first batch
            for tensors_by_group in dataloader:
                per_group_probe = self.module._split_tensors_by_group(tensors_by_group)
                n_groups = len(per_group_probe)
                break
            else:
                raise ValueError("Dataloader is empty")
            # Re-create dataloader since we consumed the first batch
            # Instead, we just set n_groups from the dataloader's own data
            # This is a fallback; callers should pass n_groups explicitly

        latent_shared = {g: [] for g in range(n_groups)}
        latent_private = {g: [] for g in range(n_groups)}
        original_indices = {g: [] for g in range(n_groups)}

        # For multimodal: per-(group, modality) private latents
        is_multimodal = self.module.is_multimodal
        latent_private_multimodal = {} if is_multimodal else None
        if is_multimodal:
            for g in range(n_groups):
                for mod in self.module.group_modalities[g]:
                    latent_private_multimodal[(g, mod)] = []

        for tensors_by_group in dataloader:
            per_group = self.module._split_tensors_by_group(tensors_by_group)
            inference_inputs = self.module._get_inference_input(tensors_by_group)
            outputs = self.module.inference(**inference_inputs)

            for g in range(n_groups):
                # Shared (PoE) latent
                poe_log_z = outputs["poe_stats"][g]["logtheta_log_z"]
                if not normalized:
                    latent_shared[g].append(poe_log_z.cpu())

                # Private latent (group-level)
                private_log_z = outputs["private_stats"][g]["log_z"]
                private_qz = outputs["private_stats"][g]["qz"]
                if not normalized:
                    latent_private[g].append(private_log_z.cpu())
                else:
                    if give_mean:
                        samples = private_qz.sample([mc_samples])
                        theta = torch.nn.functional.softmax(samples, dim=-1).mean(dim=0)
                    else:
                        theta = outputs["private_stats"][g]["theta"]
                    latent_private[g].append(theta.cpu())

                original_indices[g].append(per_group[g]["indices"].cpu())

                # Multimodal: collect per-modality private latents
                if is_multimodal and "per_modality_private" in outputs:
                    for mod in self.module.group_modalities[g]:
                        mod_private = outputs["per_modality_private"].get((g, mod))
                        if mod_private is not None:
                            if not normalized:
                                latent_private_multimodal[(g, mod)].append(mod_private["log_z"].cpu())
                            else:
                                mod_qz = mod_private["qz"]
                                if give_mean:
                                    samples = mod_qz.sample([mc_samples])
                                    theta = torch.nn.functional.softmax(samples, dim=-1).mean(dim=0)
                                else:
                                    theta = mod_private["theta"]
                                latent_private_multimodal[(g, mod)].append(theta.cpu())

        result = {
            "latent_shared": latent_shared,
            "latent_private": latent_private,
            "original_indices": original_indices,
        }
        if is_multimodal:
            result["latent_private_multimodal"] = latent_private_multimodal
        return result

    def _process_all_cells_with_cycling(self, group_indices_list, normalized, give_mean, mc_samples, batch_size):
        """Process all cells using cycling approach to handle unequal group sizes."""
        n_groups = len(group_indices_list)
        group_sizes = [len(indices) for indices in group_indices_list]
        min_group_size = min(group_sizes)
        max_group_size = max(group_sizes)

        if min_group_size == 0:
            raise ValueError("One of the groups is empty")

        results = {
            "latent_shared": {g: [] for g in range(n_groups)},
            "latent_private": {g: [] for g in range(n_groups)},
            "original_indices": {g: [] for g in range(n_groups)},
        }

        for start_idx in range(0, max_group_size, min_group_size):
            chunk_indices = []
            for g in range(n_groups):
                group_chunk = []
                for i in range(min_group_size):
                    idx = (start_idx + i) % len(group_indices_list[g])
                    group_chunk.append(group_indices_list[g][idx])
                chunk_indices.append(group_chunk)

            chunk_scdl = ConcatDataLoader(
                self.adata_manager,
                indices_list=chunk_indices,
                shuffle=False,
                drop_last=False,
                batch_size=batch_size,
            )

            chunk_results = self._process_batches(chunk_scdl, normalized, give_mean, mc_samples, n_groups)

            for key in results:
                for g in range(n_groups):
                    results[key][g].extend(chunk_results[key][g])

        return results

    def _format_results(self, results, n_per_group):
        """Format the final results dictionary for N groups."""
        n_groups = len(n_per_group)

        latent_private = {}
        latent_shared = {}
        latent_private_reordered = {}
        latent_shared_reordered = {}

        for g in range(n_groups):
            n_g = n_per_group[g]
            g_private = torch.cat(results["latent_private"][g]).numpy()[:n_g]
            g_shared = torch.cat(results["latent_shared"][g]).numpy()[:n_g]
            g_indices = torch.cat(results["original_indices"][g]).numpy().flatten()[:n_g]

            latent_private[g] = g_private
            latent_shared[g] = g_shared
            latent_private_reordered[g] = g_private[np.argsort(g_indices)]
            latent_shared_reordered[g] = g_shared[np.argsort(g_indices)]

        output = {
            "shared": latent_shared,
            "private": latent_private,
            "shared_reordered": latent_shared_reordered,
            "private_reordered": latent_private_reordered,
        }

        # Add per-(group, modality) private latents if multimodal
        if "latent_private_multimodal" in results:
            private_multimodal = {}
            private_multimodal_reordered = {}
            for (g, mod), tensors in results["latent_private_multimodal"].items():
                if tensors:
                    n_g = n_per_group[g]
                    g_mod_private = torch.cat(tensors).numpy()[:n_g]
                    g_indices = torch.cat(results["original_indices"][g]).numpy().flatten()[:n_g]
                    private_multimodal[(g, mod)] = g_mod_private
                    private_multimodal_reordered[(g, mod)] = g_mod_private[np.argsort(g_indices)]
            output["private_multimodal"] = private_multimodal
            output["private_multimodal_reordered"] = private_multimodal_reordered

        return output

    def get_loadings(self) -> dict:
        """Extract per-gene weights in the linear decoder.

        Shape is genes by `n_latent`.

        """
        num_datasets = len(self.module.input_dims)
        datasets_obs_indices = self.module.groups_obs_indices
        datasets_var_indices = self.module.groups_var_indices
        adata = self.adata
        loadings_dict = {}
        for i in range(num_datasets):
            dataset_obs_indices = datasets_obs_indices[i]
            s_adata = adata[dataset_obs_indices, :].copy()
            cols_private = [f"Z_private_{n}" for n in range(self.module.n_dimensions_private)]
            cols_shared = [f"Z_shared_{n}" for n in range(self.module.n_dimensions_shared)]
            var_names = s_adata[:, datasets_var_indices[i]].var_names
            loadings_private = pd.DataFrame(
                self.module.get_loadings(i, "private"), index=var_names, columns=cols_private
            )
            loadings_shared = pd.DataFrame(self.module.get_loadings(i, "shared"), index=var_names, columns=cols_shared)

            loadings_dict[(i, "private")] = loadings_private
            loadings_dict[(i, "shared")] = loadings_shared

        return loadings_dict
