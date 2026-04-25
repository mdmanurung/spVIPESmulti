import logging
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from scvi import REGISTRY_KEYS
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass
from scvi.utils import setup_anndata_dsp
from tqdm import tqdm

from spVIPES.data import AnnDataManager
from spVIPES.dataloaders._concat_dataloader import ConcatDataLoader
from spVIPES.model._mig_presets import MIG_PRESETS
from spVIPES.model.base.training_mixin import MultiGroupTrainingMixin
from spVIPES.module.spVIPESmodule import spVIPESmodule

logger = logging.getLogger(__name__)


def process_transport_plan(transport_plan, adata, groups_key):
    """
    Process the transport plan using cluster labels to create a common set of clusters between datasets.
    Only supports exactly 2 groups.

    Parameters
    ----------
    transport_plan : np.ndarray
        The original transport plan matrix (shape: cells1 x cells2).
    adata : AnnData
        The AnnData object containing the combined datasets.
    groups_key : str
        Key for grouping of cells in `adata.obs`.

    Returns
    -------
    processed_labels : np.ndarray
        Array of processed cluster labels for all cells.
    """
    transport_plan = np.nan_to_num(transport_plan, nan=0.0)

    # Extract the groups - only 2 groups supported for transport plan processing
    groups = adata.obs[groups_key].unique()
    if len(groups) != 2:
        raise ValueError(
            f"Transport plan processing only supports exactly 2 groups, got {len(groups)}. "
            "Use label-based PoE for more than 2 groups."
        )

    # Extract the groups
    groups = adata.obs[groups_key].unique()
    cluster_labels = []

    def optimize_resolution(group_adata, group_transport_plan, other_group_size):
        resolutions = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        scores = []
        for res in tqdm(resolutions, desc="Optimizing resolution"):
            sc.tl.leiden(group_adata, resolution=res, key_added=f"leiden_{res}")
            cluster_transport = np.zeros((len(group_adata.obs[f"leiden_{res}"].unique()), other_group_size))
            for i, cluster in enumerate(group_adata.obs[f"leiden_{res}"].unique()):
                mask = group_adata.obs[f"leiden_{res}"] == cluster
                cluster_transport[i] = group_transport_plan[mask].sum(axis=0)

            # Normalize the cluster transport
            cluster_transport /= cluster_transport.sum(axis=1, keepdims=True)

            # Calculate the entropy of the transport distribution for each cluster
            cluster_entropies = entropy(cluster_transport, axis=1)

            # Use the negative mean entropy as the score (higher is better)
            scores.append(-np.mean(cluster_entropies))

        optimal_res = resolutions[np.argmax(scores)]
        return optimal_res

    optimal_resolutions = {}

    for i, group in enumerate(groups):
        group_mask = adata.obs[groups_key] == group
        group_adata = adata[group_mask].copy()
        # Filter out .var indices that don't correspond to this group
        group_var_names = adata.uns["groups_var_names"][group]
        group_adata = group_adata[:, group_adata.var_names.isin(group_var_names)].copy()

        # Normalize the data
        sc.pp.normalize_total(group_adata)
        sc.pp.log1p(group_adata)
        sc.pp.pca(group_adata)

        # Compute neighborhood graph
        sc.pp.neighbors(group_adata)

        # Optimize resolution
        other_group_size = adata[adata.obs[groups_key] != group].shape[0]
        group_transport_plan = transport_plan if i == 0 else transport_plan.T
        optimal_res = optimize_resolution(group_adata, group_transport_plan, other_group_size)
        optimal_resolutions[group] = optimal_res

        # Perform Leiden clustering with optimal resolution
        sc.tl.leiden(group_adata, resolution=optimal_res)
        group_clusters = group_adata.obs["leiden"].astype(str)
        group_clusters = group + "_" + group_clusters
        cluster_labels.extend(group_clusters)

    # Add the cluster labels to adata.obs
    adata.obs["group_cluster_labels"] = pd.Categorical(cluster_labels)

    # Create a DataFrame of transport values between clusters
    clusters1 = adata[adata.obs[groups_key] == groups[0]].obs["group_cluster_labels"]
    clusters2 = adata[adata.obs[groups_key] == groups[1]].obs["group_cluster_labels"]

    transport_df = pd.DataFrame(
        {
            "source_cluster": np.repeat(clusters1, len(clusters2)),
            "target_cluster": np.tile(clusters2, len(clusters1)),
            "transport_value": transport_plan.flatten(),
        }
    )

    # Create a pivot table of median transport values between clusters
    pivot_df = transport_df.pivot_table(
        values="transport_value", index="source_cluster", columns="target_cluster", aggfunc="median"
    )

    def rename_clusters(pivot_df):
        # Convert the pivot_df to a cost matrix (negative because we want to maximize)
        cost_matrix = -pivot_df.values

        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create the rename dictionary
        rename_dict = {}
        for i, (source_idx, target_idx) in enumerate(zip(row_ind, col_ind)):
            source_cluster = pivot_df.index[source_idx]
            target_cluster = pivot_df.columns[target_idx]
            new_name = f"Cluster_{i}"
            rename_dict[source_cluster] = new_name
            rename_dict[target_cluster] = new_name

        # Handle any unmatched clusters
        all_clusters = set(pivot_df.index) | set(pivot_df.columns)
        matched_clusters = set(rename_dict.keys())
        unmatched_clusters = all_clusters - matched_clusters

        for cluster in unmatched_clusters:
            new_name = f"Cluster_{len(rename_dict) // 2}"
            rename_dict[cluster] = new_name

        return rename_dict

    rename_dict = rename_clusters(pivot_df)

    # Apply renaming to the AnnData object
    adata.obs["processed_transport_labels"] = adata.obs["group_cluster_labels"].map(rename_dict)

    # Ensure the categories are in the correct order and format
    categories = np.array(sorted(set(rename_dict.values()), key=lambda x: int(x.split("_")[1])))
    adata.obs["processed_transport_labels"] = pd.Categorical(
        adata.obs["processed_transport_labels"], categories=categories, ordered=True
    )

    # Store the optimal resolutions in adata.uns
    adata.uns["optimal_resolutions"] = optimal_resolutions

    return adata.obs["processed_transport_labels"].values


class spVIPES(MultiGroupTrainingMixin, BaseModelClass):
    """
    Implementation of the spVIPES model.

    spVIPES (shared-private Variational Inference with Product of Experts and Supervision)
    is a method for integrating multi-group single-cell datasets using a shared-private
    latent space approach. The model learns both shared representations (common across
    groups) and private representations (group-specific) through a Product of Experts (PoE)
    framework.

    Parameters
    ----------
    adata : AnnData
        AnnData object that has been registered via :func:`~spVIPES.model.spVIPES.setup_anndata`.
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

    >>> import spVIPES
    >>> adata = spVIPES.data.prepare_adatas({"dataset1": dataset1, "dataset2": dataset2})
    >>> spVIPES.model.spVIPES.setup_anndata(adata, groups_key="groups", label_key="cell_type")
    >>> model = spVIPES.model.spVIPES(adata)
    >>> model.train()
    >>> latents = model.get_latent_representation()

    Usage with optimal transport:

    >>> spVIPES.model.spVIPES.setup_anndata(adata, groups_key="groups", transport_plan_key="transport_plan")
    >>> model = spVIPES.model.spVIPES(adata)
    >>> model.train()

    Notes
    -----
    - We recommend setting n_dimensions_private < n_dimensions_shared for optimal performance
    - The model automatically selects the appropriate PoE variant based on provided inputs
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
        mig_preset: str = "off",
        mig_group_shared_weight: Optional[float] = None,
        mig_label_shared_weight: Optional[float] = None,
        mig_group_private_weight: Optional[float] = None,
        mig_label_private_weight: Optional[float] = None,
        contrastive_weight: Optional[float] = None,
        contrastive_temperature: float = 0.1,
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

        setup_args = self.adata_manager._get_setup_method_args()["setup_args"]
        transport_plan_key = setup_args.get("transport_plan_key")

        if transport_plan_key:
            transport_plan_data = adata.uns.get(transport_plan_key)
            if transport_plan_data is None:
                raise ValueError(f"Transport plan not found in adata.uns['{transport_plan_key}']")
            transport_plan = torch.tensor(transport_plan_data, dtype=torch.float32)
        else:
            transport_plan = None

        pair_data = "processed_transport_labels" not in adata.obs.columns

        use_labels = "labels" in self.adata_manager.data_registry
        n_labels = self.summary_stats.n_labels if use_labels else None

        # Multimodal parameters (if available)
        groups_modality_lengths = adata.uns.get("groups_modality_lengths")
        groups_modality_var_indices = adata.uns.get("groups_modality_var_indices")
        modality_likelihoods = adata.uns.get("modality_likelihoods")
        modality_names = adata.uns.get("modality_names")

        # Resolve MIG preset + per-component overrides
        if mig_preset not in MIG_PRESETS:
            raise ValueError(
                f"Unknown mig_preset={mig_preset!r}. Available: {list(MIG_PRESETS)}"
            )
        _mig_weights = dict(MIG_PRESETS[mig_preset])
        for _name, _override in (
            ("mig_group_shared_weight", mig_group_shared_weight),
            ("mig_label_shared_weight", mig_label_shared_weight),
            ("mig_group_private_weight", mig_group_private_weight),
            ("mig_label_private_weight", mig_label_private_weight),
            ("contrastive_weight", contrastive_weight),
        ):
            if _override is not None:
                _mig_weights[_name] = _override

        self.module = spVIPESmodule(
            groups_lengths=groups_lengths,
            groups_obs_names=groups_obs_names,
            groups_var_names=groups_var_names,
            groups_var_indices=groups_var_indices,
            groups_obs_indices=groups_obs_indices,
            transport_plan=transport_plan,
            pair_data=pair_data,
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
            use_nf_prior=use_nf_prior,
            nf_type=nf_type,
            nf_transforms=nf_transforms,
            nf_target=nf_target,
            **_mig_weights,
            contrastive_temperature=contrastive_temperature,
            **model_kwargs,
        )

        is_multimodal = groups_modality_lengths is not None
        self._model_summary_string = (
            "spVIPES Model with the following params: \nn_hidden: {}, n_dimensions_shared: {}, "
            "n_dimensions_private: {}, dropout_rate: {}, transport_plan: {}, multimodal: {}, "
            "nf_prior: {}"
        ).format(
            n_hidden,
            n_dimensions_shared,
            n_dimensions_private,
            dropout_rate,
            "Provided" if transport_plan is not None else "Not provided",
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
        match_clusters: bool = False,
        transport_plan_key: Optional[str] = None,
        label_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        modality_likelihoods: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> None:
        """
        Set up AnnData object for spVIPES model.

        This method registers the AnnData object with the model, configuring the
        appropriate data fields and PoE strategy based on the provided parameters.
        The method automatically determines whether to use label-based PoE,
        optimal transport PoE, or cluster-based PoE.

        Parameters
        ----------
        adata : AnnData
            Annotated data object containing the single-cell data to be integrated.
        groups_key : str
            Key in `adata.obs` that defines the grouping of cells (e.g., dataset, batch, condition).
            This determines which cells belong to which group for integration.
        match_clusters : bool, default=False
            Whether to match clusters when using optimal transport. If True, enables
            cluster-based PoE which automatically matches cell clusters between groups.
        transport_plan_key : str, optional
            Key in `adata.uns` containing the precomputed optimal transport plan.
            If provided, enables optimal transport PoE for data integration.
        label_key : str, optional
            Key in `adata.obs` containing cell type labels. If provided, enables
            label-based PoE which uses supervised alignment based on cell types.
        batch_key : str, optional
            Key in `adata.obs` for batch information to enable batch effect correction.
        layer : str, optional
            Key in `adata.layers` to use for the expression data. If None, uses `adata.X`.
        modality_likelihoods : dict[str, str], optional
            Mapping from modality name to likelihood type for multimodal data.
            Supported values: ``"nb"`` (NegativeBinomial) and ``"gaussian"``.
            If ``None``, single-modality mode with NB likelihood (backward compatible).
        **kwargs
            Additional keyword arguments passed to the parent setup method.

        Returns
        -------
        None
            The method modifies the AnnData object in place and registers it with the model.

        Notes
        -----
        Priority of PoE strategies (when multiple options are available):
        1. Label-based PoE (if `label_key` is provided)
        2. Optimal transport PoE (if `transport_plan_key` is provided)
        3. Cluster-based PoE (if `match_clusters=True`)

        Examples
        --------
        Basic setup with groups only:

        >>> spVIPES.model.spVIPES.setup_anndata(adata, groups_key="dataset")

        Setup with cell type supervision:

        >>> spVIPES.model.spVIPES.setup_anndata(adata, groups_key="dataset", label_key="cell_type")

        Setup with optimal transport:

        >>> spVIPES.model.spVIPES.setup_anndata(adata, groups_key="dataset", transport_plan_key="transport_matrix")
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField("groups", groups_key),
        ]

        print("=== spVIPES AnnData Setup ===")
        print(f"Setting up with groups_key: '{groups_key}'")

        transport_plan_configured = False
        labels_configured = False

        if transport_plan_key is not None:
            if transport_plan_key not in adata.uns:
                raise ValueError(f"Transport plan key '{transport_plan_key}' not found in adata.uns")
            adata.uns["transport_plan"] = adata.uns[transport_plan_key]
            transport_plan_configured = True

            print(f"✓ Transport plan: Using '{transport_plan_key}' from adata.uns")

            # Process the transport plan
            transport_plan = adata.uns[transport_plan_key]

            if match_clusters:
                print("✓ Cluster matching: Enabled - will create processed transport labels")
                # Process the transport plan using the cluster labels
                processed_labels = process_transport_plan(
                    transport_plan,
                    adata,
                    groups_key,
                )
                adata.obs["processed_transport_labels"] = pd.Categorical(processed_labels)
                anndata_fields.append(CategoricalObsField("processed_transport_labels", "processed_transport_labels"))
            else:
                print("✓ Cluster matching: Disabled - will use direct cell pairing")

            # Add indices field if using transport plan
            anndata_fields.append(CategoricalObsField("indices", "indices"))

            if "indices" not in adata.obs:
                raise ValueError("'indices' must be present in adata.obs when using a transport plan")

        if label_key is not None:
            labels_configured = True
            print(f"✓ Labels: Using '{label_key}' from adata.obs")
            anndata_fields.append(CategoricalObsField("labels", label_key))
            anndata_fields.append(CategoricalObsField("indices", "indices"))

        # Inform user about the PoE method that will be used
        print("\n--- Product of Experts (PoE) Configuration ---")
        if labels_configured and transport_plan_configured:
            print("🎯 Will use: Label-based PoE (labels take priority over transport plan)")
        elif labels_configured:
            print("🎯 Will use: Label-based PoE")
        elif transport_plan_configured:
            if match_clusters:
                print("🎯 Will use: Cluster-based PoE (transport plan with cluster matching)")
            else:
                print("🎯 Will use: Paired PoE (direct cell-to-cell transport plan)")
        else:
            print("⚠️  No transport plan or labels configured - you may need one for integration")

        # Store multimodal configuration if provided
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
            List of lists containing the indices of cells in each of the groups used as input for spVIPES.
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
            Whether to drop the last incomplete batch. If None, automatically determined based on
            whether using paired PoE (True for paired, False for others).

        Returns
        -------
        Low-dimensional topic for each cell.
        """
        adata = self._validate_anndata(adata)
        n_groups = len(group_indices_list)
        n_per_group = [len(group) for group in group_indices_list]

        # Automatically determine drop_last based on PoE type if not specified
        if drop_last is None:
            if self.module.use_labels and "labels" in self.adata_manager.data_registry:
                drop_last = False
            elif self.module.use_transport_plan and self.module.pair_data:
                drop_last = False
            else:
                drop_last = False

        # For paired PoE with drop_last=False, use cycling to handle unequal group sizes
        use_cycling = (
            self.module.use_transport_plan
            and self.module.pair_data
            and not drop_last
            and not (self.module.use_labels and "labels" in self.adata_manager.data_registry)
        )

        if use_cycling:
            results = self._process_all_cells_with_cycling(
                group_indices_list, normalized, give_mean, mc_samples, batch_size
            )
            return self._format_results(results, n_per_group)

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
                n_groups = len(tensors_by_group)
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

                original_indices[g].append(tensors_by_group[g]["indices"].cpu())

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
