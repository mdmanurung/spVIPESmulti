import logging
import warnings
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS, settings
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass
from scvi.utils import setup_anndata_dsp

from spVIPESmulti.data import AnnDataManager
from spVIPESmulti.dataloaders._concat_dataloader import ConcatDataLoader
from spVIPESmulti.model._disentangle_presets import DISENTANGLE_PRESETS
from spVIPESmulti.model.base.training_mixin import MultiGroupTrainingMixin
from spVIPESmulti.module.spVIPESmultimodule import spVIPESmultimodule
from spVIPESmulti.utils import (
    resolve_group_indices_list,
    sanitize_obsm_token,
    validate_enrichment_network,
)

logger = logging.getLogger(__name__)


class spVIPESmulti(MultiGroupTrainingMixin, BaseModelClass):
    """
    Implementation of the spVIPESmulti model.

    spVIPESmulti (shared-private Variational Inference with Product of Experts and Supervision)
    is a method for integrating multi-group single-cell datasets using a shared-private
    latent space approach. The model learns both shared representations (common across
    groups) and private representations (group-specific) through a Product of Experts
    (PoE) framework.

    Parameters
    ----------
    adata : AnnData
        AnnData object that has been registered via
        :func:`~spVIPESmulti.model.spVIPESmulti.setup_anndata`.
    n_hidden : int, default=128
        Number of nodes per hidden layer in all encoder and decoder networks.
    n_dimensions_shared : int, default=25
        Dimensionality of the shared latent space ``z_shared``. Captures biology
        common across all groups.
    n_dimensions_private : int, default=10
        Dimensionality of each group's private latent space ``z_private``.
        Captures group-specific variation.
    dropout_rate : float, default=0.1
        Dropout probability applied in all encoder / decoder hidden layers.
    use_nf_prior : bool, default=False
        Replace the standard N(0, I) prior with a learned normalizing-flow prior
        (implemented via Zuko). See ``nf_type``, ``nf_transforms``, ``nf_target``.
    nf_type : str, default="NSF"
        Flow architecture when ``use_nf_prior=True``. ``"NSF"`` (Neural Spline
        Flow) or ``"MAF"`` (Masked Autoregressive Flow).
    nf_transforms : int, default=3
        Number of sequential coupling transforms in the normalizing flow.
    nf_target : str, default="shared"
        Which latent(s) receive the flow prior. One of ``"shared"``,
        ``"private"``, or ``"both"``.
    disentangle_preset : str, default="off"
        Named preset activating the optional disentanglement objective.
        Available presets: ``"off"``, ``"full"``, ``"shared_only"``,
        ``"private_only"``, ``"adversarial_only"``, ``"supervised_only"``,
        ``"no_contrastive"``, ``"minimal_safe_bio"``, ``"full_bio"``.
    disentangle_group_shared_weight : float or None, default=None
        Override the preset's weight for the adversarial group classifier on
        ``z_shared`` (gradient-reversal layer). ``None`` keeps the preset value.
    disentangle_label_shared_weight : float or None, default=None
        Override the preset's weight for the supervised label classifier on
        ``z_shared`` (mutual-information lower bound). Requires ``label_key``.
    disentangle_group_private_weight : float or None, default=None
        Override the preset's weight for the supervised group classifier on
        ``z_private``.
    disentangle_label_private_weight : float or None, default=None
        Override the preset's weight for the adversarial label classifier on
        ``z_private`` (gradient-reversal layer). Requires ``label_key``.
    disentangle_batch_shared_weight : float or None, default=None
        Override the preset's weight for the adversarial technical-batch
        classifier on ``z_shared``. Requires ``batch_key``.
    disentangle_donor_shared_weight : float or None, default=None
        Override the preset's weight for the adversarial donor classifier on
        ``z_shared``. Requires ``donor_key``.
    disentangle_donor_private_weight : float or None, default=None
        Override the preset's weight for the supervised donor classifier on
        ``z_private``. Requires ``donor_key``.
    contrastive_weight : float or None, default=None
        Override the preset's weight for the prototype InfoNCE loss on
        ``z_shared``. Requires ``label_key``. ``None`` keeps the preset value.
    contrastive_temperature : float, default=0.1
        Temperature for the InfoNCE softmax denominator.
    disentangle_warmup : bool, default=True
        Whether to warm up covariate gradient-reversal strength with the KL
        warmup schedule.
    modality_loss_weights : dict[str, float] or None, default=None
        Per-modality scalar multipliers on the reconstruction loss.
        E.g. ``{"rna": 1.0, "protein": 5.0}`` to up-weight the protein term.
        Multimodal mode only (ignored for single-modal data).
    use_jeffreys_integ : bool, default=False
        Add a Jeffreys (symmetric KL) integration loss between every pair of
        group PoE posteriors on ``z_shared``.
    jeffreys_integ_weight : float, default=1.0
        Scalar multiplier on the Jeffreys integration loss.
    group_loss_weights : list of float, optional
        Per-group weights applied to each group's ELBO term before summing.
        Automatically normalized to sum to 1.

        Two common choices:

        * **Inverse group sizes** ``[1/n_g0, 1/n_g1, ...]`` (recommended starting
          point): gives larger weight to smaller groups so they contribute
          proportionally to the shared latent space even when they have fewer cells.
          Because :class:`ConcatDataLoader` already cycles smaller groups to
          produce equal numbers of batches per epoch, inverse weights further
          emphasize rare groups — useful when you care about quality of
          representation for minority populations::

              sizes = [n_g0, n_g1, n_g2]
              group_loss_weights = [1 / n for n in sizes]  # normalized internally

        * **Proportional to group size** ``[n_g0, n_g1, ...]``: down-weights
          smaller (cycled) groups, which moves the effective ELBO closer to the
          true dataset-level ELBO where each observation contributes equally.

        Default ``None`` gives equal weights (1 / n_groups per group).
    **model_kwargs
        Additional keyword arguments forwarded to :class:`~spVIPESmulti.module.spVIPESmultimodule`.

    Examples
    --------
    Basic usage with cell type labels:

    >>> import spVIPESmulti
    >>> adata = spVIPESmulti.data.prepare_adatas({"ctrl": adata_ctrl, "treat": adata_treat})
    >>> spVIPESmulti.model.spVIPESmulti.setup_anndata(adata, groups_key="groups", label_key="cell_type")
    >>> model = spVIPESmulti.model.spVIPESmulti(adata)
    >>> model.train(max_epochs=200)
    >>> model.embed(batch_size=512)

    With full disentanglement and NF prior:

    >>> model = spVIPESmulti.model.spVIPESmulti(
    ...     adata,
    ...     use_nf_prior=True,
    ...     nf_type="NSF",
    ...     disentangle_preset="full",
    ...     contrastive_weight=0.0,  # override to disable InfoNCE
    ... )

    Notes
    -----
    - Disentanglement components that use labels (2, 4, 5) require ``label_key``
      in :meth:`setup_anndata`.
    - Batch and donor disentanglement components require ``batch_key`` and
      ``donor_key`` respectively.
    - Individual weight overrides stack on top of a preset: any numeric value
      (including ``0.0``) replaces the preset; ``None`` keeps it.
    - GPU acceleration is strongly recommended for large datasets.
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
        disentangle_batch_shared_weight: Optional[float] = None,
        disentangle_donor_shared_weight: Optional[float] = None,
        disentangle_donor_private_weight: Optional[float] = None,
        contrastive_weight: Optional[float] = None,
        contrastive_temperature: float = 0.1,
        disentangle_warmup: bool = True,
        modality_loss_weights: Optional[dict] = None,
        use_jeffreys_integ: bool = False,
        jeffreys_integ_weight: float = 1.0,
        group_loss_weights: Optional[list] = None,
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
        use_condition = "condition" in self.adata_manager.data_registry
        use_donor = "donor" in self.adata_manager.data_registry
        n_conditions = (
            len(self.adata_manager.get_state_registry("condition").categorical_mapping)
            if use_condition else None
        )
        n_donors = (
            len(self.adata_manager.get_state_registry("donor").categorical_mapping)
            if use_donor else None
        )
        setup_args = self.adata_manager.registry.get("setup_args", {})
        use_batch_covariate = setup_args.get("batch_key") is not None and n_batch > 1

        # Per-class inverse-frequency weights for label-CE components (N5-E).
        # Computed once at init from the full dataset label distribution so that
        # minority cell types (e.g. Activated MZ n≈200 vs Atypical n≈3155)
        # receive proportionally larger gradient signals.
        label_class_weights: Optional[torch.Tensor] = None
        if use_labels and n_labels is not None:
            label_codes = self.adata_manager.get_from_registry("labels")
            codes_arr = np.asarray(label_codes).astype(int).ravel()
            counts = np.bincount(codes_arr, minlength=n_labels).astype(float)
            inv_freq = 1.0 / np.maximum(counts, 1.0)
            # Normalize so weights sum to n_labels (keeps loss scale stable)
            label_class_weights = torch.tensor(
                inv_freq / inv_freq.sum() * n_labels, dtype=torch.float32
            )

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
            ("disentangle_batch_shared_weight", disentangle_batch_shared_weight),
            ("disentangle_donor_shared_weight", disentangle_donor_shared_weight),
            ("disentangle_donor_private_weight", disentangle_donor_private_weight),
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
            use_condition=use_condition,
            n_conditions=n_conditions,
            use_donor=use_donor,
            n_donors=n_donors,
            use_batch_covariate=use_batch_covariate,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_dimensions_shared=n_dimensions_shared,
            n_dimensions_private=n_dimensions_private,
            dropout_rate=dropout_rate,
            label_class_weights=label_class_weights,
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
            disentangle_warmup=disentangle_warmup,
            group_loss_weights=group_loss_weights,
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
        sample_key: Optional[str] = None,
        condition_key: Optional[str] = None,
        donor_key: Optional[str] = None,
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
        sample_key : str, optional
            Key in `adata.obs` containing sample identifiers used for
            sample-aware differential abundance aggregation.
        condition_key : str, optional
            Key in `adata.obs` containing perturbation or treatment state.
        donor_key : str, optional
            Key in `adata.obs` containing donor or biological replicate identity.
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

        logger.info("=== spVIPESmulti AnnData Setup ===")
        logger.info("Setting up with groups_key: '%s'", groups_key)

        anndata_fields.append(CategoricalObsField("indices", "indices"))

        if label_key is not None:
            logger.info("Labels: Using '%s' from adata.obs", label_key)
            anndata_fields.append(CategoricalObsField("labels", label_key))

        if sample_key is not None:
            logger.info("Samples: Using '%s' from adata.obs", sample_key)
            anndata_fields.append(CategoricalObsField("sample", sample_key))

        if condition_key is not None:
            logger.info("Conditions: Using '%s' from adata.obs", condition_key)
            anndata_fields.append(CategoricalObsField("condition", condition_key))

        if donor_key is not None:
            logger.info("Donors: Using '%s' from adata.obs", donor_key)
            anndata_fields.append(CategoricalObsField("donor", donor_key))

        logger.info("--- Product of Experts (PoE) Configuration ---")
        if label_key is not None:
            logger.info("Will use: Label-based PoE")
        else:
            logger.info("No labels configured — provide label_key for PoE-based integration")

        if modality_likelihoods is not None:
            adata.uns["modality_likelihoods"] = modality_likelihoods
            logger.info("Multimodal: Configured with likelihoods %s", modality_likelihoods)

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
        self,
        group_indices_list: Optional[list[list[int]]] = None,
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
            If ``None``, values are inferred from ``adata.uns['groups_obs_indices']``.
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
        group_indices_list, inferred = resolve_group_indices_list(adata, group_indices_list)
        if inferred:
            self._warn_group_indices_auto_inferred("get_latent_representation")

        n_groups = len(group_indices_list)
        n_per_group = [len(group) for group in group_indices_list]

        # Automatically determine drop_last
        if drop_last is None:
            drop_last = False
        if batch_size is None:
            batch_size = settings.batch_size

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

    @torch.no_grad()
    def embed(
        self,
        group_indices_list: Optional[list[list[int]]] = None,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
        prefix: str = "spvm",
        overwrite: bool = False,
        normalized: bool = False,
        give_mean: bool = True,
        mc_samples: int = 5000,
    ) -> dict[str, object]:
        """Compute and store embeddings in one call.

        Parameters
        ----------
        group_indices_list
            Optional per-group cell indices. If ``None``, inferred from
            ``adata.uns['groups_obs_indices']``.
        adata
            AnnData to write into. Defaults to the model's registered AnnData.
        batch_size
            Minibatch size passed to :meth:`get_latent_representation`.
        prefix
            Key prefix for outputs in ``adata.obsm``. Keys follow
            ``X_{prefix}_shared`` and ``X_{prefix}_private_{group}``.
        overwrite
            If ``False``, fail when any target key already exists.
        normalized
            Whether to return normalized embeddings.
        give_mean
            Whether to return posterior means when ``normalized=True``.
        mc_samples
            Monte Carlo samples for normalized means.

        Returns
        -------
        dict[str, object]
            Dictionary containing written keys and arrays:

            - ``keys``: mapping of shared/private output keys
            - ``shared``: shared embedding array (all cells)
            - ``private``: dict of private embedding arrays by group token
        """
        adata = self._validate_anndata(adata)
        group_indices_list, inferred = resolve_group_indices_list(adata, group_indices_list)
        if inferred:
            self._warn_group_indices_auto_inferred("embed")

        clean_prefix = sanitize_obsm_token(prefix)
        shared_key = f"X_{clean_prefix}_shared"

        group_mapping = adata.uns.get("groups_mapping", {})
        private_keys: dict[str, str] = {}
        group_tokens: list[str] = []
        seen_tokens: set[str] = set()
        for gi in range(len(group_indices_list)):
            mapped = group_mapping.get(gi, gi) if isinstance(group_mapping, dict) else gi
            token = sanitize_obsm_token(mapped)
            if token in seen_tokens:
                token = f"{token}_g{gi}"
            seen_tokens.add(token)
            group_tokens.append(token)
            key = f"X_{clean_prefix}_private_{token}"
            private_keys[token] = key

        target_keys = [shared_key, *private_keys.values()]
        existing = [k for k in target_keys if k in adata.obsm]
        if existing and not overwrite:
            raise ValueError(
                "Refusing to overwrite existing adata.obsm keys: "
                f"{existing}. Set overwrite=True to replace them."
            )

        latents = self.get_latent_representation(
            group_indices_list=group_indices_list,
            adata=adata,
            normalized=normalized,
            give_mean=give_mean,
            mc_samples=mc_samples,
            batch_size=batch_size,
        )

        n_obs = adata.n_obs
        shared_sample = next(iter(latents["shared_reordered"].values()))
        shared_out = np.zeros((n_obs, shared_sample.shape[1]), dtype=np.float32)
        for gi, idxs in enumerate(group_indices_list):
            shared_out[np.asarray(idxs)] = latents["shared_reordered"][gi]
        adata.obsm[shared_key] = shared_out

        private_out: dict[str, np.ndarray] = {}
        for gi, idxs in enumerate(group_indices_list):
            token = group_tokens[gi]
            key = private_keys[token]
            arr = latents["private_reordered"][gi]
            out = np.zeros((n_obs, arr.shape[1]), dtype=np.float32)
            out[np.asarray(idxs)] = arr
            adata.obsm[key] = out
            private_out[token] = out

        return {
            "keys": {
                "shared": shared_key,
                "private": private_keys,
            },
            "shared": shared_out,
            "private": private_out,
        }

    @torch.no_grad()
    def get_shared_posterior(
        self,
        group_indices_list: Optional[list[list[int]]] = None,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
        drop_last: Optional[bool] = None,
    ) -> dict[str, object]:
        """Return shared posterior parameters for each group.

        Parameters
        ----------
        group_indices_list
            Per-group cell indices. If ``None``, inferred from
            ``adata.uns['groups_obs_indices']``.
        adata
            AnnData object with compatible structure. Defaults to model AnnData.
        batch_size
            Minibatch size for loading cells.
        drop_last
            Whether to drop the last incomplete batch.

        Returns
        -------
        dict[str, object]
            Dictionary with per-group arrays in original and reordered cell order:

            - ``loc`` / ``scale``
            - ``loc_reordered`` / ``scale_reordered``
            - ``group_indices_list``
        """
        adata = self._validate_anndata(adata)
        group_indices_list, inferred = resolve_group_indices_list(adata, group_indices_list)
        if inferred:
            self._warn_group_indices_auto_inferred("get_shared_posterior")

        latents = self.get_latent_representation(
            group_indices_list=group_indices_list,
            adata=adata,
            normalized=False,
            give_mean=True,
            mc_samples=1,
            batch_size=batch_size,
            drop_last=drop_last,
        )

        return {
            "loc": latents["shared_posterior_loc"],
            "scale": latents["shared_posterior_scale"],
            "loc_reordered": latents["shared_posterior_loc_reordered"],
            "scale_reordered": latents["shared_posterior_scale_reordered"],
            "group_indices_list": group_indices_list,
        }

    def _aggregate_shared_posterior(
        self,
        adata: AnnData,
        shared_posterior: dict[str, object],
        sample_subset: Optional[Sequence[str]] = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Aggregate shared posterior by (group, sample)."""
        group_indices_list = shared_posterior["group_indices_list"]
        loc_reordered = shared_posterior["loc_reordered"]
        scale_reordered = shared_posterior["scale_reordered"]

        group_mapping = adata.uns.get("groups_mapping", {})
        sample_subset_norm = {str(s) for s in sample_subset} if sample_subset is not None else None

        used_fallback = "sample" not in adata.obs
        if used_fallback:
            warnings.warn(
                "sample_key is not registered; aggregating with one synthetic sample per group.",
                UserWarning,
                stacklevel=2,
            )

        records: list[dict[str, object]] = []
        for gi, idxs in enumerate(group_indices_list):
            idxs_arr = np.asarray(idxs)
            group_name = group_mapping.get(gi, gi) if isinstance(group_mapping, dict) else gi

            if used_fallback:
                sample_labels = np.array([f"group_{gi}"] * len(idxs_arr), dtype=object)
            else:
                sample_labels = adata.obs.iloc[idxs_arr]["sample"].astype(str).to_numpy()

            unique_samples = pd.unique(sample_labels)
            for sample_id in unique_samples:
                if sample_subset_norm is not None and str(sample_id) not in sample_subset_norm:
                    continue
                mask = sample_labels == sample_id
                if not np.any(mask):
                    continue

                loc_vals = np.asarray(loc_reordered[gi])[mask]
                scale_vals = np.asarray(scale_reordered[gi])[mask]
                agg_loc = loc_vals.mean(axis=0).astype(np.float32)
                agg_scale = np.sqrt(np.mean(np.square(scale_vals), axis=0)).astype(np.float32)

                records.append(
                    {
                        "group_idx": gi,
                        "group": str(group_name),
                        "sample": str(sample_id),
                        "n_cells": int(mask.sum()),
                        "loc": agg_loc,
                        "scale": agg_scale,
                    }
                )

        aggregated = pd.DataFrame.from_records(records)
        metadata = {
            "used_group_fallback": used_fallback,
            "sample_subset": sorted(sample_subset_norm) if sample_subset_norm is not None else None,
            "n_groups": len(group_indices_list),
        }
        return aggregated, metadata

    @torch.no_grad()
    def get_aggregated_posterior(
        self,
        group_indices_list: Optional[list[list[int]]] = None,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
        sample_subset: Optional[Sequence[str]] = None,
    ) -> dict[str, object]:
        """Aggregate shared posterior by sample within each group.

        Parameters
        ----------
        group_indices_list
            Per-group cell indices. If ``None``, inferred from
            ``adata.uns['groups_obs_indices']``.
        adata
            AnnData object with compatible structure. Defaults to model AnnData.
        batch_size
            Minibatch size for posterior extraction.
        sample_subset
            Optional list of sample identifiers to keep.

        Returns
        -------
        dict[str, object]
            - ``posterior``: DataFrame with one row per (group, sample)
            - ``metadata``: aggregation metadata
        """
        adata = self._validate_anndata(adata)
        shared = self.get_shared_posterior(
            group_indices_list=group_indices_list,
            adata=adata,
            batch_size=batch_size,
        )
        posterior_df, metadata = self._aggregate_shared_posterior(
            adata=adata,
            shared_posterior=shared,
            sample_subset=sample_subset,
        )
        return {"posterior": posterior_df, "metadata": metadata}

    @torch.no_grad()
    def differential_abundance(
        self,
        group_a: Optional[int] = None,
        group_b: Optional[int] = None,
        group_indices_list: Optional[list[list[int]]] = None,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
        sample_subset: Optional[Sequence[str]] = None,
    ) -> dict[str, object]:
        """Compute per-cell differential abundance score in shared latent space.

        Scores are computed as the signed difference in squared standardized
        distance to aggregated group posteriors:
        ``score = d(group_a) - d(group_b)``.
        Positive values indicate greater proximity to ``group_b``.

        Parameters
        ----------
        group_a
            Reference group index. If ``None`` and two groups are present, uses 0.
        group_b
            Comparison group index. If ``None`` and two groups are present, uses 1.
        group_indices_list
            Per-group cell indices. If ``None``, inferred from
            ``adata.uns['groups_obs_indices']``.
        adata
            AnnData object with compatible structure. Defaults to model AnnData.
        batch_size
            Minibatch size for posterior extraction.
        sample_subset
            Optional list of sample identifiers to keep when aggregating.

        Returns
        -------
        dict[str, object]
            - ``scores``: per-cell DA DataFrame (index: ``adata.obs_names``)
            - ``metadata``: comparison metadata and aggregation details
        """
        adata = self._validate_anndata(adata)
        shared = self.get_shared_posterior(
            group_indices_list=group_indices_list,
            adata=adata,
            batch_size=batch_size,
        )
        group_indices_list = shared["group_indices_list"]
        n_groups = len(group_indices_list)

        if group_a is None or group_b is None:
            if n_groups != 2:
                raise ValueError("group_a and group_b must be provided when n_groups != 2.")
            group_a, group_b = 0, 1

        if not (0 <= int(group_a) < n_groups and 0 <= int(group_b) < n_groups):
            raise ValueError(
                f"Invalid group comparison ({group_a}, {group_b}) for n_groups={n_groups}."
            )

        if (
            float(getattr(self.module, "disentangle_group_shared_weight", 0.0)) == 0.0
            and not bool(getattr(self.module, "use_jeffreys_integ", False))
        ):
            warnings.warn(
                "differential_abundance() is running without explicit shared-latent alignment "
                "(disentangle_group_shared_weight=0 and use_jeffreys_integ=False); results may be biased.",
                UserWarning,
                stacklevel=2,
            )

        aggregated_df, aggregation_meta = self._aggregate_shared_posterior(
            adata=adata,
            shared_posterior=shared,
            sample_subset=sample_subset,
        )
        if aggregated_df.empty:
            raise ValueError("No aggregated posterior entries were produced for the requested sample subset.")

        agg_a = aggregated_df[aggregated_df["group_idx"] == int(group_a)]
        agg_b = aggregated_df[aggregated_df["group_idx"] == int(group_b)]
        if agg_a.empty or agg_b.empty:
            raise ValueError(
                "At least one comparison group has no aggregated entries after filtering. "
                "Adjust sample_subset or comparison groups."
            )

        mu_a = np.stack(agg_a["loc"].to_list(), axis=0).mean(axis=0)
        mu_b = np.stack(agg_b["loc"].to_list(), axis=0).mean(axis=0)
        scale_a = np.sqrt(np.mean(np.square(np.stack(agg_a["scale"].to_list(), axis=0)), axis=0))
        scale_b = np.sqrt(np.mean(np.square(np.stack(agg_b["scale"].to_list(), axis=0)), axis=0))
        scale_a = np.clip(scale_a, 1e-6, None)
        scale_b = np.clip(scale_b, 1e-6, None)

        score_values = np.zeros(adata.n_obs, dtype=np.float32)
        for gi, idxs in enumerate(group_indices_list):
            idxs_arr = np.asarray(idxs)
            z = np.asarray(shared["loc_reordered"][gi])
            d_a = np.sum(np.square((z - mu_a) / scale_a), axis=1)
            d_b = np.sum(np.square((z - mu_b) / scale_b), axis=1)
            score_values[idxs_arr] = (d_a - d_b).astype(np.float32)

        group_mapping = adata.uns.get("groups_mapping", {})
        group_label_by_idx = {
            gi: str(group_mapping.get(gi, gi)) if isinstance(group_mapping, dict) else str(gi)
            for gi in range(n_groups)
        }
        cell_group = np.empty(adata.n_obs, dtype=object)
        for gi, idxs in enumerate(group_indices_list):
            cell_group[np.asarray(idxs)] = group_label_by_idx[gi]

        scores_df = pd.DataFrame(
            {
                "da_score": score_values,
                "group": cell_group,
            },
            index=adata.obs_names,
        )

        metadata = {
            "group_a": int(group_a),
            "group_b": int(group_b),
            "group_a_name": group_label_by_idx[int(group_a)],
            "group_b_name": group_label_by_idx[int(group_b)],
            "aggregation": aggregation_meta,
        }
        return {"scores": scores_df, "metadata": metadata}

    def get_enrichment_scores(
        self,
        network: pd.DataFrame,
        *,
        adata: Optional[AnnData] = None,
        methods: Optional[Sequence[str]] = None,
        source_col: str = "source",
        target_col: str = "target",
        weight_col: str = "weight",
        tmin: int = 5,
        write_to_adata: bool = True,
        obsm_key: str = "X_spvm_enrichment",
        uns_key: str = "spvm_enrichment",
        overwrite: bool = False,
        verbose: bool = False,
    ) -> dict[str, object]:
        """Run decoupler enrichment methods and return per-cell activity scores.

        Parameters
        ----------
        network
            Long-format network with source/target columns and optional weights.
        adata
            AnnData to score. Defaults to model-registered AnnData.
        methods
            Iterable of decoupler methods to run. Defaults to
            ``("ora", "gsea", "ulm")``.
        source_col
            Network column containing source/program names.
        target_col
            Network column containing target feature names.
        weight_col
            Optional network weight column.
        tmin
            Minimum number of targets per source.
        write_to_adata
            If ``True``, write combined scores to ``adata.obsm`` and provenance
            metadata to ``adata.uns``.
        obsm_key
            Destination ``adata.obsm`` key for combined scores.
        uns_key
            Destination ``adata.uns`` key for enrichment metadata.
        overwrite
            If ``False``, fail when destination keys already exist.
        verbose
            Forwarded to decoupler method calls.

        Returns
        -------
        dict[str, object]
            Dictionary with:

            - ``scores_df``: combined scores DataFrame indexed by ``adata.obs_names``
            - ``metadata``: execution metadata and overlap diagnostics
        """
        # Enrichment scoring uses gene expression directly (via decoupler) and
        # does NOT require model-setup transfer. Accept any AnnData with matching
        # cells, defaulting to the registered AnnData when none is supplied.
        if adata is None:
            adata = self.adata

        method_list = [m.lower() for m in (methods or ("ora", "gsea", "ulm"))]
        allowed_methods = {"ora", "gsea", "ulm"}
        invalid = sorted(set(method_list) - allowed_methods)
        if invalid:
            raise ValueError(
                f"Unsupported method(s): {invalid}. Supported methods: {sorted(allowed_methods)}."
            )

        normalized_network, network_stats = validate_enrichment_network(
            network,
            source_col=source_col,
            target_col=target_col,
            weight_col=weight_col,
            adata=adata,
            tmin=tmin,
        )
        # decoupler methods expect canonical long-table columns.
        rename_map = {source_col: "source", target_col: "target"}
        if weight_col in normalized_network.columns:
            rename_map[weight_col] = "weight"
        normalized_network = normalized_network.rename(columns=rename_map)
        for msg in network_stats.get("warnings", []):
            warnings.warn(msg, UserWarning, stacklevel=2)

        normalized_obsm_key = obsm_key
        if not normalized_obsm_key.startswith("X_"):
            normalized_obsm_key = f"X_{sanitize_obsm_token(normalized_obsm_key)}"
        normalized_uns_key = sanitize_obsm_token(uns_key)

        if write_to_adata:
            existing = []
            if normalized_obsm_key in adata.obsm:
                existing.append(normalized_obsm_key)
            if normalized_uns_key in adata.uns:
                existing.append(normalized_uns_key)
            if existing and not overwrite:
                raise ValueError(
                    "Refusing to overwrite existing AnnData keys: "
                    f"{existing}. Set overwrite=True to replace them."
                )

        try:
            import decoupler as dc
        except ImportError as err:
            raise ImportError(
                "decoupler is required for enrichment scoring. "
                "Install optional dependencies with: pip install -e .[enrichment]"
            ) from err

        per_method_scores: list[pd.DataFrame] = []
        method_meta: dict[str, dict[str, object]] = {}
        for method in method_list:
            run_fn = getattr(dc.mt, method, None)
            if run_fn is None:
                raise ValueError(
                    f"decoupler.mt.{method} is unavailable in the installed decoupler version."
                )

            tmp = adata.copy()
            run_kwargs = {
                "data": tmp,
                "net": normalized_network,
                "tmin": tmin,
                "verbose": verbose,
            }
            run_fn(**run_kwargs)

            score_key = f"score_{method}"
            if hasattr(dc, "pp") and hasattr(dc.pp, "get_obsm"):
                score_adata = dc.pp.get_obsm(tmp, key=score_key)
                score_values = score_adata.X
                if hasattr(score_values, "toarray"):
                    score_values = score_values.toarray()
                score_df = pd.DataFrame(
                    np.asarray(score_values),
                    index=tmp.obs_names,
                    columns=[str(c) for c in score_adata.var_names],
                )
            elif score_key in tmp.obsm:
                score_values = tmp.obsm[score_key]
                if hasattr(score_values, "toarray"):
                    score_values = score_values.toarray()
                score_values = np.asarray(score_values)
                if score_values.ndim == 1:
                    score_values = score_values[:, None]
                score_df = pd.DataFrame(
                    score_values,
                    index=tmp.obs_names,
                    columns=[f"{method}_{i}" for i in range(score_values.shape[1])],
                )
            else:
                raise RuntimeError(
                    f"decoupler did not write '{score_key}' to AnnData outputs."
                )

            score_df = score_df.add_prefix(f"{method}__")
            per_method_scores.append(score_df)
            method_meta[method] = {
                "score_key": score_key,
                "n_programs": int(score_df.shape[1]),
            }

        scores_df = pd.concat(per_method_scores, axis=1)
        metadata = {
            "methods": method_list,
            "network_stats": network_stats,
            "method_stats": method_meta,
            "storage_keys": {
                "obsm": normalized_obsm_key,
                "uns": normalized_uns_key,
            },
            "warnings": list(network_stats.get("warnings", [])),
        }

        if write_to_adata:
            adata.obsm[normalized_obsm_key] = scores_df.to_numpy(dtype=np.float32, copy=False)
            adata.uns[normalized_uns_key] = {
                **metadata,
                "columns": scores_df.columns.tolist(),
            }

        return {
            "scores_df": scores_df,
            "metadata": metadata,
        }

    def summarize_enrichment(
        self,
        scores_df: pd.DataFrame,
        groupby: str,
        *,
        adata: Optional[AnnData] = None,
        agg: str = "mean",
    ) -> pd.DataFrame:
        """Aggregate enrichment scores by an ``adata.obs`` grouping column.

        Parameters
        ----------
        scores_df
            DataFrame returned by :meth:`get_enrichment_scores`.
        groupby
            Column in ``adata.obs`` used to aggregate scores.
        adata
            AnnData carrying the grouping column. Defaults to registered AnnData.
        agg
            Aggregation method passed to pandas groupby (for example ``mean``).

        Returns
        -------
        pd.DataFrame
            Aggregated scores with one row per group.
        """
        if not isinstance(scores_df, pd.DataFrame):
            raise TypeError(
                f"scores_df must be a pandas DataFrame, got {type(scores_df).__name__}."
            )
        adata = self._validate_anndata(adata)
        if groupby not in adata.obs:
            raise KeyError(
                f"'{groupby}' not found in adata.obs. Available columns: {list(adata.obs.columns)}"
            )
        if not scores_df.index.equals(adata.obs_names):
            raise ValueError(
                "scores_df index must exactly match adata.obs_names for aggregation."
            )
        merged = scores_df.copy()
        merged[groupby] = adata.obs[groupby].astype(str).values
        return merged.groupby(groupby, observed=True).agg(agg)

    def interpretation_report(
        self,
        scores_df: pd.DataFrame,
        groupby: str,
        *,
        adata: Optional[AnnData] = None,
        agg: str = "mean",
        top_n: int = 5,
        z_shared_key: str = "X_spvm_shared",
        label_key: Optional[str] = None,
        k: int = 20,
        leiden_resolution: float = 0.8,
    ) -> dict[str, object]:
        """Create a compact interpretation summary for enrichment outputs.

        Parameters
        ----------
        scores_df
            Enrichment score matrix from :meth:`get_enrichment_scores`.
        groupby
            Categorical column in ``adata.obs`` for aggregation.
        adata
            AnnData carrying group/label metadata and optional shared latent.
        agg
            Aggregation method for enrichment summaries.
        top_n
            Number of top programs (by absolute score) to report per group.
        z_shared_key
            Key in ``adata.obsm`` containing shared latent coordinates used for
            optional integration metrics.
        label_key
            Optional label column in ``adata.obs``. When provided alongside
            ``adata.obs['groups']`` and ``z_shared_key``, integration metrics are computed.
        k
            k-neighbourhood size passed to integration metrics.
        leiden_resolution
            Leiden resolution for ARI metric.

        Returns
        -------
        dict[str, object]
            Dictionary with keys:

            - ``enrichment_summary``: aggregated enrichment table
            - ``top_programs``: per-group top program names
            - ``integration_metrics``: optional integration metric DataFrame
            - ``warnings``: informational warnings emitted by this helper
        """
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}.")

        adata = self._validate_anndata(adata)
        enrichment_summary = self.summarize_enrichment(
            scores_df,
            groupby,
            adata=adata,
            agg=agg,
        )

        top_rows: list[dict[str, object]] = []
        for group_name, row in enrichment_summary.iterrows():
            top_programs = row.abs().nlargest(min(top_n, row.shape[0])).index.tolist()
            top_rows.append(
                {
                    groupby: group_name,
                    "top_programs": top_programs,
                }
            )
        top_programs_df = pd.DataFrame(top_rows)

        integration_metrics = None
        report_warnings: list[str] = []
        can_compute_metrics = (
            z_shared_key in adata.obsm
            and label_key is not None
            and label_key in adata.obs
            and "groups" in adata.obs
        )
        if can_compute_metrics:
            from spVIPESmulti.metrics import integration_report

            integration_metrics = integration_report(
                np.asarray(adata.obsm[z_shared_key]),
                adata.obs["groups"].values,
                adata.obs[label_key].values,
                k=k,
                leiden_resolution=leiden_resolution,
            )
        else:
            report_warnings.append(
                "Integration metrics were not computed. Provide z_shared_key in adata.obsm, "
                "label_key in adata.obs, and ensure adata.obs['groups'] exists."
            )

        return {
            "enrichment_summary": enrichment_summary,
            "top_programs": top_programs_df,
            "integration_metrics": integration_metrics,
            "warnings": report_warnings,
        }

    def evaluate(
        self,
        *,
        adata: Optional[AnnData] = None,
        group_indices_list: Optional[list[list[int]]] = None,
        batch_size: Optional[int] = None,
        label_key: Optional[str] = None,
        z_shared_key: Optional[str] = None,
        k: int = 20,
        leiden_resolution: float = 0.8,
        include_private: bool = False,
    ) -> dict[str, object]:
        """Compute diagnostic evaluation metrics for a trained model.

        Returns integration-quality diagnostics on the shared (and optionally
        private) latent space without any training or loss-function changes.
        When validation metrics are present in training history, the latest
        values are returned under ``held_out_metrics``.

        Parameters
        ----------
        adata
            AnnData to evaluate. Defaults to the model-registered AnnData.
        group_indices_list
            Per-group cell indices. If ``None``, inferred from
            ``adata.uns['groups_obs_indices']``.
        batch_size
            Minibatch size for latent extraction.
        label_key
            Column in ``adata.obs`` containing cell-type labels.  Required for
            label-preservation metrics (cLISI, kNN-purity, Leiden ARI).
            If absent, those columns will be ``nan``.
        z_shared_key
            Key in ``adata.obsm`` to read pre-computed shared embeddings from.
            If ``None`` or not present, embeddings are extracted fresh from the
            model (may be slow for large datasets).
        k
            Neighbourhood size for kNN-based metrics.
        leiden_resolution
            Resolution for Leiden clustering used in ARI computation.
        include_private
            If ``True``, also compute per-group silhouette on each private
            latent space and include those rows in the returned DataFrame.

        Returns
        -------
        dict[str, object]
            Dictionary with keys:

            - ``metrics``: :class:`pandas.DataFrame` with one row per
              evaluated latent space (columns: ``latent``, ``ilisi``,
              ``clisi``, ``kbet``, ``knn_purity``, ``leiden_ari``,
              ``silhouette``).
              ``kbet`` is a rejection rate, so lower values indicate better
              group mixing.
            - ``metadata``: dictionary of evaluation configuration values
              (``n_cells``, ``n_groups``, ``k``, ``label_key``,
              ``leiden_resolution``, ``include_private``,
              ``used_precomputed_embedding``).
            - ``held_out_metrics``: latest validation metrics from training
              history when available on the model's registered AnnData.
            - ``warnings``: list of informational warning strings emitted
              during evaluation.
        """
        from spVIPESmulti.metrics import integration_report

        adata = self._validate_anndata(adata)
        group_indices_list, inferred = resolve_group_indices_list(adata, group_indices_list)
        if inferred:
            self._warn_group_indices_auto_inferred("evaluate")

        eval_warnings: list[str] = []

        # ── Shared latent ────────────────────────────────────────────────────
        used_precomputed = False
        if z_shared_key is not None and z_shared_key in adata.obsm:
            z_shared_full = np.asarray(adata.obsm[z_shared_key])
            used_precomputed = True
        else:
            if z_shared_key is not None:
                eval_warnings.append(
                    f"z_shared_key='{z_shared_key}' not found in adata.obsm; "
                    "extracting embeddings from model instead."
                )
            latents = self.get_latent_representation(
                group_indices_list=group_indices_list,
                adata=adata,
                normalized=False,
                give_mean=True,
                mc_samples=1,
                batch_size=batch_size,
            )
            n_obs = adata.n_obs
            shared_sample = next(iter(latents["shared_reordered"].values()))
            z_shared_full = np.zeros((n_obs, shared_sample.shape[1]), dtype=np.float32)
            for gi, idxs in enumerate(group_indices_list):
                z_shared_full[np.asarray(idxs)] = latents["shared_reordered"][gi]

        # ── Group labels ─────────────────────────────────────────────────────
        if "groups" in adata.obs:
            group_labels = adata.obs["groups"].values
        else:
            group_labels = np.empty(adata.n_obs, dtype=object)
            for gi, idxs in enumerate(group_indices_list):
                group_labels[np.asarray(idxs)] = str(gi)
            eval_warnings.append(
                "adata.obs['groups'] not found; using inferred integer group labels."
            )

        # ── Cell-type labels ─────────────────────────────────────────────────
        if label_key is not None and label_key in adata.obs:
            cell_labels = adata.obs[label_key].values
        else:
            if label_key is not None:
                eval_warnings.append(
                    f"label_key='{label_key}' not found in adata.obs; "
                    "label-dependent metrics (clisi, knn_purity, leiden_ari) will be nan."
                )
            # Dummy labels: all cells in the same class → clisi/purity/ARI are meaningless
            cell_labels = np.array(["unknown"] * adata.n_obs, dtype=object)

        # ── Private latents ──────────────────────────────────────────────────
        z_private_dict: Optional[dict[str, np.ndarray]] = None
        if include_private:
            if not used_precomputed:
                # latents was already computed above
                group_mapping = adata.uns.get("groups_mapping", {})
                z_private_dict = {}
                for gi, idxs in enumerate(group_indices_list):
                    token = (
                        str(group_mapping.get(gi, gi))
                        if isinstance(group_mapping, dict)
                        else str(gi)
                    )
                    z_private_dict[token] = latents["private_reordered"][gi]
            else:
                # Need a fresh extraction for private latents
                latents = self.get_latent_representation(
                    group_indices_list=group_indices_list,
                    adata=adata,
                    normalized=False,
                    give_mean=True,
                    mc_samples=1,
                    batch_size=batch_size,
                )
                group_mapping = adata.uns.get("groups_mapping", {})
                z_private_dict = {}
                for gi, idxs in enumerate(group_indices_list):
                    token = (
                        str(group_mapping.get(gi, gi))
                        if isinstance(group_mapping, dict)
                        else str(gi)
                    )
                    z_private_dict[token] = latents["private_reordered"][gi]

        metrics_df = integration_report(
            z_shared_full,
            group_labels,
            cell_labels,
            z_private_dict=z_private_dict,
            k=k,
            leiden_resolution=leiden_resolution,
        )

        metadata = {
            "n_cells": int(adata.n_obs),
            "n_groups": len(group_indices_list),
            "k": k,
            "label_key": label_key,
            "leiden_resolution": leiden_resolution,
            "include_private": include_private,
            "used_precomputed_embedding": used_precomputed,
        }

        held_out_metrics = None
        if adata is self.adata:
            history = getattr(self, "history", None)
            validation_keys = (
                "elbo_validation",
                "reconstruction_loss_validation",
                "kl_local_validation",
                "validation_loss",
            )
            if history is not None and any(key in history for key in validation_keys):
                held_out_metrics = {}
                for key in validation_keys:
                    if key not in history or len(history[key]) == 0:
                        continue
                    value = history[key].iloc[-1]
                    try:
                        value = float(value.iloc[0]) if hasattr(value, "iloc") else float(value)
                    except Exception:
                        value = float(np.asarray(value).ravel()[0])
                    held_out_metrics[key] = value
                if "reconstruction_loss_validation" in held_out_metrics:
                    held_out_metrics["held_out_nll"] = held_out_metrics["reconstruction_loss_validation"]
            elif getattr(self, "validation_indices", None) is not None:
                eval_warnings.append(
                    "Validation indices are present but validation metrics were not found in model.history. "
                    "Re-train the model after enabling validation logging or use the current model only for latent diagnostics."
                )
        elif getattr(self, "validation_indices", None) is not None:
            eval_warnings.append(
                "Held-out metrics are only available for the model's registered AnnData because they are sourced from training history."
            )

        return {
            "metrics": metrics_df,
            "metadata": metadata,
            "held_out_metrics": held_out_metrics,
            "warnings": eval_warnings,
        }

    def _process_batches(self, dataloader, normalized, give_mean, mc_samples, n_groups=None):
        """Process batches and return intermediate results for N groups."""
        # `@torch.no_grad()` only disables autograd; it does NOT flip
        # `nn.Module.training` to False. Without `eval()`, BatchNorm running
        # stats keep updating and Dropout keeps dropping during inference,
        # silently mutating the model and adding noise to embeddings.
        was_training = self.module.training
        self.module.eval()
        try:
            return self._process_batches_impl(
                dataloader, normalized, give_mean, mc_samples, n_groups
            )
        finally:
            self.module.train(was_training)

    def _process_batches_impl(self, dataloader, normalized, give_mean, mc_samples, n_groups=None):
        """Implementation of batch processing; assumes module is in eval mode."""
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
        shared_posterior_loc = {g: [] for g in range(n_groups)}
        shared_posterior_scale = {g: [] for g in range(n_groups)}
        original_indices = {g: [] for g in range(n_groups)}

        # For multimodal: per-(group, modality) private latents
        is_multimodal = self.module.is_multimodal
        latent_private_multimodal = {} if is_multimodal else None
        if is_multimodal:
            for g in range(n_groups):
                for mod in self.module.group_modalities[g]:
                    latent_private_multimodal[(g, mod)] = []

        for tensors_by_group in dataloader:
            # _get_inference_input already splits tensors by group internally; reuse
            # its extracted global_indices rather than calling _split_tensors_by_group
            # a second time per batch.
            inference_inputs = self.module._get_inference_input(tensors_by_group)
            outputs = self.module.inference(**inference_inputs)

            for g in range(n_groups):
                # Shared (PoE) latent
                poe_log_z = outputs["poe_stats"][g]["logtheta_log_z"]
                shared_posterior_loc[g].append(outputs["poe_stats"][g]["logtheta_loc"].cpu())
                shared_posterior_scale[g].append(outputs["poe_stats"][g]["logtheta_scale"].cpu())
                if not normalized:
                    latent_shared[g].append(poe_log_z.cpu())
                else:
                    qz_poe = outputs["poe_stats"][g]["logtheta_qz"]
                    if give_mean:
                        samples = qz_poe.sample([mc_samples])
                        theta = torch.nn.functional.softmax(samples, dim=-1).mean(dim=0)
                    else:
                        theta = outputs["poe_stats"][g]["logtheta_theta"]
                    latent_shared[g].append(theta.cpu())

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

                _idx = inference_inputs["global_indices"][g]
                original_indices[g].append(_idx.cpu() if _idx is not None else torch.arange(poe_log_z.shape[0]))

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
            "shared_posterior_loc": shared_posterior_loc,
            "shared_posterior_scale": shared_posterior_scale,
            "original_indices": original_indices,
        }
        if is_multimodal:
            result["latent_private_multimodal"] = latent_private_multimodal
        return result

    def _format_results(self, results, n_per_group):
        """Format the final results dictionary for N groups."""
        n_groups = len(n_per_group)

        latent_private = {}
        latent_shared = {}
        latent_private_reordered = {}
        latent_shared_reordered = {}
        shared_posterior_loc = {}
        shared_posterior_scale = {}
        shared_posterior_loc_reordered = {}
        shared_posterior_scale_reordered = {}

        for g in range(n_groups):
            n_g = n_per_group[g]
            g_private = torch.cat(results["latent_private"][g]).numpy()[:n_g]
            g_shared = torch.cat(results["latent_shared"][g]).numpy()[:n_g]
            g_shared_loc = torch.cat(results["shared_posterior_loc"][g]).numpy()[:n_g]
            g_shared_scale = torch.cat(results["shared_posterior_scale"][g]).numpy()[:n_g]
            g_indices = torch.cat(results["original_indices"][g]).numpy().flatten()[:n_g]
            sort_idx = np.argsort(g_indices)

            latent_private[g] = g_private
            latent_shared[g] = g_shared
            latent_private_reordered[g] = g_private[sort_idx]
            latent_shared_reordered[g] = g_shared[sort_idx]
            shared_posterior_loc[g] = g_shared_loc
            shared_posterior_scale[g] = g_shared_scale
            shared_posterior_loc_reordered[g] = g_shared_loc[sort_idx]
            shared_posterior_scale_reordered[g] = g_shared_scale[sort_idx]

        output = {
            "shared": latent_shared,
            "private": latent_private,
            "shared_reordered": latent_shared_reordered,
            "private_reordered": latent_private_reordered,
            "shared_posterior_loc": shared_posterior_loc,
            "shared_posterior_scale": shared_posterior_scale,
            "shared_posterior_loc_reordered": shared_posterior_loc_reordered,
            "shared_posterior_scale_reordered": shared_posterior_scale_reordered,
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

        For single-modal models the returned dict is keyed by ``(group_index, latent_type)``
        where ``latent_type`` is ``"private"`` or ``"shared"``.

        For multimodal models the returned dict is keyed by
        ``((group_index, modality), latent_type)``.

        Shape of each array is ``(n_features, n_latent_dims)``.
        """
        adata = self.adata
        loadings_dict = {}

        if self.module.is_multimodal:
            for (group, modality) in self.module.decoders:
                mod_var_indices = self.module.groups_modality_var_indices[group][modality]
                var_names = adata[:, mod_var_indices].var_names
                cols_private = [f"Z_private_{n}" for n in range(self.module.n_dimensions_private)]
                cols_shared = [f"Z_shared_{n}" for n in range(self.module.n_dimensions_shared)]
                loadings_dict[((group, modality), "private")] = pd.DataFrame(
                    self.module.get_loadings((group, modality), "private"),
                    index=var_names,
                    columns=cols_private,
                )
                loadings_dict[((group, modality), "shared")] = pd.DataFrame(
                    self.module.get_loadings((group, modality), "shared"),
                    index=var_names,
                    columns=cols_shared,
                )
        else:
            num_datasets = len(self.module.input_dims)
            datasets_obs_indices = self.module.groups_obs_indices
            datasets_var_indices = self.module.groups_var_indices
            for i in range(num_datasets):
                dataset_obs_indices = datasets_obs_indices[i]
                s_adata = adata[dataset_obs_indices, :].copy()
                cols_private = [f"Z_private_{n}" for n in range(self.module.n_dimensions_private)]
                cols_shared = [f"Z_shared_{n}" for n in range(self.module.n_dimensions_shared)]
                var_names = s_adata[:, datasets_var_indices[i]].var_names
                loadings_dict[(i, "private")] = pd.DataFrame(
                    self.module.get_loadings(i, "private"), index=var_names, columns=cols_private
                )
                loadings_dict[(i, "shared")] = pd.DataFrame(
                    self.module.get_loadings(i, "shared"), index=var_names, columns=cols_shared
                )

        return loadings_dict

    def traverse_latent(
        self,
        adata=None,
        group_idx: int = 0,
        n_steps: int = 15,
        n_samples: int = 50,
        n_stds: float = 3.0,
        seed: int = 0,
    ) -> "pd.DataFrame":
        """Score genes by traversal of each z_shared dimension.

        Convenience wrapper around :func:`spVIPESmulti.traversal.traverse_latent`.
        See that function for full documentation.

        Parameters
        ----------
        adata:
            AnnData to use. Defaults to the model's registered AnnData.
        group_idx:
            Which group's decoder to use (0-based).
        n_steps:
            Number of traversal steps per dimension.
        n_samples:
            Number of cells to average over during traversal.
        n_stds:
            Traversal range: ±n_stds × empirical std of each dimension.
        seed:
            Random seed for cell sampling.

        Returns
        -------
        pd.DataFrame
            Shape ``(n_genes, n_dims_shared)``. Each entry is the traversal
            effect (max−min ``px_scale_shared``) of that dimension on that gene.

        Examples
        --------
        >>> trav = model.traverse_latent(group_idx=0, n_steps=15)
        >>> top = spVIPESmulti.traversal.calculate_differential_vars(trav)
        """
        from spVIPESmulti.traversal import traverse_latent as _traverse_latent

        return _traverse_latent(
            self,
            adata=adata,
            group_idx=group_idx,
            n_steps=n_steps,
            n_samples=n_samples,
            n_stds=n_stds,
            seed=seed,
        )
