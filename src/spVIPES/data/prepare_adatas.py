from typing import Optional, Union

import anndata as ad
import numpy as np
from scipy.sparse import issparse


def prepare_adatas(
    adatas: dict[str, ad.AnnData],
    layers: Optional[list[list[Union[str, None]]]] = None,
):
    """
    Prepare and concatenate multiple AnnData objects for spVIPES integration.

    This function takes multiple single-cell datasets and prepares them for
    multi-group integration by concatenating them into a single AnnData object
    while preserving group-specific metadata. It sets up all the necessary
    data structures for spVIPES to perform shared-private latent space learning.

    Parameters
    ----------
    adatas : dict[str, AnnData]
        Dictionary mapping group names (strings) to their corresponding AnnData objects.
        Each AnnData contains single-cell expression data for one group/dataset.
        Requires at least 2 groups.
    layers : list[list[str or None]], optional
        Specification of which layers to use from each AnnData object. Currently
        not implemented in the function body.

    Returns
    -------
    AnnData
        Concatenated AnnData object containing all groups with additional metadata:

        - **groups** : Added to `.obs` indicating which group each cell belongs to
        - **indices** : Added to `.obs` with within-group cell indices
        - **groups_var_indices** : In `.uns`, indices of variables for each group
        - **groups_obs_indices** : In `.uns`, indices of observations for each group
        - **groups_obs_names** : In `.uns`, observation names for each group
        - **groups_obs** : In `.uns`, observation metadata for each group
        - **groups_lengths** : In `.uns`, number of features per group
        - **groups_var_names** : In `.uns`, variable names for each group
        - **groups_mapping** : In `.uns`, mapping from indices to group names

    Raises
    ------
    ValueError
        If fewer than 2 groups are provided.

    Notes
    -----
    The function performs several important preprocessing steps:

    1. **Variable name prefixing**: Adds group prefixes to avoid name conflicts
    2. **Metadata harmonization**: Combines observation metadata across groups
    3. **Index tracking**: Creates mappings to track group-specific indices
    4. **Outer join concatenation**: Preserves all variables from all groups

    This prepared data structure enables spVIPES to handle datasets with different
    feature sets (genes) while maintaining the ability to separate shared and
    private latent representations.

    Examples
    --------
    Basic usage with two datasets:

    >>> import spVIPES
    >>> import scanpy as sc
    >>>
    >>> # Load your datasets
    >>> adata1 = sc.read_h5ad("dataset1.h5ad")
    >>> adata2 = sc.read_h5ad("dataset2.h5ad")
    >>>
    >>> # Prepare for spVIPES
    >>> adatas_dict = {"treatment": adata1, "control": adata2}
    >>> combined_adata = spVIPES.data.prepare_adatas(adatas_dict)
    >>>
    >>> # Now ready for spVIPES setup
    >>> spVIPES.model.spVIPES.setup_anndata(combined_adata, groups_key="groups")

    Integration with different feature sets:

    >>> # Datasets can have different genes
    >>> print(f"Dataset 1: {adata1.n_vars} genes")
    >>> print(f"Dataset 2: {adata2.n_vars} genes")
    >>>
    >>> combined = spVIPES.data.prepare_adatas({"batch1": adata1, "batch2": adata2})
    >>> print(f"Combined: {combined.n_vars} genes")  # Union of all genes
    """
    groups_obs_names = []
    groups_obs = {}
    groups_lengths = {}
    groups_var_names = {}  # Changed to dictionary
    groups_mapping = {}
    if len(adatas) < 2:
        raise ValueError("At least 2 groups are required")

    for i, (groups, adata) in enumerate(adatas.items()):
        if adata is not None:
            groups_lengths[i] = adata.shape[1]
            groups_obs_names.append(adata.obs_names)
            if groups_obs.get(groups, None) is None:
                groups_obs[groups] = adata.obs.copy()
                groups_obs[groups].loc[:, "group"] = groups

            else:
                cols_to_use = adata.obs.columns.difference(groups_obs[groups].columns)
                groups_obs[groups] = groups_obs[groups].join(adata.obs[cols_to_use])
            # Store var_names for each group
            adata.obs["groups"] = groups
            adata.var_names = f"{groups}_" + adata.var_names
            groups_var_names[groups] = adata.var_names
            groups_mapping[i] = groups

    multigroups_adata = ad.concat(adatas, join="outer", label="groups", index_unique="-")
    multigroups_adata.uns["groups_var_indices"] = [
        np.where(multigroups_adata.var_names.str.startswith(k))[0] for k in adatas.keys()
    ]
    multigroups_adata.uns["groups_obs_indices"] = [
        np.where(multigroups_adata.obs["groups"].str.startswith(k))[0] for k in adatas.keys()
    ]
    multigroups_adata.uns["groups_obs_names"] = groups_obs_names
    multigroups_adata.uns["groups_obs"] = groups_obs
    multigroups_adata.uns["groups_lengths"] = groups_lengths
    multigroups_adata.uns["groups_var_names"] = groups_var_names
    multigroups_adata.uns["groups_mapping"] = groups_mapping

    # Create indices column
    indices = []
    for _, group_indices in zip(adatas.keys(), multigroups_adata.uns["groups_obs_indices"]):
        group_size = len(group_indices)
        indices.extend(np.arange(group_size, dtype=np.int32))
    multigroups_adata.obs["indices"] = indices

    return multigroups_adata


def prepare_multimodal_adatas(
    adatas: dict[str, dict[str, ad.AnnData]],
    modality_likelihoods: Optional[dict[str, str]] = None,
):
    """
    Prepare and concatenate multimodal AnnData objects for spVIPES integration.

    Each group can have one or more modalities (e.g., RNA, protein, ATAC).
    All data is concatenated into a single AnnData with metadata tracking
    per-group, per-modality feature indices.

    Parameters
    ----------
    adatas : dict[str, dict[str, AnnData]]
        Nested dictionary: outer keys are group names, inner keys are modality names,
        values are AnnData objects. E.g.::

            {
                "treatment": {"rna": adata_rna_treat, "protein": adata_prot_treat},
                "control": {"rna": adata_rna_ctrl, "protein": adata_prot_ctrl},
            }

        Requires at least 2 groups. All groups must share at least one modality.
        Cells (observations) must be the same within each group across modalities.
    modality_likelihoods : dict[str, str], optional
        Mapping from modality name to likelihood type. Supported values:
        ``"nb"`` (NegativeBinomial for count data) and ``"gaussian"``
        (for log-normalized data). If ``None``, all modalities default to ``"nb"``.

    Returns
    -------
    AnnData
        Concatenated AnnData object with multimodal metadata in ``.uns``:

        - **is_multimodal** : ``True``
        - **modality_names** : list of modality names
        - **modality_likelihoods** : dict mapping modality → likelihood type
        - **groups_modality_lengths** : ``{group_idx: {modality: n_features}}``
        - **groups_modality_var_indices** : ``{group_idx: {modality: var_index_array}}``
        - **groups_lengths** : ``{group_idx: total_n_features}`` (sum across modalities)
        - **groups_var_indices** : list of var index arrays per group
        - **groups_obs_indices** : list of obs index arrays per group
        - **groups_obs_names** : list of obs names per group
        - **groups_var_names** : dict of var names per group
        - **groups_mapping** : dict mapping group index → group name

    Raises
    ------
    ValueError
        If fewer than 2 groups, or groups share no common modality.
    """
    if len(adatas) < 2:
        raise ValueError("At least 2 groups are required")

    group_names = list(adatas.keys())
    all_modalities = set()
    group_modalities = {}
    for group_name, mod_dict in adatas.items():
        mods = set(mod_dict.keys())
        if not mods:
            raise ValueError(f"Group '{group_name}' has no modalities")
        group_modalities[group_name] = mods
        all_modalities |= mods

    # Verify at least one shared modality
    shared_modalities = set.intersection(*group_modalities.values())
    if not shared_modalities:
        raise ValueError("Groups must share at least one modality")

    modality_names = sorted(all_modalities)

    # Set default likelihoods
    if modality_likelihoods is None:
        modality_likelihoods = {m: "nb" for m in modality_names}
    else:
        for m in modality_names:
            if m not in modality_likelihoods:
                modality_likelihoods[m] = "nb"

    # Validate likelihood values
    valid_likelihoods = {"nb", "gaussian"}
    for m, lk in modality_likelihoods.items():
        if lk not in valid_likelihoods:
            raise ValueError(f"Unsupported likelihood '{lk}' for modality '{m}'. Must be one of {valid_likelihoods}")

    # Build per-group combined AnnData (concatenating modalities along var axis)
    combined_adatas = {}
    groups_obs_names = []
    groups_mapping = {}
    groups_lengths = {}
    groups_var_names = {}
    groups_modality_lengths = {}
    # Track var name prefixes for modality-level indices after final concatenation
    group_modality_var_prefixes = {}

    for i, group_name in enumerate(group_names):
        groups_mapping[i] = group_name
        mod_dict = adatas[group_name]
        mod_adatas = []
        groups_modality_lengths[i] = {}
        group_modality_var_prefixes[i] = {}

        # Get the shared obs (cells) from first modality
        first_mod = next(iter(mod_dict.values()))
        groups_obs_names.append(first_mod.obs_names)

        for modality in modality_names:
            if modality not in mod_dict:
                continue
            mod_adata = mod_dict[modality].copy()
            n_features = mod_adata.n_vars
            groups_modality_lengths[i][modality] = n_features

            # Prefix var names: {group}_{modality}_{original_var_name}
            prefix = f"{group_name}_{modality}_"
            mod_adata.var_names = prefix + mod_adata.var_names
            group_modality_var_prefixes[i][modality] = prefix
            mod_adatas.append(mod_adata)

        # Concatenate modalities within this group (along var axis)
        if len(mod_adatas) == 1:
            group_adata = mod_adatas[0]
        else:
            group_adata = ad.concat(mod_adatas, axis=1, merge="same")

        group_adata.obs["groups"] = group_name
        groups_lengths[i] = group_adata.n_vars
        groups_var_names[group_name] = group_adata.var_names
        combined_adatas[group_name] = group_adata

    # Concatenate all groups (along obs axis, outer join on vars)
    multigroups_adata = ad.concat(combined_adatas, join="outer", label="groups", index_unique="-")

    # Compute group-level var and obs indices
    multigroups_adata.uns["groups_var_indices"] = [
        np.where(multigroups_adata.var_names.str.startswith(f"{group_names[i]}_"))[0]
        for i in range(len(group_names))
    ]
    multigroups_adata.uns["groups_obs_indices"] = [
        np.where(multigroups_adata.obs["groups"].str.startswith(group_names[i]))[0]
        for i in range(len(group_names))
    ]

    # Compute per-group, per-modality var indices in the concatenated adata
    groups_modality_var_indices = {}
    for i, group_name in enumerate(group_names):
        groups_modality_var_indices[i] = {}
        for modality in modality_names:
            prefix = group_modality_var_prefixes[i].get(modality)
            if prefix is not None:
                indices = np.where(multigroups_adata.var_names.str.startswith(prefix))[0]
                groups_modality_var_indices[i][modality] = indices

    # Store all metadata
    multigroups_adata.uns["is_multimodal"] = True
    multigroups_adata.uns["modality_names"] = modality_names
    multigroups_adata.uns["modality_likelihoods"] = modality_likelihoods
    multigroups_adata.uns["groups_modality_lengths"] = groups_modality_lengths
    multigroups_adata.uns["groups_modality_var_indices"] = groups_modality_var_indices
    multigroups_adata.uns["groups_obs_names"] = groups_obs_names
    multigroups_adata.uns["groups_lengths"] = groups_lengths
    multigroups_adata.uns["groups_var_names"] = groups_var_names
    multigroups_adata.uns["groups_mapping"] = groups_mapping

    # Create indices column (within-group cell indices)
    indices = []
    for _, group_indices in zip(group_names, multigroups_adata.uns["groups_obs_indices"]):
        group_size = len(group_indices)
        indices.extend(np.arange(group_size, dtype=np.int32))
    multigroups_adata.obs["indices"] = indices

    return multigroups_adata
