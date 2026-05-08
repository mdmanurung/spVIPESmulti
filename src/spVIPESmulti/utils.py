"""Utility functions for working with spVIPESmulti latent spaces.

These functions cover the post-training workflow that every tutorial notebook
repeats manually:

1. :func:`highly_variable_genes_union` — compute per-group HVGs and return
   their union, optionally subsetting the AnnData.
2. :func:`store_latents` — stitch per-group latents back into ``adata.obsm``
   in original cell order.
3. :func:`add_latent_dims_to_obs` — copy latent dimensions into ``adata.obs``
   so they can be used as ``color=`` arguments in scanpy plots.
4. :func:`compute_shared_umap` — run neighbours + UMAP on the shared latent
   and store the result under a named key.
5. :func:`compute_private_umaps` — same for per-group private latents.
6. :func:`get_top_genes` — rank genes by loading magnitude per latent dimension.
7. :func:`score_cells_on_factor` — project a single latent dimension into
   ``adata.obs``.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------


def highly_variable_genes_union(
    adata: "AnnData",
    group_key: str,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    subset: bool = True,
    **hvg_kwargs,
) -> "AnnData":
    """Select highly variable genes (HVGs) per group and return their union.

    Runs :func:`scanpy.pp.highly_variable_genes` independently on each group
    defined by ``group_key``, then takes the union of all per-group HVG sets.
    This avoids losing genes that are highly variable in only one condition.

    Parameters
    ----------
    adata:
        AnnData object. Must contain raw or normalised counts; make sure to
        apply the same preprocessing you would pass to ``sc.pp.highly_variable_genes``.
    group_key:
        Column in ``adata.obs`` that defines the groups (e.g. ``"antigen_specific"``).
    n_top_genes:
        Number of HVGs to select per group.
    flavor:
        HVG flavour passed to :func:`scanpy.pp.highly_variable_genes`
        (``"seurat_v3"``, ``"seurat"``, or ``"cell_ranger"``).
    subset:
        If ``True`` (default), return a copy of ``adata`` subsetted to the
        union gene set. If ``False``, add a ``highly_variable_union`` boolean
        column to ``adata.var`` and return the original (unsubsetted) object.
    **hvg_kwargs:
        Additional keyword arguments forwarded to
        :func:`scanpy.pp.highly_variable_genes`.

    Returns
    -------
    AnnData
        When ``subset=True``: a new AnnData containing only the union HVGs.
        When ``subset=False``: the original ``adata`` with
        ``adata.var["highly_variable_union"]`` added.

    Examples
    --------
    >>> import spVIPESmulti
    >>> adata = spVIPESmulti.utils.highly_variable_genes_union(
    ...     adata, group_key="antigen_specific", n_top_genes=2000
    ... )
    """
    import scanpy as sc

    if group_key not in adata.obs.columns:
        raise KeyError(
            f"'{group_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    import numpy as np
    import scipy.sparse as sp_sparse

    # Pull batch_key out of kwargs so we can handle it ourselves per (group, batch).
    # Passing batch_key into sc.pp.highly_variable_genes causes it to fit LOESS
    # independently per batch; if a batch has few cells or near-constant gene
    # expression the design matrix becomes ill-conditioned (ValueError: reciprocal
    # condition number). Computing HVG per (group × batch) independently avoids this.
    batch_key = hvg_kwargs.pop("batch_key", None)

    hvg_sets: list[set] = []
    for group in adata.obs[group_key].unique():
        adata_group = adata[adata.obs[group_key] == group]

        batches: list = (
            list(adata_group.obs[batch_key].unique())
            if batch_key is not None and batch_key in adata_group.obs.columns
            else [None]
        )

        group_hvg: set = set()
        for batch in batches:
            adata_b = (
                adata_group[adata_group.obs[batch_key] == batch].copy()
                if batch is not None
                else adata_group.copy()
            )
            # Keep only genes expressed in at least min_cells cells within this
            # (group, batch) slice.  sum > 0 is insufficient: near-constant genes
            # with identical low means cause LOESS to become ill-conditioned.
            n_cells = adata_b.n_obs
            min_cells = max(3, int(n_cells * 0.02))
            X = adata_b.X
            if sp_sparse.issparse(X):
                gene_cell_counts = np.asarray((X > 0).sum(axis=0)).flatten()
            else:
                gene_cell_counts = np.asarray((np.asarray(X) > 0).sum(axis=0)).flatten()
            expressed = gene_cell_counts >= min_cells
            n_expressed = int(expressed.sum())
            if n_expressed < 10:
                continue
            adata_b = adata_b[:, expressed].copy()
            # Retry with increasing LOESS span if the fit is ill-conditioned.
            for _span in (0.3, 0.5, 0.75):
                try:
                    sc.pp.highly_variable_genes(
                        adata_b,
                        n_top_genes=min(n_top_genes, n_expressed),
                        flavor=flavor,
                        span=_span,
                        **hvg_kwargs,
                    )
                    break
                except ValueError:
                    if _span == 0.75:
                        raise
            group_hvg |= set(adata_b.var_names[adata_b.var["highly_variable"]])

        hvg_sets.append(group_hvg)

    hvg_union = set.union(*hvg_sets)

    if subset:
        return adata[:, list(hvg_union)].copy()

    adata.var["highly_variable_union"] = adata.var_names.isin(hvg_union)
    return adata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def sanitize_obsm_token(value: object) -> str:
    """Return a safe token for use in ``adata.obsm`` keys.

    Non-alphanumeric characters are replaced with underscores and repeated
    underscores are collapsed. Empty results fall back to ``"group"``.
    """
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return token or "group"


def resolve_group_indices_list(
    adata: "AnnData",
    group_indices_list: Optional[list[Union[np.ndarray, list[int], tuple[int, ...]]]] = None,
) -> tuple[list[list[int]], bool]:
    """Resolve and validate group indices for training/latent APIs.

    Parameters
    ----------
    adata:
        AnnData carrying group metadata in ``adata.uns``.
    group_indices_list:
        Explicit per-group cell indices. If ``None``, indices are inferred from
        ``adata.uns["groups_obs_indices"]``.

    Returns
    -------
    tuple[list[list[int]], bool]
        A validated list of Python ``list[int]`` plus a boolean indicating
        whether values were inferred from ``adata.uns``.

    Raises
    ------
    ValueError
        If indices are missing/malformed, or contain out-of-range/duplicate cells.
    """
    inferred = group_indices_list is None
    if inferred:
        if "groups_obs_indices" not in adata.uns:
            raise ValueError(
                "Could not infer group indices: adata.uns['groups_obs_indices'] is missing. "
                "Run spVIPESmulti.data.prepare_adatas(...) or "
                "spVIPESmulti.data.prepare_multimodal_adatas(...), then call "
                "spVIPESmulti.model.spVIPESmulti.setup_anndata(...), or pass "
                "group_indices_list explicitly."
            )
        group_indices_list = adata.uns["groups_obs_indices"]

    if not isinstance(group_indices_list, (list, tuple)) or len(group_indices_list) == 0:
        raise ValueError(
            "group_indices_list must be a non-empty list of per-group index lists."
        )

    normalized: list[list[int]] = []
    for gi, group in enumerate(group_indices_list):
        if group is None:
            raise ValueError(f"group_indices_list[{gi}] is None; expected a sequence of integer indices.")
        arr = np.asarray(group)
        if arr.ndim != 1:
            raise ValueError(f"group_indices_list[{gi}] must be one-dimensional, got shape {arr.shape}.")
        if arr.size == 0:
            raise ValueError(f"group_indices_list[{gi}] is empty; each group must contain at least one cell.")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"group_indices_list[{gi}] must contain integer indices, got dtype {arr.dtype}."
            )
        int_group = arr.astype(np.int64).tolist()
        normalized.append(int_group)

    n_obs = int(adata.n_obs)
    flat = np.asarray([idx for group in normalized for idx in group], dtype=np.int64)
    if np.any(flat < 0) or np.any(flat >= n_obs):
        bad = flat[(flat < 0) | (flat >= n_obs)][:5].tolist()
        raise ValueError(
            f"group_indices_list contains out-of-range cell indices (first examples: {bad}); "
            f"valid range is [0, {n_obs - 1}]."
        )
    if np.unique(flat).size != flat.size:
        raise ValueError(
            "group_indices_list contains duplicate cell indices across groups. "
            "Each cell must appear in exactly one group."
        )

    return normalized, inferred


def validate_enrichment_network(
    network: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    adata: Optional["AnnData"] = None,
    tmin: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate and normalize a decoupler-style long network table.

    Parameters
    ----------
    network:
        Long-format network table with at least source/target columns.
    source_col:
        Column containing program names (for example TF or pathway names).
    target_col:
        Column containing target features (for example gene symbols).
    weight_col:
        Optional column containing source-target weights.
    adata:
        Optional AnnData used to compute target overlap diagnostics.
    tmin:
        Minimum number of targets per source.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, object]]
        The normalized network table plus validation/coverage metadata.

    Raises
    ------
    TypeError
        If ``network`` is not a pandas DataFrame.
    ValueError
        If required columns are missing or the normalized table is empty.
    """
    if not isinstance(network, pd.DataFrame):
        raise TypeError(
            f"network must be a pandas DataFrame, got {type(network).__name__}."
        )
    if tmin < 1:
        raise ValueError(f"tmin must be >= 1, got {tmin}.")

    required = [source_col, target_col]
    missing = [c for c in required if c not in network.columns]
    if missing:
        raise ValueError(
            "network is missing required column(s): "
            f"{missing}. Expected at least {required}."
        )

    keep_cols = [source_col, target_col]
    has_weight = weight_col in network.columns
    if has_weight:
        keep_cols.append(weight_col)

    df = network[keep_cols].copy()
    df = df.dropna(subset=[source_col, target_col])
    if df.empty:
        raise ValueError(
            "network is empty after dropping rows with missing source/target values."
        )

    df[source_col] = df[source_col].astype(str)
    df[target_col] = df[target_col].astype(str)

    source_sizes = df.groupby(source_col, observed=True)[target_col].nunique()
    valid_sources = source_sizes[source_sizes >= int(tmin)].index
    df = df[df[source_col].isin(valid_sources)].copy()
    if df.empty:
        raise ValueError(
            "No sources satisfy the tmin threshold. "
            f"Lower tmin (current: {tmin}) or provide a denser network."
        )

    warnings_list: list[str] = []
    if not has_weight:
        warnings_list.append(
            f"weight column '{weight_col}' not found; weighted methods may be less informative."
        )

    overlap_stats = {
        "n_targets_in_adata": None,
        "n_targets_overlap": None,
        "target_overlap_ratio": None,
    }
    if adata is not None:
        var_names = pd.Index(adata.var_names.astype(str))
        net_targets = pd.Index(df[target_col].astype(str).unique())
        overlap = net_targets.intersection(var_names)
        overlap_ratio = float(len(overlap) / max(1, len(net_targets)))
        overlap_stats = {
            "n_targets_in_adata": int(var_names.size),
            "n_targets_overlap": int(len(overlap)),
            "target_overlap_ratio": overlap_ratio,
        }
        if len(overlap) == 0:
            warnings_list.append(
                "No target overlap between network and adata.var_names; enrichment will likely fail."
            )
        elif overlap_ratio < 0.05:
            warnings_list.append(
                f"Low network target overlap with adata.var_names ({overlap_ratio:.1%})."
            )

    stats = {
        "n_rows_input": int(network.shape[0]),
        "n_rows_valid": int(df.shape[0]),
        "n_sources": int(df[source_col].nunique()),
        "n_targets": int(df[target_col].nunique()),
        "tmin": int(tmin),
        "has_weight": bool(has_weight),
        "warnings": warnings_list,
        **overlap_stats,
    }
    return df.reset_index(drop=True), stats


def _validate_loadings_df(df: pd.DataFrame, latent_type: str) -> None:
    """Validate a pre-computed loadings DataFrame.

    Parameters
    ----------
    df:
        DataFrame as returned by ``model.get_loadings()[(group_idx, latent_type)]``.
    latent_type:
        Either ``"shared"`` or ``"private"``.

    Raises
    ------
    TypeError
        If ``df`` is not a :class:`pandas.DataFrame`.
    ValueError
        If the DataFrame has NaN values, columns that do not follow the
        ``Z_{latent_type}_{n}`` naming convention, or non-contiguous column
        indices.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"loadings_df must be a pandas DataFrame, got {type(df).__name__}."
        )
    if df.isnull().any().any():
        raise ValueError("loadings_df contains NaN values.")
    if not all(isinstance(c, str) for c in df.columns):
        raise ValueError("loadings_df columns must be strings.")

    prefix = f"Z_{latent_type}_"
    bad_cols = [c for c in df.columns if not c.startswith(prefix)]
    if bad_cols:
        raise ValueError(
            f"All columns must start with '{prefix}'. "
            f"Unexpected columns: {bad_cols[:5]}"
        )

    try:
        indices = [int(c[len(prefix):]) for c in df.columns]
    except ValueError:
        raise ValueError(
            f"Column suffixes after '{prefix}' must be integers, "
            f"e.g. '{prefix}0', '{prefix}1', ..."
        )

    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(
            f"Column indices must be contiguous [0, 1, ..., n-1]. "
            f"Got: {indices[:10]}"
        )


def _resolve_loadings(
    loadings_df: Optional[pd.DataFrame],
    model: object,
    group_idx: int,
    latent_type: str,
) -> pd.DataFrame:
    """Return a validated loadings DataFrame, fetching from *model* if needed."""
    if loadings_df is None:
        if model is None:
            raise ValueError(
                "Provide either 'loadings_df' (pre-computed) or 'model' "
                "(to call model.get_loadings() automatically)."
            )
        loadings_df = model.get_loadings()[(group_idx, latent_type)]
    _validate_loadings_df(loadings_df, latent_type)
    return loadings_df


# ---------------------------------------------------------------------------
# Latent storage
# ---------------------------------------------------------------------------


def store_latents(
    adata: "AnnData",
    latents: dict,
    group_indices_list: list[np.ndarray],
    obsm_prefix: str = "X_spVIPESmulti",
) -> "AnnData":
    """Stitch per-group latent arrays back into ``adata.obsm`` (original cell order).

    Consolidates the manual concatenation pattern used in every tutorial
    notebook. Handles all keys returned by
    :meth:`~spVIPESmulti.model.spVIPESmulti.get_latent_representation`:
    ``shared_reordered``, ``private_reordered``, and (for multimodal models)
    ``private_multimodal_reordered``.

    Parameters
    ----------
    adata:
        AnnData object (same one passed to :meth:`setup_anndata`).
    latents:
        Dict returned by :meth:`~spVIPESmulti.model.spVIPESmulti.get_latent_representation`.
    group_indices_list:
        List of index arrays, one per group (same list passed to ``train``
        and ``get_latent_representation``).
    obsm_prefix:
        Prefix for new keys written to ``adata.obsm``.

    Returns
    -------
    AnnData
        The same ``adata`` object with new ``obsm`` entries (modified in-place):

        - ``{prefix}_shared`` — shared latent for all cells
        - ``{prefix}_private_g{i}`` — private latent, one entry per group
        - ``{prefix}_private_{group}_{modality}`` — multimodal private latents

    Examples
    --------
    >>> latents = model.get_latent_representation(group_indices_list)
    >>> spVIPESmulti.utils.store_latents(adata, latents, group_indices_list)
    >>> sc.pp.neighbors(adata, use_rep="X_spVIPESmulti_shared")
    """
    n_obs = adata.n_obs

    # ---- shared ----
    if "shared_reordered" in latents:
        sample = next(iter(latents["shared_reordered"].values()))
        out = np.zeros((n_obs, sample.shape[1]), dtype=np.float32)
        for gi, idxs in enumerate(group_indices_list):
            out[np.asarray(idxs)] = latents["shared_reordered"][gi]
        adata.obsm[f"{obsm_prefix}_shared"] = out

    # ---- private (single-modal) ----
    if "private_reordered" in latents:
        for gi, idxs in enumerate(group_indices_list):
            arr = latents["private_reordered"][gi]
            out = np.zeros((n_obs, arr.shape[1]), dtype=np.float32)
            out[np.asarray(idxs)] = arr
            adata.obsm[f"{obsm_prefix}_private_g{gi}"] = out

    # ---- private (multimodal) ----
    if "private_multimodal_reordered" in latents:
        pm = latents["private_multimodal_reordered"]
        for (gi, mod), arr in pm.items():
            out = np.zeros((n_obs, arr.shape[1]), dtype=np.float32)
            out[np.asarray(group_indices_list[gi])] = arr
            adata.obsm[f"{obsm_prefix}_private_{gi}_{mod}"] = out

    return adata


# ---------------------------------------------------------------------------
# obs column helpers
# ---------------------------------------------------------------------------


def add_latent_dims_to_obs(
    adata: "AnnData",
    obsm_key: str,
    prefix: Optional[str] = None,
    max_dims: Optional[int] = None,
) -> "AnnData":
    """Copy latent dimensions from ``adata.obsm`` into ``adata.obs`` columns.

    After calling this, latent dimensions can be used directly as ``color=``
    arguments in :func:`scanpy.pl.violin`, :func:`scanpy.pl.umap`, etc.

    Parameters
    ----------
    adata:
        AnnData object containing ``obsm_key`` in ``adata.obsm``.
    obsm_key:
        Key in ``adata.obsm`` to copy from (e.g. ``"X_spVIPESmulti_private_g0"``).
    prefix:
        Column name prefix. Defaults to ``obsm_key`` with leading ``"X_"``
        stripped (e.g. ``"X_spVIPESmulti_private_g0"`` → ``"spVIPESmulti_private_g0"``).
    max_dims:
        Maximum number of dimensions to copy. ``None`` copies all.

    Returns
    -------
    AnnData
        The same ``adata`` with new obs columns ``{prefix}_0``, ``{prefix}_1``, …

    Examples
    --------
    >>> spVIPESmulti.utils.add_latent_dims_to_obs(adata_g0, "X_spVIPESmulti_private_g0", max_dims=5)
    >>> sc.pl.violin(adata_g0, "spVIPESmulti_private_g0_1", groupby="cell_type")
    """
    if obsm_key not in adata.obsm:
        raise KeyError(
            f"'{obsm_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    arr = adata.obsm[obsm_key]
    if prefix is None:
        prefix = obsm_key[2:] if obsm_key.startswith("X_") else obsm_key
    n_dims = arr.shape[1] if max_dims is None else min(max_dims, arr.shape[1])
    for i in range(n_dims):
        adata.obs[f"{prefix}_{i}"] = arr[:, i].astype(float)
    return adata


# ---------------------------------------------------------------------------
# UMAP helpers
# ---------------------------------------------------------------------------


def compute_shared_umap(
    adata: "AnnData",
    obsm_key: str = "X_spVIPESmulti_shared",
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    umap_key: str = "X_umap_spvipesmulti_shared",
) -> "AnnData":
    """Compute UMAP on the shared latent and store it under a named key.

    Wraps :func:`scanpy.pp.neighbors` + :func:`scanpy.tl.umap` with a
    private neighbours key so the result does not overwrite any existing
    ``X_umap`` embedding.

    Parameters
    ----------
    adata:
        AnnData with ``obsm_key`` already populated (e.g. via
        :func:`store_latents`).
    obsm_key:
        Key in ``adata.obsm`` to use as input.
    n_neighbors:
        Number of neighbours for the kNN graph.
    min_dist:
        UMAP ``min_dist`` parameter.
    umap_key:
        Destination key in ``adata.obsm`` for the 2-D UMAP coordinates.

    Returns
    -------
    AnnData
        Same ``adata`` with ``adata.obsm[umap_key]`` written.

    Examples
    --------
    >>> spVIPESmulti.utils.compute_shared_umap(adata)
    >>> spVIPESmulti.pl.umap_shared(adata, color="cell_type")
    """
    import scanpy as sc

    if obsm_key not in adata.obsm:
        raise KeyError(
            f"'{obsm_key}' not found in adata.obsm. "
            f"Run store_latents() first or provide the correct obsm_key."
        )
    nn_key = "_spvipesmulti_nn_shared"
    sc.pp.neighbors(adata, use_rep=obsm_key, key_added=nn_key, n_neighbors=n_neighbors)
    sc.tl.umap(adata, neighbors_key=nn_key, min_dist=min_dist)
    adata.obsm[umap_key] = adata.obsm["X_umap"].copy()
    return adata


def compute_private_umaps(
    adatas_per_group: dict[str, "AnnData"],
    obsm_key: str = "X_spVIPESmulti_private",
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    umap_key: str = "X_umap_spvipesmulti_private",
) -> dict[str, "AnnData"]:
    """Compute UMAP on each group's private latent.

    Parameters
    ----------
    adatas_per_group:
        Mapping ``{group_name: AnnData}`` where each AnnData has ``obsm_key``
        in its ``obsm``.
    obsm_key:
        Key in each group's ``obsm`` to use as input.
    n_neighbors:
        Number of neighbours for the kNN graph.
    min_dist:
        UMAP ``min_dist`` parameter.
    umap_key:
        Destination key in each group's ``obsm``.

    Returns
    -------
    dict[str, AnnData]
        Same mapping, each AnnData updated in-place with ``obsm[umap_key]``.

    Examples
    --------
    >>> adatas = {"day0": adata_g0, "day3": adata_g1}
    >>> spVIPESmulti.utils.compute_private_umaps(adatas)
    >>> spVIPESmulti.pl.umap_private(adatas, color="cell_type")
    """
    import scanpy as sc

    for name, adata in adatas_per_group.items():
        if obsm_key not in adata.obsm:
            raise KeyError(
                f"Group '{name}': '{obsm_key}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        nn_key = "_spvipesmulti_nn_private"
        sc.pp.neighbors(adata, use_rep=obsm_key, key_added=nn_key, n_neighbors=n_neighbors)
        sc.tl.umap(adata, neighbors_key=nn_key, min_dist=min_dist)
        adata.obsm[umap_key] = adata.obsm["X_umap"].copy()
    return adatas_per_group


# ---------------------------------------------------------------------------
# Loadings utilities
# ---------------------------------------------------------------------------


def get_top_genes(
    loadings_df: Optional[pd.DataFrame] = None,
    *,
    model: object = None,
    group_idx: int = 0,
    latent_type: str = "shared",
    n_top: int = 10,
    signed: bool = True,
) -> pd.DataFrame:
    """Rank genes by loading magnitude per latent dimension.

    Parameters
    ----------
    loadings_df:
        Pre-computed loadings DataFrame (as returned by
        ``model.get_loadings()[(group_idx, latent_type)]``).
        Shape ``(n_genes, n_dims)``, index = gene names,
        columns = ``Z_{latent_type}_{0..n-1}``.
        If ``None``, ``model`` must be provided.
    model:
        Fitted spVIPESmulti model. Used to fetch loadings when ``loadings_df``
        is ``None``.
    group_idx:
        Group (dataset) index passed to ``model.get_loadings()``.
    latent_type:
        ``"shared"`` or ``"private"``.
    n_top:
        Number of top genes to return per dimension.
    signed:
        If ``True``, return ``n_top`` most-positive and ``n_top``
        most-negative genes per dimension separately.
        If ``False``, rank by absolute value only.

    Returns
    -------
    pd.DataFrame
        One row per latent dimension. Columns:

        - ``dim`` — dimension name (e.g. ``"Z_shared_0"``)
        - ``pos_genes`` — top positive genes (only when ``signed=True``)
        - ``neg_genes`` — top negative genes (only when ``signed=True``)
        - ``top_genes`` — top genes by absolute value (only when ``signed=False``)

    Examples
    --------
    >>> top = spVIPESmulti.utils.get_top_genes(model=model, n_top=5)
    >>> print(top[["dim", "pos_genes"]].to_string(index=False))
    """
    df = _resolve_loadings(loadings_df, model, group_idx, latent_type)
    rows = []
    for col in df.columns:
        series = df[col]
        if signed:
            pos = series.nlargest(n_top).index.tolist()
            neg = series.nsmallest(n_top).index.tolist()
            rows.append({"dim": col, "pos_genes": pos, "neg_genes": neg})
        else:
            top = series.abs().nlargest(n_top).index.tolist()
            rows.append({"dim": col, "top_genes": top})
    cols = ["dim", "pos_genes", "neg_genes"] if signed else ["dim", "top_genes"]
    return pd.DataFrame(rows, columns=cols)


def score_cells_on_factor(
    adata: "AnnData",
    dim_idx: int,
    obsm_key: str,
    col_name: Optional[str] = None,
) -> "AnnData":
    """Write a single latent dimension from ``adata.obsm`` into ``adata.obs``.

    Useful when you want to colour a UMAP or violin plot by a specific factor
    without copying all dimensions.

    Parameters
    ----------
    adata:
        AnnData object.
    dim_idx:
        Zero-based index of the latent dimension to extract.
    obsm_key:
        Key in ``adata.obsm`` to read from.
    col_name:
        Column name to write in ``adata.obs``. Defaults to
        ``"{obsm_key_stripped}_{dim_idx}"``, e.g. ``"spVIPESmulti_private_g0_2"``.

    Returns
    -------
    AnnData
        Same ``adata`` with a new obs column.

    Examples
    --------
    >>> spVIPESmulti.utils.score_cells_on_factor(adata_g0, dim_idx=2, obsm_key="X_spVIPESmulti_private_g0")
    >>> sc.pl.violin(adata_g0, "spVIPESmulti_private_g0_2", groupby="cell_type")
    """
    if obsm_key not in adata.obsm:
        raise KeyError(
            f"'{obsm_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    arr = adata.obsm[obsm_key]
    if dim_idx >= arr.shape[1]:
        raise ValueError(
            f"dim_idx={dim_idx} is out of range for obsm '{obsm_key}' "
            f"with {arr.shape[1]} dimensions."
        )
    if col_name is None:
        stripped = obsm_key[2:] if obsm_key.startswith("X_") else obsm_key
        col_name = f"{stripped}_{dim_idx}"
    adata.obs[col_name] = arr[:, dim_idx].astype(float)
    return adata
