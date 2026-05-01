"""Utility functions for working with spVIPES latent spaces.

These functions cover the post-training workflow that every tutorial notebook
repeats manually:

1. :func:`store_latents` — stitch per-group latents back into ``adata.obsm``
   in original cell order.
2. :func:`add_latent_dims_to_obs` — copy latent dimensions into ``adata.obs``
   so they can be used as ``color=`` arguments in scanpy plots.
3. :func:`compute_shared_umap` — run neighbours + UMAP on the shared latent
   and store the result under a named key.
4. :func:`compute_private_umaps` — same for per-group private latents.
5. :func:`get_top_genes` — rank genes by loading magnitude per latent dimension.
6. :func:`score_cells_on_factor` — project a single latent dimension into
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
# Internal helpers
# ---------------------------------------------------------------------------


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
    obsm_prefix: str = "X_spVIPES",
) -> "AnnData":
    """Stitch per-group latent arrays back into ``adata.obsm`` (original cell order).

    Consolidates the manual concatenation pattern used in every tutorial
    notebook. Handles all keys returned by
    :meth:`~spVIPES.model.spVIPES.get_latent_representation`:
    ``shared_reordered``, ``private_reordered``, and (for multimodal models)
    ``private_multimodal_reordered``.

    Parameters
    ----------
    adata:
        AnnData object (same one passed to :meth:`setup_anndata`).
    latents:
        Dict returned by :meth:`~spVIPES.model.spVIPES.get_latent_representation`.
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
    >>> spVIPES.utils.store_latents(adata, latents, group_indices_list)
    >>> sc.pp.neighbors(adata, use_rep="X_spVIPES_shared")
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
        Key in ``adata.obsm`` to copy from (e.g. ``"X_spVIPES_private_g0"``).
    prefix:
        Column name prefix. Defaults to ``obsm_key`` with leading ``"X_"``
        stripped (e.g. ``"X_spVIPES_private_g0"`` → ``"spVIPES_private_g0"``).
    max_dims:
        Maximum number of dimensions to copy. ``None`` copies all.

    Returns
    -------
    AnnData
        The same ``adata`` with new obs columns ``{prefix}_0``, ``{prefix}_1``, …

    Examples
    --------
    >>> spVIPES.utils.add_latent_dims_to_obs(adata_g0, "X_spVIPES_private_g0", max_dims=5)
    >>> sc.pl.violin(adata_g0, "spVIPES_private_g0_1", groupby="cell_type")
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
    obsm_key: str = "X_spVIPES_shared",
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    umap_key: str = "X_umap_spvipes_shared",
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
    >>> spVIPES.utils.compute_shared_umap(adata)
    >>> spVIPES.pl.umap_shared(adata, color="cell_type")
    """
    import scanpy as sc

    if obsm_key not in adata.obsm:
        raise KeyError(
            f"'{obsm_key}' not found in adata.obsm. "
            f"Run store_latents() first or provide the correct obsm_key."
        )
    nn_key = "_spvipes_nn_shared"
    sc.pp.neighbors(adata, use_rep=obsm_key, key_added=nn_key, n_neighbors=n_neighbors)
    sc.tl.umap(adata, neighbors_key=nn_key, min_dist=min_dist)
    adata.obsm[umap_key] = adata.obsm["X_umap"].copy()
    return adata


def compute_private_umaps(
    adatas_per_group: dict[str, "AnnData"],
    obsm_key: str = "X_spVIPES_private",
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    umap_key: str = "X_umap_spvipes_private",
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
    >>> spVIPES.utils.compute_private_umaps(adatas)
    >>> spVIPES.pl.umap_private(adatas, color="cell_type")
    """
    import scanpy as sc

    for name, adata in adatas_per_group.items():
        if obsm_key not in adata.obsm:
            raise KeyError(
                f"Group '{name}': '{obsm_key}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        nn_key = "_spvipes_nn_private"
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
        Fitted spVIPES model. Used to fetch loadings when ``loadings_df``
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
    >>> top = spVIPES.utils.get_top_genes(model=model, n_top=5)
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
        ``"{obsm_key_stripped}_{dim_idx}"``, e.g. ``"spVIPES_private_g0_2"``.

    Returns
    -------
    AnnData
        Same ``adata`` with a new obs column.

    Examples
    --------
    >>> spVIPES.utils.score_cells_on_factor(adata_g0, dim_idx=2, obsm_key="X_spVIPES_private_g0")
    >>> sc.pl.violin(adata_g0, "spVIPES_private_g0_2", groupby="cell_type")
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
