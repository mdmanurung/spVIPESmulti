"""Plotting utilities for spVIPESmulti results.

All functions are standalone (not model methods) and accept pre-computed
arrays / AnnData objects as inputs so they can be used independently of
the training workflow.

Functions
---------
heatmap_loadings
    Heatmap of top-N gene loadings per latent dimension.
umap_shared
    Convenience wrapper for plotting the shared UMAP.
umap_private
    Grid of per-group private UMAP panels.
factor_violin
    Violin plot of a single latent factor stratified by a cell metadata column.
training_curves
    Multi-panel plot of training history metrics.
loadings_dotplot
    scanpy dotplot of top genes for selected latent dimensions.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from spVIPESmulti.utils import _resolve_loadings, get_top_genes, score_cells_on_factor

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from anndata import AnnData
    from matplotlib.axes import Axes


# ---------------------------------------------------------------------------
# Loadings heatmap
# ---------------------------------------------------------------------------


def heatmap_loadings(
    loadings_df: Optional[pd.DataFrame] = None,
    *,
    model: object = None,
    group_idx: int = 0,
    latent_type: str = "shared",
    n_top: int = 5,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional["Axes"] = None,
) -> "Axes":
    """Heatmap of the top-*n_top* genes by absolute loading per latent dimension.

    Parameters
    ----------
    loadings_df:
        Pre-computed loadings DataFrame (shape ``(n_genes, n_dims)``).
        If ``None``, ``model`` must be provided.
    model:
        Fitted spVIPESmulti model used to fetch loadings when ``loadings_df`` is
        ``None``.
    group_idx:
        Group (dataset) index for ``model.get_loadings()``.
    latent_type:
        ``"shared"`` or ``"private"``.
    n_top:
        Number of top genes (by absolute loading) to show per dimension.
    figsize:
        Figure size. Defaults to ``(2 * n_top, 0.6 * n_dims)``.
    ax:
        Existing matplotlib axes to draw on.  If ``None``, a new figure is
        created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object for further customisation.

    Examples
    --------
    >>> ax = spVIPESmulti.pl.heatmap_loadings(model=model, n_top=10)
    >>> ax.figure.savefig("loadings.pdf")
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "seaborn is required for heatmap_loadings. "
            "Install it with: pip install seaborn"
        )
    import matplotlib.pyplot as plt

    df = _resolve_loadings(loadings_df, model, group_idx, latent_type)
    top_genes_df = get_top_genes(df, n_top=n_top, signed=False)

    # Collect unique genes preserving order of first appearance
    seen: dict[str, None] = {}
    for genes in top_genes_df["top_genes"]:
        for g in genes:
            seen[g] = None
    gene_union = list(seen.keys())

    plot_df = df.loc[gene_union].T  # (n_dims, n_genes_selected)

    n_dims, n_genes = plot_df.shape
    if ax is None:
        if figsize is None:
            figsize = (max(8, 0.6 * n_genes), max(4, 0.5 * n_dims))
        _, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        plot_df,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        xticklabels=gene_union,
        yticklabels=plot_df.index.tolist(),
        linewidths=0.3,
        linecolor="grey",
    )
    ax.set_xlabel("Gene")
    ax.set_ylabel(f"Latent dimension ({latent_type})")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0)
    return ax


# ---------------------------------------------------------------------------
# Shared UMAP
# ---------------------------------------------------------------------------


def umap_shared(
    adata: "AnnData",
    color: Union[str, list[str]],
    basis: str = "X_umap_spvipesmulti_shared",
    **kwargs,
) -> None:
    """Plot the shared latent UMAP embedding.

    Thin wrapper around :func:`scanpy.pl.embedding` that defaults ``basis``
    to the key written by :func:`~spVIPESmulti.utils.compute_shared_umap`.

    Parameters
    ----------
    adata:
        AnnData object with the shared UMAP stored in ``adata.obsm[basis]``.
    color:
        Key(s) in ``adata.obs`` or gene name(s) to colour the embedding by.
    basis:
        Key in ``adata.obsm`` containing the 2-D coordinates.
    **kwargs:
        Forwarded verbatim to :func:`scanpy.pl.embedding`.

    Examples
    --------
    >>> spVIPESmulti.pl.umap_shared(adata, color=["cell_type", "groups"])
    """
    import scanpy as sc

    sc.pl.embedding(adata, basis=basis, color=color, **kwargs)


# ---------------------------------------------------------------------------
# Private UMAP grid
# ---------------------------------------------------------------------------


def umap_private(
    adatas_per_group: dict[str, "AnnData"],
    color: Union[str, list[str]],
    basis: str = "X_umap_spvipesmulti_private",
    ncols: int = 3,
    figsize: Optional[tuple[float, float]] = None,
    **kwargs,
) -> "plt.Figure":
    """Grid of per-group private UMAP panels.

    Parameters
    ----------
    adatas_per_group:
        Mapping ``{group_name: AnnData}`` where each AnnData has ``basis``
        in its ``obsm``.  Built with
        :func:`~spVIPESmulti.utils.compute_private_umaps`.
    color:
        Single key in ``adata.obs`` or gene name to colour each panel.
        Lists are not supported here (one ``color`` per panel).
    basis:
        Key in each group AnnData's ``obsm`` for the 2-D coordinates.
    ncols:
        Number of columns in the grid.
    figsize:
        Total figure size. Defaults to ``(5 * ncols, 4 * nrows)``.
    **kwargs:
        Forwarded to :func:`scanpy.pl.embedding`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing all panels.

    Examples
    --------
    >>> fig = spVIPESmulti.pl.umap_private(adatas, color="cell_type")
    >>> fig.savefig("private_umaps.pdf")
    """
    import matplotlib.pyplot as plt
    import scanpy as sc

    names = list(adatas_per_group.keys())
    n = len(names)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (5 * min(n, ncols), 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for i, name in enumerate(names):
        adata = adatas_per_group[name]
        sc.pl.embedding(
            adata,
            basis=basis,
            color=color,
            ax=axes_flat[i],
            show=False,
            title=name,
            **kwargs,
        )

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Factor violin
# ---------------------------------------------------------------------------


def factor_violin(
    adata: "AnnData",
    dim_idx: int,
    groupby: str,
    obsm_key: str,
    latent_type: str = "private",
    ax: Optional["Axes"] = None,
    **kwargs,
) -> None:
    """Violin plot of a single latent factor stratified by a metadata column.

    If the factor column is not yet in ``adata.obs``, it is added automatically
    via :func:`~spVIPESmulti.utils.score_cells_on_factor`.

    Parameters
    ----------
    adata:
        AnnData object.
    dim_idx:
        Zero-based index of the latent dimension.
    groupby:
        Column in ``adata.obs`` to group cells by on the x-axis.
    obsm_key:
        Key in ``adata.obsm`` containing the latent matrix.
    latent_type:
        Used only to construct the default obs column name
        (``"Z_{latent_type}_{dim_idx}"``).
    ax:
        Existing axes to draw on.
    **kwargs:
        Forwarded to :func:`scanpy.pl.violin`.

    Examples
    --------
    >>> spVIPESmulti.pl.factor_violin(adata_g0, dim_idx=1, groupby="cell_type",
    ...                          obsm_key="X_spVIPESmulti_private_g0")
    """
    import scanpy as sc

    stripped = obsm_key[2:] if obsm_key.startswith("X_") else obsm_key
    col_name = f"{stripped}_{dim_idx}"
    if col_name not in adata.obs.columns:
        score_cells_on_factor(adata, dim_idx=dim_idx, obsm_key=obsm_key, col_name=col_name)
    sc.pl.violin(adata, col_name, groupby=groupby, ax=ax, **kwargs)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def training_curves(
    model: object,
    metrics: Optional[list[str]] = None,
    figsize: Optional[tuple[float, float]] = None,
) -> "plt.Figure":
    """Multi-panel plot of spVIPESmulti training history.

    Parameters
    ----------
    model:
        Fitted spVIPESmulti model with a ``history`` attribute.
    metrics:
        List of keys in ``model.history`` to plot. ``None`` plots all
        available keys.
    figsize:
        Total figure size. Defaults to ``(7 * ncols, 4 * nrows)``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one sub-panel per metric.

    Examples
    --------
    >>> fig = spVIPESmulti.pl.training_curves(model)
    >>> fig.savefig("training.pdf")
    """
    import matplotlib.pyplot as plt

    history = model.history
    if metrics is None:
        metrics = list(history.keys())
    if not metrics:
        raise ValueError("model.history is empty — has the model been trained?")

    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    if figsize is None:
        figsize = (7 * min(len(metrics), ncols), 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for i, key in enumerate(metrics):
        ax = axes_flat[i]
        values = history[key]
        # history values may be pd.Series or list
        if hasattr(values, "values"):
            values = values.values
        ax.plot(values, label=key)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)

    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Loadings dotplot
# ---------------------------------------------------------------------------


def loadings_dotplot(
    adata: "AnnData",
    dims: Union[list[int], list[str]],
    groupby: str,
    *,
    loadings_df: Optional[pd.DataFrame] = None,
    model: object = None,
    group_idx: int = 0,
    latent_type: str = "shared",
    n_top: int = 5,
    **kwargs,
) -> None:
    """scanpy dotplot of top genes for selected latent dimensions.

    For each requested dimension, the ``n_top`` genes with the largest
    absolute loadings are collected. A single :func:`scanpy.pl.dotplot` is
    then drawn with those gene sets as ``var_names``.

    Parameters
    ----------
    adata:
        AnnData object (must contain the genes in its ``var_names``).
    dims:
        Latent dimensions to visualise. Either a list of integer indices
        (e.g. ``[0, 2, 4]``) or column name strings
        (e.g. ``["Z_shared_0", "Z_shared_2"]``).
    groupby:
        Column in ``adata.obs`` to group cells on the y-axis.
    loadings_df:
        Pre-computed loadings DataFrame. If ``None``, ``model`` must be provided.
    model:
        Fitted spVIPESmulti model.
    group_idx:
        Group index for ``model.get_loadings()``.
    latent_type:
        ``"shared"`` or ``"private"``.
    n_top:
        Number of top genes per dimension.
    **kwargs:
        Forwarded to :func:`scanpy.pl.dotplot`.

    Examples
    --------
    >>> spVIPESmulti.pl.loadings_dotplot(
    ...     adata, dims=[0, 1, 2], groupby="cell_type", model=model, n_top=8
    ... )
    """
    import scanpy as sc

    df = _resolve_loadings(loadings_df, model, group_idx, latent_type)

    # Normalise dims to column names
    col_names = df.columns.tolist()
    prefix = f"Z_{latent_type}_"
    resolved_dims: list[str] = []
    for d in dims:
        if isinstance(d, int):
            col = f"{prefix}{d}"
            if col not in col_names:
                raise ValueError(
                    f"Dimension index {d} → '{col}' not found in loadings_df columns. "
                    f"Available: {col_names}"
                )
            resolved_dims.append(col)
        else:
            if d not in col_names:
                raise ValueError(
                    f"Dimension '{d}' not found in loadings_df columns. "
                    f"Available: {col_names}"
                )
            resolved_dims.append(d)

    # Collect top genes per dim in a labelled dict for scanpy dotplot var_names
    var_names: dict[str, list[str]] = {}
    seen: set[str] = set()
    for col in resolved_dims:
        series = df[col]
        top = series.abs().nlargest(n_top).index.tolist()
        # Filter genes present in adata.var_names
        top = [g for g in top if g in adata.var_names]
        # deduplicate across dims
        unique_top = [g for g in top if g not in seen]
        seen.update(unique_top)
        if unique_top:
            var_names[col] = unique_top

    if not var_names:
        raise ValueError(
            "No genes from the loadings were found in adata.var_names. "
            "Ensure adata uses the same gene set as the trained model."
        )

    sc.pl.dotplot(adata, var_names=var_names, groupby=groupby, **kwargs)
