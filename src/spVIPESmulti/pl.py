"""Plotting utilities for spVIPESmulti results.

All functions are standalone (not model methods) and accept pre-computed
arrays / AnnData objects as inputs so they can be used independently of
the training workflow.

Functions
---------
heatmap_loadings
    Heatmap of top-N gene loadings per latent dimension.
enrichment_heatmap
    Heatmap of per-cell or per-group enrichment activity scores.
interpretation_dashboard
    Two-panel dashboard with shared embedding and enrichment heatmap.
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
plot_latent_dims_in_umap
    One UMAP panel per latent dimension, colored by that dimension's value.
plot_latent_dims_in_heatmap
    Heatmap of mean z_shared activity per cell type or grouping.
plot_latent_dimension_stats
    Barplot of per-dimension std, flagging vanished/inactive dimensions.
show_top_differential_vars
    Horizontal bar chart of top genes for one z_shared dimension (traversal).
differential_vars_heatmap
    Heatmap of traversal effects across all z_shared dimensions and top genes.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
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

    Train and validation curves for the same metric are overlaid on one panel.
    The x-axis shows actual epoch numbers from the history index.

    Parameters
    ----------
    model:
        Fitted spVIPESmulti model with a ``history`` attribute.
    metrics:
        Base metric names (without ``_train``/``_validation`` suffix) to plot.
        ``None`` plots all available metrics.
    figsize:
        Total figure size. Defaults to ``(7 * ncols, 4 * nrows)``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one sub-panel per metric, train and val overlaid.

    Examples
    --------
    >>> fig = spVIPESmulti.pl.training_curves(model)
    >>> fig.savefig("training.pdf")
    """
    import matplotlib.pyplot as plt

    history = model.history
    if not history:
        raise ValueError("model.history is empty — has the model been trained?")

    # --- group raw history keys into panels ---
    # Each panel: base_name -> {"train": df, "val": df}  or {"only": df}
    panels: dict[str, dict[str, object]] = {}

    def _add(base: str, side: str, df: object) -> None:
        panels.setdefault(base, {})[side] = df

    processed: set[str] = set()
    for key in history.keys():
        if key in processed:
            continue
        if key == "train_loss":
            _add("loss", "train", history[key])
        elif key == "validation_loss":
            _add("loss", "val", history[key])
        elif key.endswith("_train"):
            base = key[:-6]
            _add(base, "train", history[key])
            val_key = base + "_validation"
            if val_key in history:
                _add(base, "val", history[val_key])
                processed.add(val_key)
        elif key.endswith("_validation"):
            base = key[:-11]
            if "val" not in panels.get(base, {}):
                _add(base, "val", history[key])
        else:
            _add(key, "only", history[key])
        processed.add(key)

    # filter to requested metrics when provided
    if metrics is not None:
        def _base(m: str) -> str:
            if m.endswith("_train"):
                return m[:-6]
            if m.endswith("_validation"):
                return m[:-11]
            return m
        requested = {_base(m) for m in metrics}
        panels = {k: v for k, v in panels.items() if k in requested}

    if not panels:
        raise ValueError("No matching metrics found in model.history.")

    panel_list = list(panels.items())
    ncols = 2
    nrows = math.ceil(len(panel_list) / ncols)
    if figsize is None:
        figsize = (7 * min(len(panel_list), ncols), 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    _colors = {"train": "steelblue", "val": "darkorange", "only": "steelblue"}

    # Authoritative source for validation frequency: the Trainer object stored
    # on the model after training.  Falls back to 1 (every epoch) if not found.
    _trainer = getattr(model, "trainer", None)
    _cvene = int(getattr(_trainer, "check_val_every_n_epoch", None) or 1)

    def _xy(df):
        x = df.index.to_numpy() if hasattr(df, "index") else range(len(df))
        y = df.values.flatten() if hasattr(df, "values") else list(df)
        return x.astype(float) if hasattr(x, "astype") else x, y

    for i, (title, sides) in enumerate(panel_list):
        ax = axes_flat[i]
        # Build (side, x, y) triples with val x scaled to actual epoch numbers.
        plotdata = []
        max_x = 0.0
        for side, df in sides.items():
            x, y = _xy(df)
            if side == "val" and _cvene > 1:
                # scvi appends val metrics sequentially (0, 1, 2, ...); multiply
                # by check_val_every_n_epoch to recover actual epoch numbers.
                x = x * _cvene
            if len(x):
                max_x = max(max_x, float(x[-1]))
            plotdata.append((side, x, y))
        for side, x, y in plotdata:
            label = side if side != "only" else title
            ax.plot(x, y, label=label, color=_colors.get(side, "steelblue"), alpha=0.85)
        # Pin x-axis to [0, max_x] — suppresses matplotlib's default 5 % right
        # margin that would make the axis extend beyond MAX_EPOCHS.
        if max_x > 0:
            ax.set_xlim(0, max_x)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        if len(sides) > 1:
            ax.legend(fontsize=8)

    for j in range(len(panel_list), len(axes_flat)):
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


# ---------------------------------------------------------------------------
# Enrichment interpretation helpers
# ---------------------------------------------------------------------------


def enrichment_heatmap(
    scores_df: pd.DataFrame,
    *,
    group_labels: Optional[Sequence[object]] = None,
    top_n: int = 20,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional["Axes"] = None,
    cmap: str = "RdBu_r",
    center: float = 0.0,
) -> "Axes":
    """Heatmap of enrichment activity scores.

    Parameters
    ----------
    scores_df
        Enrichment score matrix (cells x programs).
    group_labels
        Optional group labels for per-group mean aggregation before plotting.
    top_n
        Number of highest-variance programs to display.
    figsize
        Figure size when creating a new figure.
    ax
        Existing axes to draw on.
    cmap
        Matplotlib colormap name.
    center
        Value used to center the diverging colormap.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the heatmap.
    """
    if not isinstance(scores_df, pd.DataFrame):
        raise TypeError(
            f"scores_df must be a pandas DataFrame, got {type(scores_df).__name__}."
        )
    if scores_df.empty:
        raise ValueError("scores_df is empty.")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}.")

    try:
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "seaborn is required for enrichment_heatmap. "
            "Install it with: pip install seaborn"
        )
    import matplotlib.pyplot as plt

    plot_df = scores_df.copy()
    if group_labels is not None:
        if len(group_labels) != scores_df.shape[0]:
            raise ValueError(
                "group_labels length must match number of rows in scores_df."
            )
        grouped = plot_df.copy()
        grouped["__group"] = [str(g) for g in group_labels]
        plot_df = grouped.groupby("__group", observed=True).mean()

    n_cols = min(top_n, plot_df.shape[1])
    top_cols = plot_df.var(axis=0).nlargest(n_cols).index.tolist()
    plot_df = plot_df[top_cols]

    if ax is None:
        if figsize is None:
            figsize = (max(8, 0.4 * plot_df.shape[1]), max(4, 0.35 * plot_df.shape[0]))
        _, ax = plt.subplots(figsize=figsize)

    sns.heatmap(plot_df, cmap=cmap, center=center, ax=ax)
    ax.set_xlabel("Program")
    ax.set_ylabel("Group" if group_labels is not None else "Cell")
    return ax


def interpretation_dashboard(
    adata: "AnnData",
    scores_df: pd.DataFrame,
    groupby: str,
    *,
    shared_basis: str = "X_umap_spvipesmulti_shared",
    top_n: int = 20,
    figsize: tuple[float, float] = (14.0, 5.0),
    cmap: str = "RdBu_r",
) -> "plt.Figure":
    """Create a compact two-panel interpretation dashboard.

    Left panel:
        Shared embedding scatter colored by ``groupby`` (when available).
    Right panel:
        Enrichment heatmap aggregated by ``groupby``.
    """
    import matplotlib.pyplot as plt

    if groupby not in adata.obs:
        raise KeyError(
            f"'{groupby}' not found in adata.obs. Available columns: {list(adata.obs.columns)}"
        )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_left, ax_right = axes

    if shared_basis in adata.obsm:
        coords = np.asarray(adata.obsm[shared_basis])
        if coords.shape[1] < 2:
            raise ValueError(
                f"adata.obsm['{shared_basis}'] must have at least 2 columns."
            )
        cats = adata.obs[groupby].astype(str).values
        unique = pd.unique(cats)
        for name in unique:
            mask = cats == name
            ax_left.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.7, label=str(name))
        ax_left.set_title(f"Shared embedding ({groupby})")
        ax_left.set_xlabel("dim1")
        ax_left.set_ylabel("dim2")
        ax_left.legend(fontsize=7, loc="best")
    else:
        ax_left.text(
            0.5,
            0.5,
            f"Missing embedding key: {shared_basis}",
            ha="center",
            va="center",
            transform=ax_left.transAxes,
        )
        ax_left.set_axis_off()

    enrichment_heatmap(
        scores_df,
        group_labels=adata.obs[groupby].values,
        top_n=top_n,
        ax=ax_right,
        cmap=cmap,
    )
    ax_right.set_title("Enrichment summary")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-dimension UMAP coloring
# ---------------------------------------------------------------------------


def plot_latent_dims_in_umap(
    adata: "AnnData",
    obsm_key: str,
    *,
    dims: Optional[list[int]] = None,
    basis: str = "X_umap",
    ncols: int = 5,
    point_size: float = 8.0,
    cmap: str = "RdBu_r",
    figsize_per_panel: tuple[float, float] = (3.5, 3.0),
) -> "plt.Figure":
    """One UMAP panel per latent dimension, colored by that dimension's value.

    Useful for identifying what each z_shared dimension encodes biologically.
    Requires a UMAP to already be computed (e.g., via scanpy.tl.umap).

    Parameters
    ----------
    adata:
        AnnData with ``adata.obsm[obsm_key]`` (latent matrix) and
        ``adata.obsm[basis]`` (2-D UMAP coordinates).
    obsm_key:
        Key in ``adata.obsm`` for the latent space to visualize, e.g.
        ``"X_spvm_shared"``.
    dims:
        Dimension indices to plot. ``None`` plots all dimensions.
    basis:
        Key in ``adata.obsm`` with 2-D UMAP coordinates.
    ncols:
        Number of panels per row.
    point_size:
        Scatter point size.
    cmap:
        Colormap for dimension values.
    figsize_per_panel:
        Width and height of each individual panel.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = spVIPESmulti.pl.plot_latent_dims_in_umap(adata, "X_spvm_shared")
    >>> fig.savefig("shared_dims_umap.pdf", bbox_inches="tight")
    """
    import matplotlib.pyplot as plt

    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not in adata.obsm.")
    if basis not in adata.obsm:
        raise KeyError(
            f"'{basis}' not in adata.obsm. Run scanpy.tl.umap first, or pass the correct basis key."
        )

    latent = np.asarray(adata.obsm[obsm_key])
    coords = np.asarray(adata.obsm[basis])[:, :2]
    n_dims_total = latent.shape[1]

    if dims is None:
        dims = list(range(n_dims_total))

    nrows = math.ceil(len(dims) / ncols)
    fw = figsize_per_panel[0] * min(len(dims), ncols)
    fh = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)
    axes_flat = axes.flatten()

    for i, d in enumerate(dims):
        ax = axes_flat[i]
        vals = latent[:, d]
        vmax = np.percentile(np.abs(vals), 99)
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=vals,
            s=point_size,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_title(f"Z_shared_{d}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)

    for j in range(len(dims), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cell-type × latent-dim heatmap
# ---------------------------------------------------------------------------


def plot_latent_dims_in_heatmap(
    adata: "AnnData",
    obsm_key: str,
    groupby: str,
    *,
    normalize: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "RdBu_r",
) -> "plt.Figure":
    """Heatmap of mean z_shared activity per cell-type (or other grouping).

    Shows which latent dimensions are active in which cell populations,
    helping to reveal the biological meaning of each dimension.

    Parameters
    ----------
    adata:
        AnnData with ``adata.obsm[obsm_key]`` and ``adata.obs[groupby]``.
    obsm_key:
        Key in ``adata.obsm`` for the latent matrix, e.g. ``"X_spvm_shared"``.
    groupby:
        Column in ``adata.obs`` to group cells by (e.g. ``"cell_type"``).
    normalize:
        If ``True``, z-score each column (dimension) so all dims are on the
        same scale despite different magnitudes.
    figsize:
        Figure size. Defaults to auto-scaled by number of dims and groups.
    cmap:
        Colormap for heatmap values.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = spVIPESmulti.pl.plot_latent_dims_in_heatmap(
    ...     adata, "X_spvm_shared", groupby="cell_type"
    ... )
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn is required. Install with: pip install seaborn")
    import matplotlib.pyplot as plt

    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not in adata.obsm.")
    if groupby not in adata.obs:
        raise KeyError(f"'{groupby}' not in adata.obs.")

    latent = np.asarray(adata.obsm[obsm_key])
    n_dims = latent.shape[1]
    col_names = [f"Z_shared_{d}" for d in range(n_dims)]

    df = pd.DataFrame(latent, columns=col_names, index=adata.obs_names)
    df[groupby] = adata.obs[groupby].values
    group_means = df.groupby(groupby, observed=True)[col_names].mean()

    if normalize:
        col_std = group_means.std(axis=0).replace(0, 1)
        col_mean = group_means.mean(axis=0)
        group_means = (group_means - col_mean) / col_std

    n_groups_plot, n_dims_plot = group_means.shape
    if figsize is None:
        figsize = (max(6, 0.5 * n_dims_plot), max(3, 0.35 * n_groups_plot))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        group_means,
        cmap=cmap,
        center=0,
        ax=ax,
        linewidths=0.3,
        linecolor="lightgrey",
    )
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(groupby)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Latent dimension activity statistics plot
# ---------------------------------------------------------------------------


def plot_latent_dimension_stats(
    dim_stats_df: pd.DataFrame,
    *,
    highlight_vanished: bool = True,
    figsize: Optional[tuple[float, float]] = None,
) -> "plt.Figure":
    """Barplot of per-dimension standard deviation, flagging vanished dims.

    Parameters
    ----------
    dim_stats_df:
        Output of :func:`~spVIPESmulti.metrics.latent_dimension_stats`.
    highlight_vanished:
        If ``True``, bars for vanished dimensions are colored red.
    figsize:
        Figure size. Defaults to auto-scaled by number of dimensions.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> stats = spVIPESmulti.metrics.latent_dimension_stats(z_shared)
    >>> fig = spVIPESmulti.pl.plot_latent_dimension_stats(stats)
    """
    import matplotlib.pyplot as plt

    n_dims = len(dim_stats_df)
    if figsize is None:
        figsize = (max(6, 0.35 * n_dims), 3.5)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [
        "firebrick" if (highlight_vanished and v) else "steelblue"
        for v in dim_stats_df["is_vanished"]
    ]
    ax.bar(dim_stats_df["dim"], dim_stats_df["std"], color=colors, width=0.7)
    if highlight_vanished and dim_stats_df["is_vanished"].any():
        ax.axhline(
            y=dim_stats_df["std"][dim_stats_df["is_vanished"]].max(),
            color="firebrick",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="vanished threshold",
        )
        ax.legend(fontsize=8)

    ax.set_xlabel("Latent dimension index")
    ax.set_ylabel("Std across cells")
    ax.set_title("z_shared dimension activity")
    ax.set_xticks(dim_stats_df["dim"])
    ax.tick_params(axis="x", rotation=90 if n_dims > 20 else 0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Traversal gene plots
# ---------------------------------------------------------------------------


def show_top_differential_vars(
    diff_vars_df: pd.DataFrame,
    dim_idx: int,
    *,
    top_n: int = 20,
    figsize: Optional[tuple[float, float]] = None,
    color: str = "steelblue",
) -> "plt.Figure":
    """Horizontal bar chart of top genes for one z_shared dimension.

    Parameters
    ----------
    diff_vars_df:
        Output of :func:`~spVIPESmulti.traversal.calculate_differential_vars`.
        Tidy DataFrame with columns ``dim``, ``gene``, ``effect``.
    dim_idx:
        Dimension index to plot (e.g. ``0`` to show ``Z_shared_0``).
    top_n:
        Number of top genes to show.
    figsize:
        Figure size. Defaults to ``(6, top_n * 0.35)``.
    color:
        Bar color.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> trav = spVIPESmulti.traversal.traverse_latent(model)
    >>> top = spVIPESmulti.traversal.calculate_differential_vars(trav)
    >>> fig = spVIPESmulti.pl.show_top_differential_vars(top, dim_idx=0)
    """
    import matplotlib.pyplot as plt

    dim_name = f"Z_shared_{dim_idx}"
    subset = diff_vars_df[diff_vars_df["dim"] == dim_name].head(top_n)
    if subset.empty:
        raise ValueError(
            f"No entries for dim_idx={dim_idx} ('{dim_name}') in diff_vars_df. "
            f"Available dims: {diff_vars_df['dim'].unique().tolist()}"
        )

    subset = subset.sort_values("effect", ascending=True)
    if figsize is None:
        figsize = (6, max(2.5, len(subset) * 0.35))

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(subset["gene"], subset["effect"], color=color)
    ax.set_xlabel("Traversal effect (max − min normalized expression)")
    ax.set_title(f"Top genes driven by {dim_name}")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def differential_vars_heatmap(
    traversal_df: pd.DataFrame,
    *,
    top_n_genes: int = 40,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "YlOrRd",
) -> "plt.Figure":
    """Heatmap of traversal effects: z_shared dimensions × top genes.

    Shows which genes are most affected by each latent dimension, giving a
    global overview of what the shared latent space encodes.

    Parameters
    ----------
    traversal_df:
        Output of :func:`~spVIPESmulti.traversal.traverse_latent`.
        Shape ``(n_genes, n_dims_shared)``.
    top_n_genes:
        Number of most-affected genes to include (selected by max effect
        across all dimensions).
    figsize:
        Figure size. Defaults to auto-scaled.
    cmap:
        Colormap.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> trav = spVIPESmulti.traversal.traverse_latent(model)
    >>> fig = spVIPESmulti.pl.differential_vars_heatmap(trav, top_n_genes=30)
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn is required. Install with: pip install seaborn")
    import matplotlib.pyplot as plt

    max_effect_per_gene = traversal_df.max(axis=1)
    top_genes = max_effect_per_gene.nlargest(top_n_genes).index.tolist()
    plot_df = traversal_df.loc[top_genes].T  # (n_dims, n_genes)

    n_dims, n_genes = plot_df.shape
    if figsize is None:
        figsize = (max(8, 0.3 * n_genes), max(4, 0.45 * n_dims))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_df,
        cmap=cmap,
        ax=ax,
        linewidths=0.2,
        linecolor="lightgrey",
    )
    ax.set_xlabel("Gene")
    ax.set_ylabel("z_shared dimension")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    return fig
