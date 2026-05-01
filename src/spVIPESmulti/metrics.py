"""Integration quality metrics for spVIPESmulti latent spaces.

All metrics work on raw NumPy arrays (no AnnData dependency) and use only
NumPy, pandas, and scikit-learn — all of which are available transitively
through scvi-tools.

Metric semantics
----------------
Shared latent (z_shared) — you want groups to *mix* and labels to *separate*:
- ``ilisi``:  higher → better group mixing  (range: 1 → n_groups)
- ``clisi``:  lower  → better label separation (range: 1 → n_labels)
- ``kbet``:   higher → better group mixing  (range: 0 → 1)
- ``knn_purity``:  higher → better label preservation (range: 0 → 1)
- ``leiden_ari``:  higher → better label structure  (range: 0 → 1)

Private latent (z_private) — you want groups to *separate*:
- ``per_group_silhouette``:  higher → groups more separated (range: −1 → 1)

``integration_report`` bundles all of these into a single DataFrame.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def ilisi(rep: np.ndarray, groups: np.ndarray, k: int = 30) -> float:
    """Inverse Simpson's diversity index over k-NN neighbours (group labels).

    A local measure of group mixing. For each cell, computes the inverse
    Simpson index over the group composition of its k nearest neighbours.
    Averaged across all cells.

    Parameters
    ----------
    rep:
        2-D array of shape ``(n_cells, n_dims)``.
    groups:
        1-D array of group labels, length ``n_cells``.
    k:
        Number of neighbours (excluding the cell itself).

    Returns
    -------
    float
        Mean iLISI score. Range [1, n_groups]. Higher = better mixing.
    """
    from sklearn.neighbors import NearestNeighbors

    groups = np.asarray(groups)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep)
    _, idx = nn.kneighbors(rep)
    idx = idx[:, 1:]
    out = np.empty(rep.shape[0])
    for i in range(rep.shape[0]):
        _, c = np.unique(groups[idx[i]], return_counts=True)
        p = c / c.sum()
        out[i] = 1.0 / float((p * p).sum())
    return float(out.mean())


def clisi(rep: np.ndarray, labels: np.ndarray, k: int = 30) -> float:
    """Inverse Simpson's diversity index over k-NN neighbours (cell-type labels).

    Identical computation to :func:`ilisi` but applied to cell-type labels.
    Lower values indicate that neighbours share the same label → better
    label preservation.

    Parameters
    ----------
    rep:
        2-D array of shape ``(n_cells, n_dims)``.
    labels:
        1-D array of cell-type labels, length ``n_cells``.
    k:
        Number of neighbours.

    Returns
    -------
    float
        Mean cLISI score. Range [1, n_labels]. Lower = better label separation.
    """
    return ilisi(rep, labels, k=k)


def kbet(rep: np.ndarray, groups: np.ndarray, k: int = 20) -> float:
    """Chi-squared proxy for kBET (Büttner et al., 2019).

    For each cell, compares the observed group frequency in its k-NN
    neighbourhood to the global expected frequency via a chi-squared
    statistic. The per-cell chi-squared values are averaged and transformed
    via ``exp(−chi_mean)`` so that perfect mixing maps to 1.0.

    Parameters
    ----------
    rep:
        2-D array of shape ``(n_cells, n_dims)``.
    groups:
        1-D array of group labels, length ``n_cells``.
    k:
        Neighbourhood size.

    Returns
    -------
    float
        kBET score in [0, 1]. Higher = better mixing.
    """
    from sklearn.neighbors import NearestNeighbors

    groups = np.asarray(groups)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep)
    _, idx = nn.kneighbors(rep)
    idx = idx[:, 1:]
    expected = (
        pd.Series(groups)
        .value_counts(normalize=True)
        .reindex(np.unique(groups))
        .values
    )
    chi = np.empty(rep.shape[0])
    for i in range(rep.shape[0]):
        observed = (
            pd.Series(groups[idx[i]])
            .value_counts(normalize=True)
            .reindex(np.unique(groups), fill_value=0)
            .values
        )
        chi[i] = ((observed - expected) ** 2 / (expected + 1e-9)).sum()
    return float(np.exp(-chi.mean()))


def knn_purity(rep: np.ndarray, labels: np.ndarray, k: int = 20) -> float:
    """Fraction of k-NN neighbours that share the query cell's label.

    Parameters
    ----------
    rep:
        2-D array of shape ``(n_cells, n_dims)``.
    labels:
        1-D array of cell-type labels, length ``n_cells``.
    k:
        Neighbourhood size.

    Returns
    -------
    float
        Mean k-NN purity in [0, 1]. Higher = better label preservation.
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep)
    _, idx = nn.kneighbors(rep)
    idx = idx[:, 1:]
    return float(
        np.mean([(labels[idx[i]] == labels[i]).mean() for i in range(len(labels))])
    )


def leiden_ari(rep: np.ndarray, labels: np.ndarray, resolution: float = 0.8) -> float:
    """Leiden clustering ARI against true cell-type labels.

    Builds a k-NN graph on ``rep``, runs Leiden clustering, then computes
    Adjusted Rand Index against the provided labels.

    Parameters
    ----------
    rep:
        2-D array of shape ``(n_cells, n_dims)``.
    labels:
        1-D array of ground-truth labels, length ``n_cells``.
    resolution:
        Leiden resolution parameter.

    Returns
    -------
    float
        ARI in [−1, 1]. Higher = better label structure recovered.
        Returns ``nan`` if the ``igraph`` package is not installed
        (required by scanpy's Leiden implementation).
    """
    import importlib.util

    if importlib.util.find_spec("igraph") is None:
        import warnings

        warnings.warn(
            "leiden_ari requires igraph. Install with: pip install igraph. "
            "Returning nan.",
            ImportWarning,
            stacklevel=2,
        )
        return float("nan")

    import anndata as ad
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    tmp = ad.AnnData(X=rep.astype(np.float32))
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15)
    sc.tl.leiden(tmp, resolution=resolution, random_state=0)
    return float(adjusted_rand_score(np.asarray(labels), tmp.obs["leiden"].values))


def per_group_silhouette(z_private: np.ndarray, groups: np.ndarray) -> float:
    """Silhouette score of group identity on the private latent space.

    Sub-samples to at most 2,000 cells for speed.

    Parameters
    ----------
    z_private:
        2-D array of shape ``(n_cells, n_dims_private)``.
    groups:
        1-D array of group labels, length ``n_cells``.

    Returns
    -------
    float
        Silhouette score in [−1, 1]. Higher = groups more separated in
        z_private, which is the desired behaviour for private latents.
        Returns ``nan`` if fewer than 2 unique groups are present.
    """
    from sklearn.metrics import silhouette_score

    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        return float("nan")
    rng = np.random.default_rng(0)
    n = min(2000, z_private.shape[0])
    pick = rng.choice(z_private.shape[0], size=n, replace=False)
    return float(silhouette_score(z_private[pick], groups[pick], sample_size=n))


def integration_report(
    z_shared: np.ndarray,
    group_labels: np.ndarray,
    cell_labels: np.ndarray,
    *,
    z_private_dict: Optional[dict[str, np.ndarray]] = None,
    k: int = 20,
    leiden_resolution: float = 0.8,
) -> pd.DataFrame:
    """Compute a full integration-quality report for a trained spVIPESmulti model.

    Evaluates group mixing and label preservation on the shared latent space,
    and (optionally) group separability on per-group private latent spaces.

    Parameters
    ----------
    z_shared:
        2-D array of shape ``(n_cells, n_dims_shared)``, in the original
        AnnData cell order (e.g., from ``store_latents`` or manual stitching).
    group_labels:
        1-D array of group assignments for every cell, length ``n_cells``.
    cell_labels:
        1-D array of cell-type annotations for every cell, length ``n_cells``.
    z_private_dict:
        Optional mapping ``{group_name: ndarray}`` where each value has shape
        ``(n_cells_in_group, n_dims_private)``.  When provided, silhouette
        scores are computed per group (comparing within-group sub-populations
        does not make sense, so the silhouette is computed **across all groups
        pooled together**).
    k:
        Neighbourhood size for kNN-based metrics (iLISI, cLISI, kBET, purity).
    leiden_resolution:
        Resolution for Leiden clustering used in ARI computation.

    Returns
    -------
    pd.DataFrame
        One row per evaluated latent space. Columns:

        ``latent``
            Name of the latent space evaluated.
        ``ilisi``
            Group mixing iLISI on z_shared (higher = better).
        ``clisi``
            Label cLISI on z_shared (lower = better separation).
        ``kbet``
            kBET proxy on z_shared (higher = better mixing).
        ``knn_purity``
            k-NN purity on z_shared (higher = better label preservation).
        ``leiden_ari``
            Leiden ARI on z_shared (higher = better label structure).
        ``silhouette``
            Group silhouette on z_private (higher = better separation).
            ``nan`` for the shared row (not applicable).

    Examples
    --------
    >>> report = spVIPESmulti.metrics.integration_report(
    ...     z_shared, adata.obs["groups"].values, adata.obs["cell_types"].values,
    ...     z_private_dict={"SLN111": z_priv_0, "SLN208": z_priv_1},
    ... )
    >>> print(report.to_string(index=False))
    """
    group_labels = np.asarray(group_labels)
    cell_labels = np.asarray(cell_labels)

    shared_row = {
        "latent": "z_shared",
        "ilisi": ilisi(z_shared, group_labels, k=k),
        "clisi": clisi(z_shared, cell_labels, k=k),
        "kbet": kbet(z_shared, group_labels, k=k),
        "knn_purity": knn_purity(z_shared, cell_labels, k=k),
        "leiden_ari": leiden_ari(z_shared, cell_labels, resolution=leiden_resolution),
        "silhouette": float("nan"),
    }
    rows = [shared_row]

    if z_private_dict is not None:
        # Pool all private latents + group labels to compute cross-group silhouette
        all_z = np.concatenate(list(z_private_dict.values()), axis=0)
        all_g = np.concatenate(
            [np.full(v.shape[0], k) for k, v in z_private_dict.items()]
        )
        sil = per_group_silhouette(all_z, all_g)
        for group_name in z_private_dict:
            rows.append(
                {
                    "latent": f"z_private ({group_name})",
                    "ilisi": float("nan"),
                    "clisi": float("nan"),
                    "kbet": float("nan"),
                    "knn_purity": float("nan"),
                    "leiden_ari": float("nan"),
                    "silhouette": sil,
                }
            )

    return pd.DataFrame(rows, columns=["latent", "ilisi", "clisi", "kbet", "knn_purity", "leiden_ari", "silhouette"])
