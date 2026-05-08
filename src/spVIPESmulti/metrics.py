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
    """kBET rejection rate (Büttner et al., 2019) — chi-squared per-cell test.

    For each cell, compares the observed group frequency in its k-NN
    neighbourhood to the global expected frequency via a chi-squared
    statistic, then returns the **fraction of cells whose neighbourhood
    rejects H0 at alpha=0.05** (i.e. is *not* well-mixed).

    A *lower* rejection rate means better mixing. In practice callers
    typically report ``1 - kbet(...)`` so that higher = better.

    .. note::
        The previous implementation returned ``exp(-mean_chi2)`` which is
        a monotone transformation, not the rejection rate. The new
        implementation matches the original paper definition.

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
        kBET rejection rate in [0, 1]. **Lower = better mixing.**
    """
    from scipy.stats import chi2 as _chi2
    from sklearn.neighbors import NearestNeighbors

    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    dof = len(unique_groups) - 1
    if dof < 1:
        return float("nan")  # only one group — metric undefined
    critical = _chi2.ppf(0.95, df=dof)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep)
    _, idx = nn.kneighbors(rep)
    idx = idx[:, 1:]
    expected = (
        pd.Series(groups)
        .value_counts(normalize=True)
        .reindex(unique_groups)
        .values
    )
    chi = np.empty(rep.shape[0])
    for i in range(rep.shape[0]):
        observed = (
            pd.Series(groups[idx[i]])
            .value_counts(normalize=True)
            .reindex(unique_groups, fill_value=0)
            .values
        )
        chi[i] = k * ((observed - expected) ** 2 / (expected + 1e-9)).sum()
    return float((chi > critical).mean())


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
        # W-041: compute per-group silhouette using cell-type labels within
        # each group's private latent space — not a global pool.
        # The semantic question is: "within group g's z_private, do cell types
        # separate?" Each group gets its own silhouette score.
        #
        # We need cell-type labels aligned to each group's rows. Build a mapping
        # from global cell order: for group g we use the cells in group_labels==g.
        unique_group_names = list(z_private_dict.keys())
        group_label_arr = np.asarray(group_labels)
        cell_label_arr = np.asarray(cell_labels)

        # Build cumulative start indices assuming z_private_dict values are in
        # the same row-order as the global arrays filtered by group.
        group_pos = 0
        for group_name, z_priv in z_private_dict.items():
            n_g = z_priv.shape[0]
            # Slice the cell-type labels for this group's cells
            group_mask = group_label_arr == group_name
            if group_mask.sum() == n_g:
                ct_labels = cell_label_arr[group_mask]
            else:
                # Fallback: slice by position in the concatenated order
                ct_labels = cell_label_arr[group_pos : group_pos + n_g]
            group_pos += n_g

            unique_ct = np.unique(ct_labels)
            if len(unique_ct) >= 2 and n_g >= 2:
                rng = np.random.default_rng(0)
                n_sub = min(2000, n_g)
                pick = rng.choice(n_g, size=n_sub, replace=False)
                from sklearn.metrics import silhouette_score
                sil = float(silhouette_score(z_priv[pick], ct_labels[pick]))
            else:
                sil = float("nan")
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


# ---------------------------------------------------------------------------
# Latent dimension statistics
# ---------------------------------------------------------------------------


def latent_dimension_stats(
    latent_array: np.ndarray,
    mu_array: Optional[np.ndarray] = None,
    sigma_array: Optional[np.ndarray] = None,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Per-dimension activity statistics for a latent matrix.

    Computes the standard deviation and mean absolute value of each column.
    When ``mu_array`` and ``sigma_array`` are provided (posterior mean and
    standard deviation), uses the **per-dimension marginal KL** against
    N(0, 1) to flag collapsed dimensions (W-052):

    .. math::
        \\text{KL}_d = \\frac{1}{N}\\sum_i \\left[
            \\frac{\\mu_{i,d}^2 + \\sigma_{i,d}^2}{2}
            - \\log \\sigma_{i,d} - \\frac{1}{2}
        \\right]

    A dimension is flagged ``is_collapsed`` when its mean KL < ``threshold``
    (default 0.05, following Bowman et al. and the standard VAE collapse
    literature). When only the sample array is provided, falls back to the
    std-heuristic with the same ``threshold``.

    Parameters
    ----------
    latent_array:
        2-D array of shape ``(n_cells, n_dims)``.
    mu_array:
        Optional posterior mean array, same shape as ``latent_array``.
    sigma_array:
        Optional posterior std array, same shape as ``latent_array``.
    threshold:
        KL threshold below which a dimension is flagged as collapsed
        (when mu/sigma provided), or std threshold (fallback).

    Returns
    -------
    pd.DataFrame
        One row per dimension. Columns:

        ``dim``
            Dimension index.
        ``std``
            Population standard deviation of z across cells.
        ``mean_abs``
            Mean absolute value of z across cells.
        ``mean_kl``
            Per-dimension mean KL against N(0, 1). ``nan`` if mu/sigma
            not provided.
        ``is_collapsed``
            ``True`` if the dimension is deemed collapsed.
        ``rank``
            Rank by std (1 = most active).

    Examples
    --------
    >>> stats = spVIPESmulti.metrics.latent_dimension_stats(z_shared)
    >>> print(stats[stats.is_collapsed])
    """
    latent_array = np.asarray(latent_array)
    n_dims = latent_array.shape[1]
    stds = latent_array.std(axis=0)
    mean_abs = np.abs(latent_array).mean(axis=0)
    ranks = (-stds).argsort().argsort() + 1  # rank 1 = largest std

    if mu_array is not None and sigma_array is not None:
        mu = np.asarray(mu_array)
        sigma = np.asarray(sigma_array).clip(min=1e-8)
        # Marginal KL per dimension: mean over cells
        kl_per_dim = (0.5 * (mu ** 2 + sigma ** 2 - 1.0 - 2.0 * np.log(sigma))).mean(axis=0)
        is_collapsed = kl_per_dim < threshold
    else:
        # Fallback: std heuristic
        kl_per_dim = np.full(n_dims, float("nan"))
        is_collapsed = stds < threshold

    return pd.DataFrame(
        {
            "dim": np.arange(n_dims),
            "std": stds,
            "mean_abs": mean_abs,
            "mean_kl": kl_per_dim,
            "is_collapsed": is_collapsed,
            "rank": ranks,
        }
    )


# ---------------------------------------------------------------------------
# Reconstruction quality metrics
# ---------------------------------------------------------------------------


def reconstruction_error(
    model,
    adata=None,
    group_indices_list=None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Per-group reconstruction RMSE and Poisson NLL using the model's decoder.

    Runs inference to obtain z_shared and z_private posteriors, passes them
    through the decoder, and compares the expected expression (``px_scale``)
    against the observed normalized counts.

    Parameters
    ----------
    model:
        Trained spVIPESmulti model.
    adata:
        AnnData to evaluate. Defaults to the model's registered AnnData.
    group_indices_list:
        Per-group cell indices. If ``None``, inferred from
        ``adata.uns['groups_obs_indices']``.
    batch_size:
        Mini-batch size for inference.

    Returns
    -------
    pd.DataFrame
        One row per group. Columns:

        ``group``
            Group index.
        ``rmse``
            Root mean-squared error between ``px_scale`` (normalized predicted
            expression) and observed normalized counts.
        ``poisson_nll``
            Mean Poisson negative log-likelihood of observed counts given the
            predicted rate ``px_rate_shared``.

    Examples
    --------
    >>> err = spVIPESmulti.metrics.reconstruction_error(model)
    >>> print(err)
    """
    import torch
    from scvi import REGISTRY_KEYS

    from spVIPESmulti.dataloaders._concat_dataloader import ConcatDataLoader
    from spVIPESmulti.utils import resolve_group_indices_list

    if adata is None:
        adata = model.adata

    group_indices_list, _ = resolve_group_indices_list(adata, group_indices_list)
    n_groups = len(group_indices_list)

    scdl = ConcatDataLoader(
        model.adata_manager,
        indices_list=group_indices_list,
        shuffle=False,
        drop_last=False,
        batch_size=batch_size,
    )

    module = model.module
    was_training = module.training
    module.eval()

    accum_sq_err = {g: 0.0 for g in range(n_groups)}
    accum_nll = {g: 0.0 for g in range(n_groups)}
    accum_count = {g: 0 for g in range(n_groups)}

    try:
        with torch.no_grad():
            for tensors_by_group in scdl:
                per_group = module._split_tensors_by_group(tensors_by_group)
                inference_inputs = module._get_inference_input(tensors_by_group)
                inf_out = module.inference(**inference_inputs)
                gen_out = module.generative(
                    inf_out["private_stats"],
                    inf_out["shared_stats"],
                    inf_out["poe_stats"],
                    inf_out["library"],
                    inference_inputs["groups"],
                    inference_inputs["batch_index"],
                )

                for g in range(n_groups):
                    if g >= len(per_group):
                        continue
                    key = str(g)
                    if key not in gen_out["private_poe"]:
                        continue

                    px_scale = gen_out["private_poe"][key]["px_scale"].cpu()
                    px_rate = gen_out["private_poe"][key]["px_rate_shared"].cpu()

                    # Raw observed counts, sliced to group's genes
                    x_raw = per_group[g][REGISTRY_KEYS.X_KEY]
                    if not isinstance(x_raw, torch.Tensor):
                        x_raw = torch.tensor(x_raw, dtype=torch.float32)
                    else:
                        x_raw = x_raw.float()
                    x_raw = x_raw[:, module.groups_var_indices[g]].cpu()

                    # W-044: RMSE compares expected counts (px.mean) to raw counts,
                    # not the simplex-scaled px_scale vs normalised proportions.
                    px_mean = gen_out["private_poe"][key]["px"].mean.cpu()
                    n_cells = px_mean.shape[0]
                    sq_err = float(((px_mean - x_raw) ** 2).mean())

                    # W-043: Poisson NLL with mixed rate (private + shared blend)
                    # and proper log-factorial term via torch.distributions.Poisson.
                    import torch.distributions as _td
                    px_rate_priv = gen_out["private_poe"][key]["px_rate_private"].cpu()
                    px_rate_shar = gen_out["private_poe"][key]["px_rate_shared"].cpu()
                    px_mixing = getattr(gen_out["private_poe"][key]["px"], "mixture_logits", None)
                    if px_mixing is not None:
                        mixing = torch.sigmoid(px_mixing.cpu())
                        rate = mixing * px_rate_priv + (1 - mixing) * px_rate_shar
                    else:
                        rate = px_rate_shar
                    rate = rate.clamp(min=1e-8)
                    nll = float(-_td.Poisson(rate).log_prob(x_raw).mean())

                    accum_sq_err[g] += sq_err * n_cells
                    accum_nll[g] += nll * n_cells
                    accum_count[g] += n_cells
    finally:
        module.train(was_training)

    rows = []
    for g in range(n_groups):
        n = accum_count[g]
        rows.append(
            {
                "group": g,
                "rmse": float(np.sqrt(accum_sq_err[g] / n)) if n > 0 else float("nan"),
                "poisson_nll": accum_nll[g] / n if n > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=["group", "rmse", "poisson_nll"])
