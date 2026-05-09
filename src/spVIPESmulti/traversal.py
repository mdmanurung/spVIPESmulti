"""Latent traversal utilities for interpreting z_shared dimensions.

Systematically varies each z_shared dimension from low to high while holding
others fixed, measures the resulting change in decoder output (px_scale_shared),
and returns per-gene effect scores. This is more robust than static decoder
weight loadings when the decoder has non-linear interactions.

Functions
---------
traverse_latent
    Traverse z_shared dimensions and score gene effects.
calculate_differential_vars
    Rank genes per dimension from traversal output.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


def traverse_latent(
    model,
    adata=None,
    group_idx: int = 0,
    n_steps: int = 15,
    n_samples: int = 50,
    n_stds: float = 3.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Score genes by their response to traversal of each z_shared dimension.

    For each z_shared dimension d, this function:

    1. Samples ``n_samples`` cells from ``group_idx`` and extracts their
       posterior mean z_shared embeddings.
    2. Creates a traversal grid: ``n_steps`` evenly-spaced values spanning
       ±``n_stds`` empirical standard deviations around the population mean
       of dimension d.
    3. At each step, replaces dimension d in all sample cells with the step
       value (holding other dims at their posterior mean), passes z_private=0
       and the traversal z_shared through the decoder, and records
       ``px_scale_shared`` (the shared-only normalized expression).
    4. Computes per-gene effect = ``max(mean_cells(px_scale_shared))`` minus
       ``min(mean_cells(px_scale_shared))`` across the traversal steps.

    Parameters
    ----------
    model:
        Trained spVIPESmulti model.
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
        Shape ``(n_genes, n_dims_shared)``. Each entry is the max−min effect
        of that z_shared dimension on that gene's normalized expression.
        Column names are ``Z_shared_0``, ``Z_shared_1``, …
        Index is the gene names for ``group_idx``.

    Examples
    --------
    >>> trav = spVIPESmulti.traversal.traverse_latent(model, n_steps=15)
    >>> trav.head()
    """
    import torch

    if adata is None:
        adata = model.adata

    module = model.module
    n_dims_shared = module.n_dimensions_shared
    n_dims_private = module.n_dimensions_private
    decoder = module.decoders[group_idx]
    dispersion = module.dispersion
    n_batch = module.n_batch

    # Gene names for this group
    var_indices = module.groups_var_indices[group_idx]
    gene_names = adata.var_names[var_indices].tolist()
    n_genes = len(gene_names)

    # Get z_shared posterior means for all cells in this group
    from spVIPESmulti.utils import resolve_group_indices_list

    group_indices_list, _ = resolve_group_indices_list(adata, None)
    group_cell_indices = group_indices_list[group_idx]

    latent_dict = model.get_latent_representation(
        group_indices_list=group_indices_list,
        adata=adata,
    )
    # get_latent_representation returns a dict with 'shared_reordered' per group
    z_shared_all = latent_dict["shared_reordered"][group_idx]  # (n_cells_group, n_dims_shared)

    rng = np.random.default_rng(seed)
    n_cells = z_shared_all.shape[0]
    pick = rng.choice(n_cells, size=min(n_samples, n_cells), replace=False)
    device = next(module.parameters()).device
    z_shared_sample = torch.tensor(z_shared_all[pick], dtype=torch.float32, device=device)  # (n_samples, n_dims_shared)

    # Dim-wise empirical statistics (population-level, from all group cells)
    dim_mean = torch.tensor(z_shared_all.mean(axis=0), dtype=torch.float32, device=device)
    dim_std = torch.tensor(z_shared_all.std(axis=0) + 1e-6, dtype=torch.float32, device=device)

    # Fixed inputs: z_private = 0 (prior mean), library = log(1e4)
    z_private_zero = torch.zeros(z_shared_sample.shape[0], n_dims_private, device=device)
    library_fixed = torch.full((z_shared_sample.shape[0], 1), float(np.log(1e4)), device=device)
    cat_args = (
        (torch.zeros(z_shared_sample.shape[0], 1, dtype=torch.long, device=device),) if n_batch > 0 else ()
    )

    was_training = module.training
    module.eval()
    effects = np.zeros((n_genes, n_dims_shared), dtype=np.float32)

    try:
        with torch.no_grad():
            for d in range(n_dims_shared):
                step_low = float(dim_mean[d] - n_stds * dim_std[d])
                step_high = float(dim_mean[d] + n_stds * dim_std[d])
                step_values = np.linspace(step_low, step_high, n_steps)

                # Mean expression across cells at each step
                step_means = np.zeros((n_steps, n_genes), dtype=np.float32)
                z_trav = z_shared_sample.clone()
                # Set all other dims to their posterior mean for this sample
                for other_d in range(n_dims_shared):
                    if other_d != d:
                        z_trav[:, other_d] = dim_mean[other_d]

                for s, val in enumerate(step_values):
                    z_trav[:, d] = float(val)
                    _, px_scale_shared, _, _, _, _ = decoder(
                        dispersion,
                        z_private_zero,
                        z_trav,
                        library_fixed,
                        *cat_args,
                    )
                    step_means[s] = px_scale_shared.cpu().numpy().mean(axis=0)

                effects[:, d] = step_means.max(axis=0) - step_means.min(axis=0)
    finally:
        module.train(was_training)

    col_names = [f"Z_shared_{d}" for d in range(n_dims_shared)]
    return pd.DataFrame(effects, index=gene_names, columns=col_names)


def calculate_differential_vars(
    traversal_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank genes by traversal effect per z_shared dimension.

    Parameters
    ----------
    traversal_df:
        Output of :func:`traverse_latent`. Shape ``(n_genes, n_dims_shared)``.
    top_n:
        Number of top genes to return per dimension.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns: ``dim``, ``gene``, ``effect``.
        Sorted by ``dim`` then descending ``effect``.

    Examples
    --------
    >>> trav = spVIPESmulti.traversal.traverse_latent(model)
    >>> top = spVIPESmulti.traversal.calculate_differential_vars(trav, top_n=15)
    >>> top.head(30)
    """
    rows = []
    for col in traversal_df.columns:
        series = traversal_df[col].nlargest(top_n)
        for gene, effect in series.items():
            rows.append({"dim": col, "gene": gene, "effect": float(effect)})
    return pd.DataFrame(rows, columns=["dim", "gene", "effect"])
