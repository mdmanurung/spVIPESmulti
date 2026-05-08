"""Synthetic data generators shared across audit regression tests.

All generators take a ``seed`` and return numpy arrays / torch tensors with
shapes documented in their docstrings. No spVIPESmulti import here so this
module stays cheap to import.
"""
from __future__ import annotations

import numpy as np


def make_nb_counts(
    n_cells: int = 512,
    n_genes: int = 200,
    theta: float = 10.0,
    lib_loc: float = 8.0,
    seed: int = 0,
):
    """Return (X_raw_int, mu_true, theta_true).

    ``X_raw_int`` has shape (n_cells, n_genes), integer counts drawn from
    NB(mu, theta) with cell-specific library sizes ~ LogNormal(lib_loc, 0.3).
    """
    rng = np.random.default_rng(seed)
    library = rng.lognormal(mean=lib_loc, sigma=0.3, size=(n_cells, 1))
    base = rng.dirichlet(alpha=np.ones(n_genes), size=1)
    mu = library * base  # (n_cells, n_genes)
    p = mu / (mu + theta)
    X = rng.negative_binomial(n=theta, p=1 - p).astype(np.int64)
    return X, mu, float(theta)


def make_lognorm_protein(
    n_cells: int = 512,
    n_proteins: int = 20,
    mu_range: tuple[float, float] = (-3.0, 3.0),
    sigma: float = 0.5,
    seed: int = 0,
):
    """Return (Y, mu_true) with shape (n_cells, n_proteins)."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(low=mu_range[0], high=mu_range[1], size=(1, n_proteins))
    mu = np.broadcast_to(mu, (n_cells, n_proteins)).copy()
    Y = rng.normal(loc=mu, scale=sigma)
    return Y, mu


def make_paired_two_group(
    n_cells_per: int = 256,
    n_genes: int = 200,
    share_frac: float = 0.5,
    seed: int = 0,
):
    """Return dict[str, np.ndarray] for two groups with paired cells.

    Row ``i`` of group A and group B correspond to the same latent cell.
    ``share_frac`` controls the fraction of genes that share the same mean
    across groups.
    """
    rng = np.random.default_rng(seed)
    n_shared = int(share_frac * n_genes)
    base = rng.dirichlet(alpha=np.ones(n_genes), size=1)[0]
    a_extra = rng.dirichlet(alpha=np.ones(n_genes), size=1)[0]
    b_extra = rng.dirichlet(alpha=np.ones(n_genes), size=1)[0]
    library = rng.lognormal(mean=8.0, sigma=0.3, size=(n_cells_per, 1))
    mean_a = library * (base if n_shared == n_genes else 0.5 * base + 0.5 * a_extra)
    mean_b = library * (base if n_shared == n_genes else 0.5 * base + 0.5 * b_extra)
    A = rng.poisson(mean_a).astype(np.int64)
    B = rng.poisson(mean_b).astype(np.int64)
    return {"A": A, "B": B}


def make_unpaired_two_group(seed: int = 0, **kwargs):
    """Same as ``make_paired_two_group`` but rows of B are shuffled."""
    out = make_paired_two_group(seed=seed, **kwargs)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(out["B"].shape[0])
    out["B"] = out["B"][perm]
    out["_perm"] = perm
    return out


def make_label_permutation_iter(group_labels: np.ndarray, n_permutations: int = 200, seed: int = 0):
    """Yield ``n_permutations`` arrays of shuffled group labels."""
    rng = np.random.default_rng(seed)
    for _ in range(n_permutations):
        yield rng.permutation(group_labels)


def make_two_group_bimodal_private(
    n_cells_per: int = 256,
    n_private: int = 4,
    seed: int = 0,
):
    """Two groups whose private latent means differ by 2 in every dim.

    Returns (Z_a, Z_b) with shape (n_cells_per, n_private) each.
    """
    rng = np.random.default_rng(seed)
    Z_a = rng.normal(loc=0.0, scale=1.0, size=(n_cells_per, n_private))
    Z_b = rng.normal(loc=2.0, scale=1.0, size=(n_cells_per, n_private))
    return Z_a, Z_b
