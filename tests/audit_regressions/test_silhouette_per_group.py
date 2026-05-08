"""W-041 / F-SILHOUETTE-GLOBAL: ``integration_report`` must compute a true
per-group silhouette (cell-type within group), not a global scalar replicated
across rows.

Surface: src/spVIPESmulti/metrics.py (integration_report)
Verification: V2.6 (primary audit §5)
"""
import numpy as np
import pytest

pytestmark = [pytest.mark.audit_regression]


def test_per_group_silhouette_differs_across_groups():
    """Group 0 has separable cell-type labels; group 1 has random labels.
    Per-group silhouettes must differ by > 0.1.
    """
    from spVIPESmulti import metrics

    rng = np.random.default_rng(0)
    n = 200  # cells per group
    n_dims = 8

    # Group A: well-separated cell types in private space
    z_a = np.vstack([
        rng.standard_normal((n // 2, n_dims)) + np.array([4, 4] + [0] * (n_dims - 2)),
        rng.standard_normal((n // 2, n_dims)) + np.array([-4, -4] + [0] * (n_dims - 2)),
    ])
    ct_a = np.array(["typeX"] * (n // 2) + ["typeY"] * (n // 2))

    # Group B: random / non-separable private space
    z_b = rng.standard_normal((n, n_dims))
    ct_b = rng.choice(["typeX", "typeY"], size=n)

    # Shared space (dummy — not used by per-group silhouette)
    z_shared = rng.standard_normal((2 * n, n_dims))
    group_labels = np.array(["A"] * n + ["B"] * n)
    cell_labels = np.concatenate([ct_a, ct_b])

    report = metrics.integration_report(
        z_shared,
        group_labels,
        cell_labels,
        z_private_dict={"A": z_a, "B": z_b},
    )

    sil_a = report.loc[report["latent"] == "z_private (A)", "silhouette"].values[0]
    sil_b = report.loc[report["latent"] == "z_private (B)", "silhouette"].values[0]
    assert abs(sil_a - sil_b) > 0.1, (
        f"Per-group silhouettes should differ (well-sep A vs random B): "
        f"sil_A={sil_a:.3f}, sil_B={sil_b:.3f}"
    )
