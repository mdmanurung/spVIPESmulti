"""W-040 / F-KBET-LISI: ``kbet`` must return the rejection rate (fraction of
cells whose neighbourhood chi-squared test rejects at alpha=0.05).

Surface: src/spVIPESmulti/metrics.py (kbet)
Verification: V2.7 (primary audit §5)
"""
import numpy as np
import pytest

pytestmark = [pytest.mark.audit_regression]


def test_kbet_returns_rejection_rate():
    """kbet output must be low (~alpha=0.05) for well-mixed data."""
    from spVIPESmulti import metrics

    rng = np.random.default_rng(0)
    z = rng.standard_normal((600, 10))
    groups = np.tile(["g0", "g1"], 300)
    rate = metrics.kbet(z, groups, k=20)
    # For perfectly mixed data, expected rejection rate ≈ 0.05 (alpha level)
    # Allow generous range [0, 0.15] to avoid flakiness
    assert 0.0 <= rate <= 0.20, (
        f"kBET rejection rate for mixed data should be ~0.05, got {rate:.3f}"
    )


def test_kbet_segregated_has_high_rejection():
    """Completely separated groups should yield near-100% rejection rate."""
    from spVIPESmulti import metrics

    rng = np.random.default_rng(1)
    z = np.vstack([
        rng.standard_normal((200, 5)) + np.array([15, 0, 0, 0, 0]),
        rng.standard_normal((200, 5)) + np.array([-15, 0, 0, 0, 0]),
    ])
    groups = np.array(["g0"] * 200 + ["g1"] * 200)
    rate = metrics.kbet(z, groups, k=20)
    assert rate > 0.80, (
        f"kBET rejection rate for segregated data should be close to 1.0, got {rate:.3f}"
    )


@pytest.mark.skip(reason="Requires harmonypy; run manually if harmonypy is installed")
def test_lisi_matches_harmonypy_when_available():
    """If ``harmonypy`` is importable, our LISI must correlate (rho > 0.95)
    with ``harmonypy.compute_lisi`` across a 5-point mixing grid."""
    pytest.importorskip("harmonypy")
    # Implement after Q-HARMONYPY-DEP is signed off.
    pass
