"""Tests for F11 nonlinear shared/private dependence diagnostics."""

import math

import numpy as np
import pytest

from scripts.benchmark_f11_nonlinear_diagnostics import recommend_f11_diagnostics
from spVIPESmulti.metrics import hsic_rbf, partial_corr_residualized


def test_hsic_rbf_detects_nonlinear_dependence_above_independent_baseline():
    rng = np.random.default_rng(0)
    z_shared = rng.normal(size=(400, 2))
    z_private_independent = rng.normal(size=(400, 2))
    z_private_dependent = np.column_stack(
        [
            z_shared[:, 0] ** 2,
            np.sin(2.0 * z_shared[:, 1]),
        ]
    ) + 0.03 * rng.normal(size=(400, 2))

    independent = hsic_rbf(z_shared, z_private_independent, seed=7)
    dependent = hsic_rbf(z_shared, z_private_dependent, seed=7)

    assert math.isfinite(independent)
    assert math.isfinite(dependent)
    assert independent >= 0.0
    assert dependent > independent * 3.0


def test_hsic_rbf_constant_inputs_use_finite_bandwidth_fallback():
    value = hsic_rbf(np.ones((30, 2)), np.ones((30, 3)))

    assert value == pytest.approx(0.0)
    assert math.isfinite(value)


def test_hsic_rbf_subsampling_is_deterministic_for_same_seed():
    rng = np.random.default_rng(1)
    z_shared = rng.normal(size=(300, 3))
    z_private = z_shared[:, :2] + 0.1 * rng.normal(size=(300, 2))

    first = hsic_rbf(z_shared, z_private, max_samples=80, seed=11)
    second = hsic_rbf(z_shared, z_private, max_samples=80, seed=11)

    assert first == pytest.approx(second)


def test_hsic_rbf_rejects_bad_inputs():
    with pytest.raises(ValueError, match="same number of rows"):
        hsic_rbf(np.ones((10, 2)), np.ones((9, 2)))

    bad = np.ones((10, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        hsic_rbf(bad, np.ones((10, 2)))

    with pytest.raises(ValueError, match="2-D"):
        hsic_rbf(np.ones(10), np.ones((10, 2)))


def test_partial_corr_residualized_removes_numeric_covariate_confounding():
    rng = np.random.default_rng(2)
    covariate = rng.normal(size=(300, 1))
    z_shared = covariate + 0.03 * rng.normal(size=(300, 1))
    z_private = -0.8 * covariate + 0.03 * rng.normal(size=(300, 1))

    raw = partial_corr_residualized(z_shared, z_private)
    adjusted = partial_corr_residualized(z_shared, z_private, covariates=covariate)

    assert raw["max_abs_partial_corr"] > 0.95
    assert adjusted["max_abs_partial_corr"] < 0.2
    assert adjusted["n_covariates"] == 1
    assert adjusted["n_pairs"] == 1


def test_partial_corr_residualized_keeps_direct_leakage_after_unrelated_adjustment():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(250, 1))
    unrelated = rng.normal(size=(250, 2))
    z_shared = np.column_stack([base[:, 0], rng.normal(size=250)])
    z_private = 0.9 * base + 0.04 * rng.normal(size=(250, 1))

    adjusted = partial_corr_residualized(z_shared, z_private, covariates=unrelated)

    assert adjusted["max_abs_partial_corr"] > 0.9
    assert adjusted["n_pairs"] == 2


def test_partial_corr_residualized_without_covariates_matches_centered_correlation():
    x = np.linspace(-1.0, 1.0, 80).reshape(-1, 1)
    y = -x

    out = partial_corr_residualized(x, y)

    assert out["mean_abs_partial_corr"] == pytest.approx(1.0)
    assert out["max_abs_partial_corr"] == pytest.approx(1.0)
    assert out["n_covariates"] == 0


def test_partial_corr_residualized_rejects_invalid_covariates_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="same number of rows"):
        partial_corr_residualized(np.ones((10, 2)), np.ones((10, 2)), covariates=np.ones((9, 1)))

    bad = np.ones((20, 2))
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        partial_corr_residualized(bad, np.ones((20, 2)))


def _f11_row(seed: int, **overrides):
    row = {
        "notes": "ok",
        "seed": seed,
        "hsic_rbf": 0.1,
        "hsic_null_p95": 0.05,
        "partial_corr_mean_abs": 0.03,
        "partial_corr_adjusted_mean_abs": 0.02,
        "orthogonality_within_stratum": 0.05,
        "hidden_nonlinear_signal": "true",
    }
    row.update(overrides)
    return row


def test_f11_recommendation_passes_only_with_complete_reproducible_hidden_signal():
    rec = recommend_f11_diagnostics([_f11_row(seed) for seed in (0, 1, 2)])

    assert rec["verdict"] == "pass"
    assert rec["hidden_nonlinear_signal"] is True


def test_f11_recommendation_iterates_when_seed_coverage_is_missing():
    rec = recommend_f11_diagnostics([_f11_row(seed) for seed in (0, 1)])

    assert rec["verdict"] == "iterate"
    assert "seed coverage" in rec["reason"]


def test_f11_recommendation_iterates_when_cv_exceeds_gate():
    rows = [_f11_row(0, hsic_rbf=0.02), _f11_row(1, hsic_rbf=0.1), _f11_row(2, hsic_rbf=0.2)]

    rec = recommend_f11_diagnostics(rows)

    assert rec["verdict"] == "iterate"
    assert "CV" in rec["reason"]


def test_f11_recommendation_can_be_informational_without_hidden_signal():
    rows = [_f11_row(seed, hidden_nonlinear_signal="false") for seed in (0, 1, 2)]

    rec = recommend_f11_diagnostics(rows)

    assert rec["verdict"] == "informational"
    assert rec["hidden_nonlinear_signal"] is False
