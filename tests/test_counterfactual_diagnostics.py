"""Diagnostics tests for intervention safety reporting."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import anndata as ad

import spVIPESmulti as sv


def _prepared(n_per_group=14, n_genes=8):
    rng = np.random.default_rng(31)
    groups = {}
    for gi, name in enumerate(("g0", "g1")):
        x = rng.poisson(4 + gi, size=(n_per_group, n_genes)).astype(np.float32)
        a = ad.AnnData(X=csr_matrix(x))
        a.obs_names = [f"{name}_c{i}" for i in range(n_per_group)]
        a.var_names = [f"g{i}" for i in range(n_genes)]
        a.obs["cell_type"] = np.where(np.arange(n_per_group) % 2 == 0, "T", "B")
        a.obs["condition"] = np.where(np.arange(n_per_group) % 2 == 0, "ctrl", "stim")
        groups[name] = a
    prepared = sv.data.prepare_adatas(groups)
    sv.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_type",
        condition_key="condition",
    )
    return prepared


def _model(prepared):
    return sv.model.spVIPESmulti(
        prepared,
        n_hidden=12,
        n_dimensions_shared=3,
        n_dimensions_private=2,
        dropout_rate=0.0,
        disentangle_preset="off",
    )


def test_leakage_score_bounds():
    import spVIPESmulti.interventions as svi

    prepared = _prepared()
    model = _model(prepared)
    score = svi.leakage_score(model, prepared, group_key="groups", latent_type="shared")
    assert 0.0 <= score <= 1.0


def test_condition_separability_valid_score():
    import spVIPESmulti.interventions as svi

    prepared = _prepared()
    model = _model(prepared)
    score = svi.condition_separability(model, prepared, label_key="condition")
    assert 0.0 <= score <= 1.0


def test_latent_variance_utilization_finite():
    import spVIPESmulti.interventions as svi

    prepared = _prepared()
    model = _model(prepared)
    encoded = svi.encode_cells(model, prepared)
    stats = svi.latent_variance_utilization(encoded, latent_type="shared")
    assert stats["total_dims"] == 3
    assert 0 <= stats["active_dims"] <= stats["total_dims"]
    assert np.isfinite(stats["variances"]).all()


def test_integration_report_contains_expected_fields():
    import spVIPESmulti.interventions as svi

    prepared = _prepared()
    model = _model(prepared)
    report = svi.integration_report(model, prepared, group_key="groups", label_key="condition")
    assert {"leakage_shared", "leakage_private", "shared_variance", "private_variance", "condition_separability"}.issubset(report)
    assert 0.0 <= report["leakage_shared"] <= 1.0


def test_shared_leakage_warning_above_threshold(monkeypatch):
    import spVIPESmulti.interventions as svi
    import spVIPESmulti.interventions.diagnostics as diagnostics

    prepared = _prepared()
    model = _model(prepared)
    direction = np.zeros(model.module.n_dimensions_shared, dtype=np.float32)
    monkeypatch.setattr(diagnostics, "leakage_score", lambda *args, **kwargs: 0.9)
    with pytest.warns(UserWarning, match="diagnostic only"):
        svi.predict_counterfactual(
            model,
            prepared,
            cells=prepared.uns["groups_obs_indices"][0][:2],
            group_idx=0,
            direction=direction,
        )
