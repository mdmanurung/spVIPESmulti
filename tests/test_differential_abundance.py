import types
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import spVIPESmulti


def _make_mock_model(n_cells=6):
    model = spVIPESmulti.model.spVIPESmulti.__new__(spVIPESmulti.model.spVIPESmulti)

    adata = ad.AnnData(X=np.zeros((n_cells, 2), dtype=np.float32))
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.uns["groups_mapping"] = {0: "group_a", 1: "group_b"}

    model._adata = adata
    model._adata_manager = None
    model._validate_anndata = types.MethodType(lambda self, a=None: self.adata if a is None else a, model)
    model.module = types.SimpleNamespace(
        disentangle_group_shared_weight=0.0,
        use_jeffreys_integ=False,
    )
    return model


def _mock_shared_posterior_payload():
    group_indices = [[0, 1, 2], [3, 4, 5]]

    loc_g0 = np.array([[-2.0], [-1.8], [-2.2]], dtype=np.float32)
    loc_g1 = np.array([[2.0], [2.2], [1.8]], dtype=np.float32)
    scale_g0 = np.ones_like(loc_g0, dtype=np.float32)
    scale_g1 = np.ones_like(loc_g1, dtype=np.float32)

    return {
        "loc": {0: loc_g0.copy(), 1: loc_g1.copy()},
        "scale": {0: scale_g0.copy(), 1: scale_g1.copy()},
        "loc_reordered": {0: loc_g0, 1: loc_g1},
        "scale_reordered": {0: scale_g0, 1: scale_g1},
        "group_indices_list": group_indices,
    }


def test_get_aggregated_posterior_fallback_warns_and_uses_one_sample_per_group(monkeypatch):
    model = _make_mock_model()
    payload = _mock_shared_posterior_payload()
    monkeypatch.setattr(model, "get_shared_posterior", lambda **kwargs: payload)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = model.get_aggregated_posterior()

    msg = [w for w in caught if "sample_key is not registered" in str(w.message)]
    assert len(msg) == 1

    posterior = out["posterior"]
    assert posterior.shape[0] == 2
    assert set(posterior["sample"].tolist()) == {"group_0", "group_1"}
    assert out["metadata"]["used_group_fallback"] is True


def test_get_aggregated_posterior_sample_subset_filtering(monkeypatch):
    model = _make_mock_model()
    payload = _mock_shared_posterior_payload()
    monkeypatch.setattr(model, "get_shared_posterior", lambda **kwargs: payload)

    # Register sample labels so we can test subset filtering behavior.
    model.adata.obs["sample"] = pd.Categorical(["s1", "s1", "s2", "s1", "s2", "s2"])

    out = model.get_aggregated_posterior(sample_subset=["s1"])
    posterior = out["posterior"]

    assert set(posterior["sample"].tolist()) == {"s1"}
    assert set(posterior["group_idx"].tolist()) == {0, 1}


def test_differential_abundance_output_size_matches_cell_count(monkeypatch):
    model = _make_mock_model()
    payload = _mock_shared_posterior_payload()
    monkeypatch.setattr(model, "get_shared_posterior", lambda **kwargs: payload)

    out = model.differential_abundance(group_a=0, group_b=1)
    scores = out["scores"]

    assert scores.shape[0] == model.adata.n_obs
    assert "da_score" in scores.columns


def test_differential_abundance_sign_behavior_under_shift(monkeypatch):
    model = _make_mock_model()
    payload = _mock_shared_posterior_payload()
    monkeypatch.setattr(model, "get_shared_posterior", lambda **kwargs: payload)

    out = model.differential_abundance(group_a=0, group_b=1)
    scores = out["scores"]

    # Group 0 cells are near -2 and should be closer to group_a -> negative scores.
    assert scores.iloc[[0, 1, 2]]["da_score"].mean() < 0
    # Group 1 cells are near +2 and should be closer to group_b -> positive scores.
    assert scores.iloc[[3, 4, 5]]["da_score"].mean() > 0


def test_differential_abundance_warns_when_alignment_is_weak(monkeypatch):
    model = _make_mock_model()
    payload = _mock_shared_posterior_payload()
    monkeypatch.setattr(model, "get_shared_posterior", lambda **kwargs: payload)

    model.adata.obs["sample"] = pd.Categorical(["s1", "s1", "s2", "s1", "s2", "s2"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = model.differential_abundance(group_a=0, group_b=1)

    msg = [
        w
        for w in caught
        if "without explicit shared-latent alignment" in str(w.message)
    ]
    assert len(msg) == 1
