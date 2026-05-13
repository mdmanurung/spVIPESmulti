"""Integration-style smoke tests for F2 counterfactual helpers."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import spVIPESmulti as sv


def _make_prepared(n_per_group=18, n_genes=10):
    rng = np.random.default_rng(21)
    groups = {}
    for gi, name in enumerate(("donor0", "donor1")):
        condition = np.where(np.arange(n_per_group) < n_per_group // 2, "ctrl", "stim")
        base = np.where(condition == "stim", 6 + gi, 3 + gi)
        x = rng.poisson(base[:, None], size=(n_per_group, n_genes)).astype(np.float32)
        a = ad.AnnData(X=csr_matrix(x))
        a.obs_names = [f"{name}_c{i}" for i in range(n_per_group)]
        a.var_names = [f"g{i}" for i in range(n_genes)]
        a.obs["cell_type"] = np.where(np.arange(n_per_group) % 2 == 0, "T", "B")
        a.obs["condition"] = condition
        a.obs["donor"] = name
        groups[name] = a
    prepared = sv.data.prepare_adatas(groups)
    sv.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_type",
        condition_key="condition",
        donor_key="donor",
    )
    return prepared


def _model(prepared):
    return sv.model.spVIPESmulti(
        prepared,
        n_hidden=16,
        n_dimensions_shared=4,
        n_dimensions_private=3,
        dropout_rate=0.0,
        disentangle_preset="off",
    )


def test_identity_decode_preserves_library_size():
    import spVIPESmulti.interventions as svi

    prepared = _make_prepared()
    model = _model(prepared)
    encoded = svi.encode_cells(model, prepared)
    result = svi.decode_counterfactual(
        model,
        encoded["shared"][0][:5],
        encoded["private"][0][:5],
        group_idx=0,
        adata=prepared,
        cells=encoded["obs_indices"][0][:5],
        library=encoded["library"][0][:5],
        include_uncertainty=False,
    )
    expected_library = np.exp(encoded["library"][0][:5].ravel())
    assert np.allclose(result.X.sum(axis=1), expected_library, rtol=0.05, atol=1e-4)


def test_identity_reconstruction_pearson_on_toy_data():
    import spVIPESmulti.interventions as svi

    prepared = _make_prepared()
    model = _model(prepared)
    encoded = svi.encode_cells(model, prepared)
    kwargs = {
        "model": model,
        "z_shared": encoded["shared"][0][:6],
        "z_private": encoded["private"][0][:6],
        "group_idx": 0,
        "adata": prepared,
        "cells": encoded["obs_indices"][0][:6],
        "library": encoded["library"][0][:6],
        "include_uncertainty": False,
    }
    first = svi.decode_counterfactual(**kwargs).X.ravel()
    second = svi.decode_counterfactual(**kwargs).X.ravel()
    corr = np.corrcoef(first, second)[0, 1]
    assert corr > 0.999


def test_transfer_condition_mean_direction_cosine():
    import spVIPESmulti.interventions as svi

    prepared = _make_prepared()
    model = _model(prepared)
    encoded = svi.encode_cells(model, prepared)
    condition = prepared.obs["condition"].astype(str).to_numpy()
    full_shared = np.zeros((prepared.n_obs, model.module.n_dimensions_shared), dtype=np.float32)
    for g in encoded["shared"]:
        full_shared[encoded["obs_indices"][g]] = encoded["shared"][g]
    expected = full_shared[condition == "stim"].mean(axis=0) - full_shared[condition == "ctrl"].mean(axis=0)

    cells = prepared.uns["groups_obs_indices"][0][:4]
    result = svi.transfer_condition(
        model,
        prepared,
        cells=cells,
        condition_from="ctrl",
        condition_to="stim",
        group_src=0,
        group_dst=1,
    )
    observed = result.info["direction"]
    cosine = float(np.dot(observed, expected) / (np.linalg.norm(observed) * np.linalg.norm(expected) + 1e-8))
    assert cosine > 0.99
    assert result.X.shape[0] == 4


def test_edit_composition_equivalence():
    import spVIPESmulti.interventions as svi

    z = np.zeros((2, 3), dtype=np.float32)
    d1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    d2 = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    sequential = svi.latent_arithmetic(svi.latent_arithmetic(z, d1), d2)
    combined = svi.latent_arithmetic(z, d1 + d2)
    assert np.allclose(sequential, combined)


def test_reject_ood_raise_mode_raises():
    import spVIPESmulti.interventions as svi

    prepared = _make_prepared()
    model = _model(prepared)
    direction = np.full(model.module.n_dimensions_shared, 100.0, dtype=np.float32)
    with pytest.raises(ValueError, match="OOD"):
        svi.predict_counterfactual(
            model,
            prepared,
            cells=prepared.uns["groups_obs_indices"][0][:3],
            group_idx=0,
            direction=direction,
            reject_ood="raise",
        )


def test_multimodal_counterfactual_not_implemented():
    import spVIPESmulti.interventions as svi

    prepared = _make_prepared()
    model = _model(prepared)
    model.module.is_multimodal = True
    with pytest.raises(NotImplementedError):
        svi.encode_cells(model, prepared)
