"""Unit tests for F2 safe counterfactual helpers."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

import anndata as ad

import spVIPESmulti as sv


def _make_intervention_adata(n_per_group=16, n_genes=12):
    rng = np.random.default_rng(11)
    groups = {}
    for gi, group_name in enumerate(("batch0", "batch1")):
        x = rng.poisson(4 + gi, size=(n_per_group, n_genes)).astype(np.float32)
        a = ad.AnnData(X=csr_matrix(x))
        a.obs_names = [f"{group_name}_c{i}" for i in range(n_per_group)]
        a.var_names = [f"g{i}" for i in range(n_genes)]
        a.obs["cell_type"] = np.where(np.arange(n_per_group) % 2 == 0, "T", "B")
        a.obs["condition"] = np.where(np.arange(n_per_group) % 2 == 0, "ctrl", "stim")
        a.obs["donor"] = [f"d{i % 3}" for i in range(n_per_group)]
        groups[group_name] = a
    prepared = sv.data.prepare_adatas(groups)
    sv.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_type",
        condition_key="condition",
        donor_key="donor",
    )
    return prepared


def _make_model(prepared):
    return sv.model.spVIPESmulti(
        prepared,
        n_hidden=16,
        n_dimensions_shared=4,
        n_dimensions_private=3,
        dropout_rate=0.0,
        disentangle_preset="off",
    )


def test_latent_operators_preserve_shape_dtype_and_finiteness():
    import spVIPESmulti.interventions as svi

    z_np = np.ones((3, 4), dtype=np.float32)
    direction_np = np.arange(4, dtype=np.float32)
    shifted_np = svi.condition_centroid_shift(z_np, direction_np, alpha=0.5)
    assert shifted_np.shape == z_np.shape
    assert shifted_np.dtype == z_np.dtype
    assert np.isfinite(shifted_np).all()

    z_t = torch.ones(3, 4, dtype=torch.float32)
    direction_t = torch.arange(4, dtype=torch.float32)
    shifted_t = svi.latent_arithmetic(z_t, direction_t, weight=0.25)
    assert torch.is_tensor(shifted_t)
    assert shifted_t.shape == z_t.shape
    assert shifted_t.dtype == z_t.dtype
    assert torch.isfinite(shifted_t).all()

    interp = svi.latent_interpolation(z_np, z_np + 2.0, alpha=0.25)
    assert np.allclose(interp, z_np + 0.5)


def test_latent_replacement_validates_dimension():
    import spVIPESmulti.interventions as svi

    with pytest.raises(IndexError):
        svi.latent_replacement(np.zeros((2, 3), dtype=np.float32), dimension=5, value=1.0)


def test_import_spvipesmulti_exposes_interventions():
    import spVIPESmulti.interventions as svi

    assert hasattr(sv, "interventions")
    assert hasattr(svi, "predict_counterfactual")
    assert hasattr(svi, "condition_centroid_shift")


def test_encode_cells_returns_posterior_loc_scale_and_metadata():
    import spVIPESmulti.interventions as svi

    prepared = _make_intervention_adata()
    model = _make_model(prepared)
    encoded = svi.encode_cells(model, prepared)

    assert set(encoded["shared"]) == {0, 1}
    assert encoded["shared"][0].shape == (16, 4)
    assert encoded["private"][1].shape == (16, 3)
    assert encoded["shared_scale"][0].shape == (16, 4)
    assert encoded["private_scale"][1].shape == (16, 3)
    assert np.all(encoded["shared_scale"][0] > 0)
    assert encoded["library"][0].shape == (16, 1)
    assert encoded["batch_index"][0].shape == (16, 1)
    assert len(encoded["obs_names"][0]) == 16


def test_decode_counterfactual_returns_target_gene_space():
    import spVIPESmulti.interventions as svi

    prepared = _make_intervention_adata()
    model = _make_model(prepared)
    encoded = svi.encode_cells(model, prepared)
    cells = encoded["obs_indices"][0][:4]
    result = svi.decode_counterfactual(
        model,
        encoded["shared"][0][:4],
        encoded["private"][0][:4],
        group_idx=1,
        adata=prepared,
        cells=cells,
        library=encoded["library"][0][:4],
    )

    assert result.X.shape == (4, len(model.module.groups_var_indices[1]))
    assert result.X.dtype == np.float32
    assert np.isfinite(result.X).all()
    assert len(result.info["var_names"]) == result.X.shape[1]


def test_predict_counterfactual_centroid_shift_reports_ood_flags():
    import spVIPESmulti.interventions as svi

    prepared = _make_intervention_adata()
    model = _make_model(prepared)
    direction = np.full(model.module.n_dimensions_shared, 0.1, dtype=np.float32)
    result = svi.predict_counterfactual(
        model,
        prepared,
        cells=prepared.uns["groups_obs_indices"][0][:3],
        group_idx=0,
        intervention="centroid_shift",
        direction=direction,
        return_uncertainty=True,
    )

    assert result.X.shape[0] == 3
    assert result.uncertainty is not None
    assert result.uncertainty.shape == result.X.shape
    assert "ood_flags" in result.info
    assert "rejected_mask" in result.info
    assert result.info["rejected_mask"].shape == (3,)


def test_group_idx_bounds_error():
    import spVIPESmulti.interventions as svi

    prepared = _make_intervention_adata()
    model = _make_model(prepared)
    with pytest.raises(ValueError, match="group_idx"):
        svi.encode_cells(model, prepared, group_idx=3)
