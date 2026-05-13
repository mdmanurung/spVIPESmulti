"""Tests for F3 optional shared-private orthogonality loss."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import anndata as ad
import numpy as np
import pytest
import torch
from scipy import sparse

import spVIPESmulti as sv
from spVIPESmulti.dataloaders._concat_dataloader import ConcatDataLoader


def _history_keys(model) -> set[str]:
    history = model.history
    if hasattr(history, "keys"):
        return set(history.keys())
    if hasattr(history, "columns"):
        return set(history.columns)
    raise TypeError(f"Unknown history type: {type(history)}")


@pytest.fixture
def minimal_two_group_adata():
    rng = np.random.default_rng(42)
    n_cells, n_genes = 80, 30
    X = sparse.csr_matrix(rng.poisson(lam=5.0, size=(n_cells, n_genes)).astype(np.float32))
    adata_full = ad.AnnData(X)
    adata_full.obs["cell_type"] = ["type_A"] * 40 + ["type_B"] * 40
    adata_full.obs["sample"] = ["s1"] * 20 + ["s2"] * 20 + ["s3"] * 20 + ["s4"] * 20

    adata = sv.data.prepare_adatas(
        {
            "group_0": adata_full[:40].copy(),
            "group_1": adata_full[40:].copy(),
        }
    )
    adata.obs["indices"] = np.arange(adata.n_obs).astype(str)

    sv.model.spVIPESmulti.setup_anndata(
        adata,
        groups_key="groups",
        label_key="cell_type",
        sample_key="sample",
    )
    return adata


def _group_indices(adata):
    return [list(map(int, group_indices)) for group_indices in adata.uns["groups_obs_indices"]]


def _train_tiny_model(adata, **model_kwargs):
    model = sv.model.spVIPESmulti(
        adata,
        n_hidden=32,
        n_dimensions_shared=6,
        n_dimensions_private=4,
        dropout_rate=0.1,
        disentangle_preset="off",
        use_nf_prior=False,
        **model_kwargs,
    )
    model.train(
        _group_indices(adata),
        batch_size=16,
        max_epochs=1,
        train_size=0.9,
        early_stopping=False,
        n_epochs_kl_warmup=0,
        orthogonality_groupby_keys=("sample",),
        orthogonality_min_cells_per_stratum=5,
        accelerator="cpu",
        devices=1,
    )
    return model


def test_orthogonality_weight_defaults_to_zero(minimal_two_group_adata):
    model = _train_tiny_model(minimal_two_group_adata)

    assert model.module.orthogonality_weight == 0.0
    history_cols = _history_keys(model)
    assert all("orthogonality_loss" not in key for key in history_cols)


def test_negative_orthogonality_weight_rejected(minimal_two_group_adata):
    with pytest.raises(ValueError, match="orthogonality_weight"):
        sv.model.spVIPESmulti(
            minimal_two_group_adata,
            disentangle_preset="off",
            orthogonality_weight=-0.1,
        )


def test_correlated_latents_have_higher_orthogonality_loss():
    from spVIPESmulti.module.spVIPESmultimodule import _orthogonality_corr_loss

    torch.manual_seed(0)
    z_shared = torch.randn(80, 6)
    z_private_corr = z_shared[:, :4] + 0.01 * torch.randn(80, 4)
    z_private_independent = torch.randn(80, 4)
    strata_ids = torch.tensor([0] * 40 + [1] * 40)

    corr_loss = _orthogonality_corr_loss(z_shared, z_private_corr, strata_ids, min_cells=10)
    independent_loss = _orthogonality_corr_loss(z_shared, z_private_independent, strata_ids, min_cells=10)

    assert torch.isfinite(corr_loss)
    assert torch.isfinite(independent_loss)
    assert corr_loss > independent_loss


def test_orthogonality_loss_is_differentiable():
    from spVIPESmulti.module.spVIPESmultimodule import _orthogonality_corr_loss

    torch.manual_seed(1)
    z_shared = torch.randn(80, 6, requires_grad=True)
    z_private = z_shared[:, :4].clone().detach().requires_grad_(True)
    strata_ids = torch.tensor([0] * 40 + [1] * 40)

    loss = _orthogonality_corr_loss(z_shared, z_private, strata_ids, min_cells=10)
    loss.backward()

    assert z_shared.grad is not None
    assert z_private.grad is not None
    assert torch.isfinite(z_shared.grad).all()
    assert torch.isfinite(z_private.grad).all()
    assert z_shared.grad.abs().sum() > 0
    assert z_private.grad.abs().sum() > 0


def test_no_eligible_strata_returns_zero_loss():
    from spVIPESmulti.module.spVIPESmultimodule import _orthogonality_corr_loss

    z_shared = torch.randn(8, 6, requires_grad=True)
    z_private = torch.randn(8, 4, requires_grad=True)
    strata_ids = torch.arange(8)

    loss = _orthogonality_corr_loss(z_shared, z_private, strata_ids, min_cells=10)

    assert loss.shape == ()
    assert loss.device == z_shared.device
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()


def test_orthogonality_loss_logged_when_weight_enabled(minimal_two_group_adata):
    model = _train_tiny_model(minimal_two_group_adata, orthogonality_weight=0.1)

    history_cols = _history_keys(model)
    assert any("orthogonality_loss" in key for key in history_cols)


def test_orthogonality_loss_independent_of_f1_metric_flag(minimal_two_group_adata):
    model = _train_tiny_model(
        minimal_two_group_adata,
        orthogonality_weight=0.1,
    )

    history_cols = _history_keys(model)
    assert any("orthogonality_loss" in key for key in history_cols)
    assert all("orthogonality_within_stratum" not in key for key in history_cols)
    assert all("orthogonality_worst_stratum" not in key for key in history_cols)
    assert all("orthogonality_excluded_strata" not in key for key in history_cols)


def _make_mod(n_obs, n_vars, seed):
    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32))
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}_{seed}" for i in range(n_obs)]
    a.var_names = [f"g{i}" for i in range(n_vars)]
    return a


def test_multimodal_orthogonality_loss_runs_and_logs():
    groups = {}
    for gi in range(2):
        gname = f"g{gi}"
        rna = _make_mod(24, 20, seed=10 * gi)
        protein = _make_mod(24, 8, seed=10 * gi + 1)
        rna.obs_names = [f"{gname}_c{i}" for i in range(24)]
        protein.obs_names = rna.obs_names
        rna.obs["cell_type"] = ["ct0"] * 12 + ["ct1"] * 12
        protein.obs["cell_type"] = rna.obs["cell_type"].to_numpy()
        rna.obs["sample"] = ["s0"] * 12 + ["s1"] * 12
        protein.obs["sample"] = rna.obs["sample"].to_numpy()
        groups[gname] = {"rna": rna, "protein": protein}

    prepared = sv.data.prepare_multimodal_adatas(
        groups,
        modality_likelihoods={"rna": "nb", "protein": "nb"},
    )
    sv.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_type",
        sample_key="sample",
        modality_likelihoods={"rna": "nb", "protein": "nb"},
    )
    model = sv.model.spVIPESmulti(
        prepared,
        n_hidden=24,
        n_dimensions_shared=5,
        n_dimensions_private=3,
        disentangle_preset="off",
        orthogonality_weight=0.1,
    )

    scdl = ConcatDataLoader(
        model.adata_manager,
        indices_list=_group_indices(prepared),
        shuffle=False,
        batch_size=12,
        drop_last=False,
    )
    tensors_by_group = next(iter(scdl))
    inference_inputs = model.module._get_inference_input(tensors_by_group)
    inference_outputs = model.module.inference(**inference_inputs)
    generative_inputs = model.module._get_generative_input(tensors_by_group, inference_outputs)
    generative_outputs = model.module.generative(**generative_inputs)
    loss_output = model.module.loss(tensors_by_group, inference_outputs, generative_outputs)

    assert "orthogonality_loss" in loss_output.extra_metrics
    assert torch.isfinite(loss_output.extra_metrics["orthogonality_loss"])
