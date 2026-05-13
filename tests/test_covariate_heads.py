"""Tests for F4-lite covariate registration and auxiliary heads."""

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

import anndata as ad

import spVIPESmulti
from spVIPESmulti.dataloaders._concat_dataloader import ConcatDataLoader

_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_f4_covariate_probes.py"
_BENCHMARK_SPEC = importlib.util.spec_from_file_location("benchmark_f4_covariate_probes", _BENCHMARK_PATH)
assert _BENCHMARK_SPEC is not None and _BENCHMARK_SPEC.loader is not None
_BENCHMARK_MODULE = importlib.util.module_from_spec(_BENCHMARK_SPEC)
sys.modules[_BENCHMARK_SPEC.name] = _BENCHMARK_MODULE
_BENCHMARK_SPEC.loader.exec_module(_BENCHMARK_MODULE)
Config = _BENCHMARK_MODULE.Config
_variant_kwargs = _BENCHMARK_MODULE._variant_kwargs
build_recommendation = _BENCHMARK_MODULE.build_recommendation


def _make_covariate_adata(n_per_group=48, n_genes=30):
    rng = np.random.default_rng(7)
    groups = {}
    for gi, group_name in enumerate(("ctrl", "stim")):
        x = rng.poisson(5, size=(n_per_group, n_genes)).astype(np.float32)
        a = ad.AnnData(X=csr_matrix(x))
        a.obs_names = [f"{group_name}_c{i}" for i in range(n_per_group)]
        a.var_names = [f"g{i}" for i in range(n_genes)]
        a.obs["cell_type"] = rng.choice(["T", "B", "Mono"], size=n_per_group)
        a.obs["condition"] = group_name
        a.obs["donor"] = [f"d{(i + gi) % 4}" for i in range(n_per_group)]
        a.obs["sample"] = a.obs["donor"].to_numpy()
        a.obs["batch"] = [f"batch{i % 2}" for i in range(n_per_group)]
        groups[group_name] = a
    return spVIPESmulti.data.prepare_adatas(groups)


def _setup_full_covariates(prepared):
    spVIPESmulti.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_type",
        condition_key="condition",
        donor_key="donor",
        sample_key="sample",
        batch_key="batch",
    )


def _loss_for_model(model, prepared, batch_size=24):
    gi = [list(map(int, g)) for g in prepared.uns["groups_obs_indices"]]
    dl = ConcatDataLoader(
        model.adata_manager,
        indices_list=gi,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
    )
    tensors = next(iter(dl))
    inference_inputs = model.module._get_inference_input(tensors)
    inference_outputs = model.module.inference(**inference_inputs)
    generative_inputs = model.module._get_generative_input(tensors, inference_outputs)
    generative_outputs = model.module.generative(**generative_inputs)
    return model.module.loss(tensors, inference_outputs, generative_outputs)


def _make_model(prepared, **kwargs):
    return spVIPESmulti.model.spVIPESmulti(
        prepared,
        n_hidden=32,
        n_dimensions_shared=8,
        n_dimensions_private=4,
        use_nf_prior=False,
        disentangle_preset="off",
        **kwargs,
    )


def test_setup_anndata_registers_condition_and_donor_keys():
    prepared = _make_covariate_adata()
    _setup_full_covariates(prepared)
    model = _make_model(prepared)

    assert "condition" in model.adata_manager.data_registry
    assert "donor" in model.adata_manager.data_registry
    assert model.adata_manager.get_state_registry("condition").original_key == "condition"
    assert model.adata_manager.get_state_registry("donor").original_key == "donor"
    assert model.module.use_condition is True
    assert model.module.n_conditions == 2
    assert model.module.use_donor is True
    assert model.module.n_donors == 4


@pytest.mark.parametrize(
    ("weight_name", "message"),
    [
        ("disentangle_donor_shared_weight", "donor_key"),
        ("disentangle_donor_private_weight", "donor_key"),
    ],
)
def test_donor_weight_without_donor_key_raises(weight_name, message):
    prepared = _make_covariate_adata()
    spVIPESmulti.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_type",
        condition_key="condition",
        batch_key="batch",
    )

    with pytest.raises(ValueError, match=message):
        _make_model(prepared, **{weight_name: 0.5})


def test_batch_shared_weight_without_batch_key_raises():
    prepared = _make_covariate_adata()
    spVIPESmulti.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_type",
        donor_key="donor",
    )

    with pytest.raises(ValueError, match="batch_key"):
        _make_model(prepared, disentangle_batch_shared_weight=0.5)


def test_negative_covariate_weight_raises():
    prepared = _make_covariate_adata()
    _setup_full_covariates(prepared)

    with pytest.raises(ValueError, match="must be >= 0"):
        _make_model(prepared, disentangle_donor_private_weight=-0.1)


@pytest.mark.parametrize(
    ("weight_name", "metric_name"),
    [
        ("disentangle_batch_shared_weight", "disentangle_batch_shared_loss"),
        ("disentangle_donor_shared_weight", "disentangle_donor_shared_loss"),
        ("disentangle_donor_private_weight", "disentangle_donor_private_loss"),
    ],
)
def test_covariate_head_emits_finite_metric_when_enabled(weight_name, metric_name):
    prepared = _make_covariate_adata()
    _setup_full_covariates(prepared)
    model = _make_model(prepared, **{weight_name: 0.5})

    loss_output = _loss_for_model(model, prepared)

    assert torch.isfinite(loss_output.loss)
    assert metric_name in loss_output.extra_metrics
    assert np.isfinite(float(loss_output.extra_metrics[metric_name]))


def test_covariate_metrics_absent_when_default_off():
    prepared = _make_covariate_adata()
    _setup_full_covariates(prepared)
    model = _make_model(prepared)

    loss_output = _loss_for_model(model, prepared)

    for key in (
        "disentangle_batch_shared_loss",
        "disentangle_donor_shared_loss",
        "disentangle_donor_private_loss",
        "covariate_grl_lambda",
    ):
        assert key not in loss_output.extra_metrics


def test_covariate_grl_lambda_uses_kl_weight_and_noops_when_disabled():
    prepared = _make_covariate_adata()
    _setup_full_covariates(prepared)
    model = _make_model(
        prepared,
        disentangle_donor_shared_weight=0.5,
    )

    assert model.module._covariate_grl_lambda(0.0) == 0.0
    assert model.module._covariate_grl_lambda(0.5) == 0.5
    assert model.module._covariate_grl_lambda(1.0) == 1.0
    assert model.module._covariate_grl_lambda(2.0) == 1.0

    off_model = _make_model(prepared)
    assert off_model.module._covariate_grl_lambda() == 0.0


def _benchmark_config(batch_key=None):
    return Config(
        run_id="test",
        kang_h5ad_path="unused.h5ad",
        seeds=[0],
        max_epochs=1,
        batch_size=16,
        max_cells_per_condition=10,
        n_top_genes=20,
        n_shared=4,
        n_private=4,
        n_hidden=16,
        condition_key="label",
        donor_key="replicate",
        label_key="cell_type",
        batch_key=batch_key,
    )


def test_f4_benchmark_full_bio_uses_available_covariates_without_batch_key():
    kwargs, skip_note = _variant_kwargs(_benchmark_config(batch_key=None), "full_bio")

    assert skip_note is None
    assert kwargs["disentangle_preset"] == "off"
    assert kwargs["disentangle_donor_shared_weight"] == 0.5
    assert kwargs["disentangle_donor_private_weight"] == 0.5
    assert "disentangle_batch_shared_weight" not in kwargs


def test_f4_benchmark_full_bio_includes_batch_head_when_batch_key_is_set():
    kwargs, skip_note = _variant_kwargs(_benchmark_config(batch_key="batch"), "full_bio")

    assert skip_note is None
    assert kwargs["disentangle_preset"] == "off"
    assert kwargs["disentangle_donor_shared_weight"] == 0.5
    assert kwargs["disentangle_donor_private_weight"] == 0.5
    assert kwargs["disentangle_batch_shared_weight"] == 0.5


def _probe_row(seed, variant, target, latent, balanced_accuracy, notes="ok; probe=ok"):
    return {
        "run_id": "test",
        "timestamp": "2026-05-10T00:00:00+00:00",
        "feature_id": "F4",
        "dataset": "kang_ifnb",
        "seed": seed,
        "variant": variant,
        "n_cells": 100,
        "n_genes": 50,
        "train_wall_time_sec": 1.0,
        "target": target,
        "latent": latent,
        "accuracy": balanced_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "notes": notes,
    }


def _decision_rows(*, donor_private=0.45, donor_shared=0.28, full_private=0.45, full_shared=0.28, full_cell=0.50):
    rows = []
    for seed in (0, 1, 2):
        rows.extend(
            [
                _probe_row(seed, "baseline", "donor", "private", 0.30),
                _probe_row(seed, "baseline", "donor", "shared", 0.45),
                _probe_row(seed, "baseline", "condition", "shared", 0.50),
                _probe_row(seed, "baseline", "cell_type", "shared", 0.50),
                _probe_row(seed, "donor_private", "donor", "private", donor_private),
                _probe_row(seed, "donor_shared", "donor", "shared", donor_shared),
                _probe_row(seed, "full_bio", "donor", "private", full_private),
                _probe_row(seed, "full_bio", "donor", "shared", full_shared),
                _probe_row(seed, "full_bio", "condition", "shared", 0.48),
                _probe_row(seed, "full_bio", "cell_type", "shared", full_cell),
                _probe_row(seed, "batch_shared", "batch", "shared", None, "skipped: Kang batch_key not provided"),
            ]
        )
    return rows


def test_f4_recommendation_passes_probe_gates_with_three_seeds_without_batch():
    rec = build_recommendation(_decision_rows(), _benchmark_config(batch_key=None))

    assert rec["verdict"] == "pass"
    assert rec["promotion"].startswith("probe gates support full_bio")
    gate_status = {g["id"]: g["status"] for g in rec["gates"]}
    assert gate_status["seed_count"] == "pass"
    assert gate_status["donor_private_retention"] == "pass"
    assert gate_status["donor_shared_erasure"] == "pass"
    assert gate_status["batch_shared_erasure"] == "skipped"


def test_f4_recommendation_rejects_failed_donor_private_gate():
    rows = _decision_rows(donor_private=0.31, full_private=0.31)

    rec = build_recommendation(rows, _benchmark_config(batch_key=None))

    assert rec["verdict"] == "reject"
    gate_status = {g["id"]: g["status"] for g in rec["gates"]}
    assert gate_status["donor_private_retention"] == "reject"
    assert gate_status["full_bio_donor_private_retention"] == "reject"
    assert rec["promotion"] == "keep F4 heads opt-in; do not promote presets"


def test_f4_recommendation_keeps_one_seed_run_informational_without_rejects():
    rows = [r for r in _decision_rows() if r["seed"] == 0]

    rec = build_recommendation(rows, _benchmark_config(batch_key=None))

    assert rec["verdict"] == "informational"
    gate_status = {g["id"]: g["status"] for g in rec["gates"]}
    assert gate_status["seed_count"] == "informational"
    assert gate_status["donor_private_retention"] == "pass"
