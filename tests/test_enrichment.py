"""Unit tests for enrichment helper APIs on spVIPESmulti model.

These tests focus on deterministic API behavior and optional dependency handling.
"""

from __future__ import annotations

import builtins
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spVIPESmulti.model.spvipesmulti import spVIPESmulti as _ModelClass


class _DummyModel:
    """Minimal stand-in exposing _validate_anndata used by model helpers."""

    def _validate_anndata(self, adata=None):
        if adata is None:
            raise ValueError("adata must be provided in tests")
        return adata

    def summarize_enrichment(self, *args, **kwargs):
        return _ModelClass.summarize_enrichment(self, *args, **kwargs)


@pytest.fixture
def adata_small():
    x = np.random.default_rng(0).poisson(2, size=(8, 6)).astype(np.float32)
    a = ad.AnnData(X=x)
    a.var_names = [f"Gene{i}" for i in range(6)]
    a.obs_names = [f"Cell{i}" for i in range(8)]
    a.obs["group"] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    return a


@pytest.fixture
def network_table():
    return pd.DataFrame(
        {
            "source": ["TF1", "TF1", "TF1", "TF1", "TF1", "TF2", "TF2", "TF2", "TF2", "TF2"],
            "target": [
                "Gene0",
                "Gene1",
                "Gene2",
                "Gene3",
                "Gene4",
                "Gene1",
                "Gene2",
                "Gene3",
                "Gene4",
                "Gene5",
            ],
            "weight": [1.0, 0.8, 0.6, 0.5, 0.4, -1.0, -0.8, -0.6, -0.5, -0.4],
        }
    )


def _make_fake_decoupler(n_programs: int = 2):
    def _runner(method_name: str):
        def _run(*, data, **kwargs):
            n_obs = data.n_obs
            vals = np.full((n_obs, n_programs), fill_value=float(len(method_name)), dtype=np.float32)
            data.obsm[f"score_{method_name}"] = vals

        return _run

    mt = types.SimpleNamespace(
        ora=_runner("ora"),
        gsea=_runner("gsea"),
        ulm=_runner("ulm"),
    )
    return types.SimpleNamespace(mt=mt)


def test_get_enrichment_scores_importerror_has_actionable_message(monkeypatch, adata_small, network_table):
    dummy = _DummyModel()

    real_import = builtins.__import__

    def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "decoupler":
            raise ImportError("mocked missing decoupler")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    with pytest.raises(ImportError, match=r"pip install -e .\[enrichment\]"):
        _ModelClass.get_enrichment_scores(
            dummy,
            network_table,
            adata=adata_small,
            methods=["ora"],
            write_to_adata=False,
        )


def test_get_enrichment_scores_fake_decoupler_path_writes_outputs(monkeypatch, adata_small, network_table):
    dummy = _DummyModel()
    fake_dc = _make_fake_decoupler(n_programs=3)
    monkeypatch.setitem(sys.modules, "decoupler", fake_dc)

    out = _ModelClass.get_enrichment_scores(
        dummy,
        network_table,
        adata=adata_small,
        methods=["ora", "gsea", "ulm"],
        write_to_adata=True,
        obsm_key="X_spvm_enrichment",
        uns_key="spvm_enrichment",
        overwrite=True,
    )

    scores_df = out["scores_df"]
    assert scores_df.shape == (adata_small.n_obs, 9)
    assert all(c.startswith(("ora__", "gsea__", "ulm__")) for c in scores_df.columns)

    assert "X_spvm_enrichment" in adata_small.obsm
    assert "spvm_enrichment" in adata_small.uns
    assert adata_small.obsm["X_spvm_enrichment"].shape == (adata_small.n_obs, 9)


def test_get_enrichment_scores_rejects_unsupported_methods(adata_small, network_table):
    dummy = _DummyModel()
    with pytest.raises(ValueError, match="Unsupported method"):
        _ModelClass.get_enrichment_scores(
            dummy,
            network_table,
            adata=adata_small,
            methods=["ora", "unknown"],
            write_to_adata=False,
        )


def test_get_enrichment_scores_overwrite_guard_prevents_mutation(monkeypatch, adata_small, network_table):
    dummy = _DummyModel()
    fake_dc = _make_fake_decoupler(n_programs=2)
    monkeypatch.setitem(sys.modules, "decoupler", fake_dc)

    adata_small.obsm["X_spvm_enrichment"] = np.zeros((adata_small.n_obs, 2), dtype=np.float32)
    before = adata_small.obsm["X_spvm_enrichment"].copy()

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _ModelClass.get_enrichment_scores(
            dummy,
            network_table,
            adata=adata_small,
            methods=["ora"],
            write_to_adata=True,
            obsm_key="X_spvm_enrichment",
            uns_key="spvm_enrichment",
            overwrite=False,
        )

    np.testing.assert_allclose(before, adata_small.obsm["X_spvm_enrichment"])


def test_summarize_enrichment_aggregates_by_obs_column(adata_small):
    dummy = _DummyModel()
    scores_df = pd.DataFrame(
        np.arange(adata_small.n_obs * 2, dtype=float).reshape(adata_small.n_obs, 2),
        index=adata_small.obs_names,
        columns=["ora__0", "ulm__0"],
    )
    summary = _ModelClass.summarize_enrichment(dummy, scores_df, "group", adata=adata_small, agg="mean")
    assert list(summary.index) == ["A", "B"]
    assert list(summary.columns) == ["ora__0", "ulm__0"]


def test_interpretation_report_returns_top_programs_and_warning_without_metrics(adata_small):
    dummy = _DummyModel()
    scores_df = pd.DataFrame(
        np.arange(adata_small.n_obs * 3, dtype=float).reshape(adata_small.n_obs, 3),
        index=adata_small.obs_names,
        columns=["ora__0", "gsea__0", "ulm__0"],
    )
    report = _ModelClass.interpretation_report(
        dummy,
        scores_df,
        "group",
        adata=adata_small,
        top_n=2,
    )
    assert "enrichment_summary" in report
    assert "top_programs" in report
    assert report["integration_metrics"] is None
    assert len(report["warnings"]) == 1
    assert report["top_programs"].shape[0] == 2


def test_interpretation_report_computes_metrics_when_inputs_present(adata_small):
    dummy = _DummyModel()
    adata_small.obs["groups"] = adata_small.obs["group"].values
    adata_small.obs["cell_type"] = ["T", "T", "B", "B", "T", "T", "B", "B"]
    adata_small.obsm["X_spvm_shared"] = np.random.default_rng(1).normal(size=(adata_small.n_obs, 4)).astype(np.float32)

    scores_df = pd.DataFrame(
        np.random.default_rng(2).normal(size=(adata_small.n_obs, 3)),
        index=adata_small.obs_names,
        columns=["ora__0", "gsea__0", "ulm__0"],
    )
    report = _ModelClass.interpretation_report(
        dummy,
        scores_df,
        "group",
        adata=adata_small,
        top_n=2,
        label_key="cell_type",
        k=3,
    )
    assert report["integration_metrics"] is not None
    assert "latent" in report["integration_metrics"].columns


@pytest.mark.integration
def test_get_enrichment_scores_real_decoupler_ulm_runs():
    dc = pytest.importorskip("decoupler")
    dummy = _DummyModel()
    adata, net = dc.ds.toy()

    out = _ModelClass.get_enrichment_scores(
        dummy,
        net,
        adata=adata,
        methods=["ulm"],
        tmin=1,
        write_to_adata=True,
        overwrite=True,
    )

    assert out["scores_df"].shape[0] == adata.n_obs
    assert out["scores_df"].shape[1] > 0
    assert "X_spvm_enrichment" in adata.obsm
