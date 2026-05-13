"""Unit tests for model.evaluate() – R4 public evaluation API.

All tests use the same dummy-model pattern as test_enrichment.py:
direct method dispatch avoids scvi-tools setup and heavy fixtures.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spVIPESmulti.model.spvipesmulti import spVIPESmulti as _ModelClass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adata(n_cells: int = 8, n_groups: int = 2, n_dims: int = 4):
    """Minimal AnnData with group obs column and shared latent in obsm."""
    rng = np.random.default_rng(0)
    x = rng.poisson(2, size=(n_cells, 6)).astype(np.float32)
    adata = ad.AnnData(X=x)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    per_group = n_cells // n_groups
    group_names = [f"grp{g}" for g in range(n_groups)]
    adata.obs["groups"] = pd.Categorical([group_names[i // per_group] for i in range(n_cells)])
    adata.obs["cell_type"] = pd.Categorical(["T" if i % 2 == 0 else "B" for i in range(n_cells)])
    adata.obsm["X_spvm_shared"] = rng.normal(size=(n_cells, n_dims)).astype(np.float32)
    adata.uns["groups_mapping"] = {g: group_names[g] for g in range(n_groups)}
    adata.uns["groups_obs_indices"] = [list(range(g * per_group, (g + 1) * per_group)) for g in range(n_groups)]
    return adata


def _make_model(adata: ad.AnnData, n_dims: int = 4):
    """Minimal model stub that satisfies evaluate()'s call contract."""
    model = _ModelClass.__new__(_ModelClass)
    model._adata = adata
    model._group_indices_auto_infer_warned = False

    def _validate_anndata(a=None):
        return adata if a is None else a

    model._validate_anndata = _validate_anndata

    # Provide a fake get_latent_representation that returns correct shapes.
    def _fake_glr(
        group_indices_list=None,
        adata=None,
        normalized=False,
        give_mean=True,
        mc_samples=1,
        batch_size=None,
        drop_last=None,
    ):
        rng = np.random.default_rng(1)
        result = {"shared_reordered": {}, "private_reordered": {}}
        for gi, idxs in enumerate(group_indices_list or []):
            n = len(idxs)
            result["shared_reordered"][gi] = rng.normal(size=(n, n_dims)).astype(np.float32)
            result["private_reordered"][gi] = rng.normal(size=(n, n_dims)).astype(np.float32)
        return result

    model.get_latent_representation = _fake_glr
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluateReturnSchema:
    def test_keys_present(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        assert set(out.keys()) == {"metrics", "metadata", "held_out_metrics", "warnings"}

    def test_metrics_is_dataframe(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        assert isinstance(out["metrics"], pd.DataFrame)

    def test_metrics_has_expected_columns(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        expected = {"latent", "ilisi", "clisi", "kbet", "knn_purity", "leiden_ari", "silhouette"}
        assert expected.issubset(set(out["metrics"].columns))

    def test_metrics_contains_z_shared_row(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        assert "z_shared" in out["metrics"]["latent"].values

    def test_metadata_fields_present(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        md = out["metadata"]
        for key in (
            "n_cells",
            "n_groups",
            "k",
            "label_key",
            "leiden_resolution",
            "include_private",
            "used_precomputed_embedding",
        ):
            assert key in md

    def test_metadata_n_cells_correct(self):
        adata = _make_adata(n_cells=8)
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        assert out["metadata"]["n_cells"] == 8

    def test_metadata_n_groups_correct(self):
        adata = _make_adata(n_groups=2)
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        assert out["metadata"]["n_groups"] == 2

    def test_used_precomputed_true_when_key_present(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        assert out["metadata"]["used_precomputed_embedding"] is True

    def test_warnings_is_list(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        assert isinstance(out["warnings"], list)

    def test_held_out_metrics_defaults_to_none_without_history(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        assert out["held_out_metrics"] is None


class TestEvaluateEmbeddingFallback:
    def test_warns_when_precomputed_key_missing(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_does_not_exist", k=3)
        assert any("not found in adata.obsm" in w for w in out["warnings"])

    def test_used_precomputed_false_when_key_missing(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_does_not_exist", k=3)
        assert out["metadata"]["used_precomputed_embedding"] is False

    def test_extracts_embedding_when_no_key_given(self):
        """evaluate() should work without z_shared_key by calling get_latent_representation."""
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, k=3)
        assert "z_shared" in out["metrics"]["latent"].values
        assert out["metadata"]["used_precomputed_embedding"] is False


class TestEvaluateLabelHandling:
    def test_label_key_populates_metadata(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", label_key="cell_type", k=3)
        assert out["metadata"]["label_key"] == "cell_type"

    def test_missing_label_key_emits_warning(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", label_key="nonexistent_col", k=3)
        assert any("nonexistent_col" in w and "not found" in w for w in out["warnings"])

    def test_no_label_key_no_warning(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)
        # Without label_key, no label-specific warning should be emitted.
        label_warnings = [w for w in out["warnings"] if "label_key" in w and "not found" in w]
        assert len(label_warnings) == 0


class TestEvaluatePrivateLatents:
    def test_include_private_adds_rows(self):
        adata = _make_adata(n_groups=2)
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3, include_private=True)
        # Should have 1 shared row + 2 private rows
        df = out["metrics"]
        private_rows = df[df["latent"].str.startswith("z_private")]
        assert len(private_rows) == 2

    def test_include_private_false_only_shared_row(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3, include_private=False)
        assert len(out["metrics"]) == 1
        assert out["metrics"].iloc[0]["latent"] == "z_shared"

    def test_include_private_metadata_flag(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3, include_private=True)
        assert out["metadata"]["include_private"] is True

    def test_finite_metrics_on_shared_row(self):
        adata = _make_adata()
        model = _make_model(adata)
        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", label_key="cell_type", k=3)
        row = out["metrics"][out["metrics"]["latent"] == "z_shared"].iloc[0]
        # ilisi, kbet, knn_purity must be finite scalars
        for col in ("ilisi", "kbet", "knn_purity"):
            assert np.isfinite(row[col]), f"{col} is not finite: {row[col]}"


class TestEvaluateHeldOutMetrics:
    def test_reads_latest_validation_metrics_from_history(self):
        adata = _make_adata()
        model = _make_model(adata)
        model.validation_indices = [np.array([0, 1]), np.array([2, 3])]
        model.history_ = {
            "elbo_validation": pd.DataFrame({"elbo_validation": [12.0, 8.5]}),
            "reconstruction_loss_validation": pd.DataFrame({"reconstruction_loss_validation": [10.0, 7.25]}),
            "kl_local_validation": pd.DataFrame({"kl_local_validation": [2.0, 1.25]}),
            "validation_loss": pd.DataFrame({"validation_loss": [10.0, 7.25]}),
        }

        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)

        assert out["held_out_metrics"] is not None
        assert out["held_out_metrics"]["elbo_validation"] == pytest.approx(8.5)
        assert out["held_out_metrics"]["reconstruction_loss_validation"] == pytest.approx(7.25)
        assert out["held_out_metrics"]["held_out_nll"] == pytest.approx(7.25)

    def test_warns_when_validation_indices_exist_but_history_is_missing(self):
        adata = _make_adata()
        model = _make_model(adata)
        model.validation_indices = [np.array([0, 1]), np.array([2, 3])]
        model.history_ = {}

        out = _ModelClass.evaluate(model, adata=adata, z_shared_key="X_spvm_shared", k=3)

        assert out["held_out_metrics"] is None
        assert any("validation metrics were not found in model.history" in msg for msg in out["warnings"])
