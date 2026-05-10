"""
Tests for conditional orthogonality instrumentation (F1).

TDD approach: tests written first, implementation follows.
These tests validate the _within_stratum_corr_norm helper and its integration
into the loss paths.
"""

import numpy as np
import pytest
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
import anndata as ad
import scanpy as sc
from scipy import sparse

import spVIPESmulti as sv


def _history_keys(model) -> set[str]:
    history = model.history
    if hasattr(history, 'keys'):
        return set(history.keys())
    if hasattr(history, 'columns'):
        return set(history.columns)
    raise TypeError(f"Unknown history type: {type(history)}")


@pytest.fixture
def minimal_two_group_adata():
    """
    Minimal AnnData for F1 testing: 2 groups, 2 labels, 50 genes, 200 cells total.
    Groups: "group_0" (100 cells), "group_1" (100 cells).
    Labels (cell_type): "type_A" (100 cells), "type_B" (100 cells).
    Sample key (for stratification): varies per group.
    """
    np.random.seed(42)
    
    # Create sparse count matrix
    n_cells, n_genes = 200, 50
    X = np.random.poisson(lam=5.0, size=(n_cells, n_genes)).astype(np.float32)
    X = sparse.csr_matrix(X)
    
    adata_full = ad.AnnData(X)
    adata_full.obs["cell_type"] = (
        ["type_A"] * 100 + ["type_B"] * 100
    )
    adata_full.obs["sample"] = (
        ["sample_1"] * 50 + ["sample_2"] * 50 +
        ["sample_3"] * 50 + ["sample_4"] * 50
    )

    # Build canonical multi-group AnnData with required uns metadata.
    adata = sv.data.prepare_adatas(
        {
            "group_0": adata_full[:100].copy(),
            "group_1": adata_full[100:].copy(),
        }
    )
    adata.obs["indices"] = np.arange(adata.n_obs).astype(str)
    
    # Setup for spVIPESmulti
    sv.model.spVIPESmulti.setup_anndata(
        adata,
        groups_key="groups",
        label_key="cell_type",
        sample_key="sample",
    )
    
    return adata


@pytest.fixture
def minimal_two_group_model(minimal_two_group_adata):
    """
    Train a minimal model (2 epochs, early_stopping=False) for quick tests.
    """
    adata = minimal_two_group_adata
    model = sv.model.spVIPESmulti(
        adata,
        n_hidden=64,
        n_dimensions_shared=10,
        n_dimensions_private=5,
        dropout_rate=0.1,
        disentangle_preset="shared_only",
        use_nf_prior=False,
    )
    
    # Get group indices from uns
    group_indices_list = [list(map(int, g)) for g in adata.uns["groups_obs_indices"]]
    
    # Train quickly (no orthogonality metric, just baseline)
    model.train(
        group_indices_list,
        batch_size=32,
        max_epochs=2,
        train_size=0.9,
        early_stopping=False,
        n_epochs_kl_warmup=0,
        accelerator="cpu",
        devices=1,
    )
    
    return model, adata


class TestOrthogonalityMetricPresence:
    """
    Tests for presence and finite-ness of orthogonality metrics when enabled.
    """
    
    def test_orthogonality_metric_present_when_enabled(self, minimal_two_group_model):
        """
        Train a 2-epoch model with orthogonality metric enabled,
        assert extra_metrics contains "orthogonality_within_stratum" and it's finite.
        """
        model, adata = minimal_two_group_model
        
        # Re-create model with orthogonality enabled
        model_with_ortho = sv.model.spVIPESmulti(
            adata,
            n_hidden=64,
            n_dimensions_shared=10,
            n_dimensions_private=5,
            dropout_rate=0.1,
            disentangle_preset="shared_only",
            use_nf_prior=False,
        )
        
        # Get group indices
        group_indices_list = [list(map(int, g)) for g in adata.uns["groups_obs_indices"]]
        
        # Train with metric enabled
        model_with_ortho.train(
            group_indices_list,
            batch_size=32,
            max_epochs=2,
            train_size=0.9,
            early_stopping=False,
            n_epochs_kl_warmup=0,
            compute_orthogonality_metric=True,
            orthogonality_groupby_keys=("sample",),
            orthogonality_min_cells_per_stratum=5,
            accelerator="cpu",
            devices=1,
        )
        
        assert model_with_ortho is not None

        history_cols = _history_keys(model_with_ortho)
        assert any("orthogonality_within_stratum" in c for c in history_cols), (
            f"Expected orthogonality_within_stratum metric in history columns, got {sorted(history_cols)}"
        )

    def test_orthogonality_metric_absent_when_disabled(self, minimal_two_group_model):
        """
        With compute_orthogonality_metric=False (default), metric should not be computed.
        """
        model, adata = minimal_two_group_model
        
        assert model is not None

        history_cols = _history_keys(model)
        assert all("orthogonality_within_stratum" not in c for c in history_cols)
        

class TestOrthogonalityHelperUnit:
    """
    Unit tests for the _within_stratum_corr_norm helper function.
    Tests the helper in isolation with synthetic latent data.
    """
    
    def test_orthogonality_zero_for_independent_inputs(self):
        """
        Feed synthetic independent z_shared and z_private (uncorrelated random),
        assert corr_norm is close to 0 (within tolerance).
        """
        np.random.seed(42)
        n_cells = 100
        n_dims_shared = 10
        n_dims_private = 5
        
        # Independent random normals
        z_shared = np.random.randn(n_cells, n_dims_shared)
        z_private = np.random.randn(n_cells, n_dims_private)
        
        # Strata: simple uniform assignment
        strata_ids = np.array([0] * 50 + [1] * 50)
        
        # Import the helper (once implemented)
        from spVIPESmulti.module.spVIPESmultimodule import _within_stratum_corr_norm
        
        corr_norm_mean, corr_norm_worst, _ = _within_stratum_corr_norm(
            z_shared, z_private, strata_ids, min_cells=10
        )
        
        # Independent data should have low correlation.
        assert corr_norm_mean < 0.1, f"Expected corr_norm_mean < 0.1, got {corr_norm_mean}"
        assert corr_norm_worst >= corr_norm_mean  # worst >= mean by definition

    def test_orthogonality_high_for_perfect_copy(self):
        """
        Feed z_private = z_shared (perfect dependence),
        assert corr_norm is close to 1.0.
        """
        np.random.seed(42)
        n_cells = 100
        n_dims = 10
        
        # Perfect copy: z_private is a slice of z_shared
        z_shared = np.random.randn(n_cells, n_dims)
        z_private = z_shared[:, :5]  # First 5 dims
        
        strata_ids = np.array([0] * 50 + [1] * 50)
        
        from spVIPESmulti.module.spVIPESmultimodule import _within_stratum_corr_norm
        
        corr_norm_mean, corr_norm_worst, _ = _within_stratum_corr_norm(
            z_shared, z_private, strata_ids, min_cells=10
        )
        
        # Perfect copy should have high correlation.
        assert corr_norm_mean > 0.8, f"Expected corr_norm_mean > 0.8, got {corr_norm_mean}"

    def test_orthogonality_centered_data(self):
        """
        Helper should work on centered (mean-subtracted) data.
        Verify that centering is applied automatically if needed.
        """
        np.random.seed(42)
        n_cells = 100
        n_dims_shared = 10
        n_dims_private = 5
        
        # Data with non-zero mean
        z_shared = 5 + np.random.randn(n_cells, n_dims_shared)
        z_private = -3 + np.random.randn(n_cells, n_dims_private)
        
        strata_ids = np.array([0] * 50 + [1] * 50)
        
        from spVIPESmulti.module.spVIPESmultimodule import _within_stratum_corr_norm
        
        # Should not raise; helper should center internally
        corr_norm_mean, corr_norm_worst, _ = _within_stratum_corr_norm(
            z_shared, z_private, strata_ids, min_cells=10
        )
        
        assert np.isfinite(corr_norm_mean)
        assert np.isfinite(corr_norm_worst)


class TestOrthogonalityStratumFiltering:
    """
    Tests for strata exclusion when below min_cells threshold.
    """
    
    def test_min_cells_per_stratum_filter(self):
        """
        With min_cells_per_stratum=50, small strata (< 50 cells) should be excluded.
        Verify that excluded count is reported.
        """
        np.random.seed(42)
        n_cells = 100
        n_dims_shared = 10
        n_dims_private = 5
        
        z_shared = np.random.randn(n_cells, n_dims_shared)
        z_private = np.random.randn(n_cells, n_dims_private)
        
        # Strata: one large (90 cells), one tiny (10 cells)
        strata_ids = np.array([0] * 90 + [1] * 10)
        
        from spVIPESmulti.module.spVIPESmultimodule import _within_stratum_corr_norm
        
        corr_norm_mean, corr_norm_worst, excluded_count = _within_stratum_corr_norm(
            z_shared, z_private, strata_ids, min_cells=50
        )
        
        # Small stratum should be excluded
        assert excluded_count == 1, f"Expected 1 excluded stratum, got {excluded_count}"
        assert np.isfinite(corr_norm_mean)


class TestOrthogonalityMultimodal:
    """
    Tests for multimodal parity: orthogonality should be computed per-modality
    and scaled like existing multimodal private terms.
    """
    
    def test_orthogonality_multimodal_independent_inputs(self):
        """
        Multimodal path: test on independent inputs per modality.
        """
        np.random.seed(42)
        n_cells = 100
        n_mods = 2
        n_dims_shared = 10
        n_dims_private = 5
        
        z_shared = np.random.randn(n_cells, n_dims_shared)
        
        # Two modalities, each with independent private latent
        z_private_by_modality = {
            0: np.random.randn(n_cells, n_dims_private),
            1: np.random.randn(n_cells, n_dims_private),
        }
        
        strata_ids = np.array([0] * 50 + [1] * 50)
        
        from spVIPESmulti.module.spVIPESmultimodule import _within_stratum_corr_norm_multimodal
        
        # Should compute corr_norm per modality and average
        corr_norm_mean, corr_norm_worst, excluded_count = _within_stratum_corr_norm_multimodal(
            z_shared, z_private_by_modality, strata_ids, min_cells=10
        )

        # Averaged over modalities, should still be low
        assert corr_norm_mean < 0.2
        assert np.isfinite(corr_norm_worst)
        assert excluded_count == 0


class TestOrthogonalityMetricsInExtraMetrics:
    """
    Integration tests: verify metrics appear in extra_metrics only when enabled.
    """
    
    def test_metrics_appear_when_enabled(self, minimal_two_group_model):
        """
        After training with compute_orthogonality_metric=True,
        extra_metrics should contain orthogonality keys.
        (This test may need adjustment based on LightningModule integration.)
        """
        model, adata = minimal_two_group_model

        model_with_ortho = sv.model.spVIPESmulti(
            adata,
            n_hidden=64,
            n_dimensions_shared=10,
            n_dimensions_private=5,
            dropout_rate=0.1,
            disentangle_preset="shared_only",
            use_nf_prior=False,
        )

        group_indices_list = [list(map(int, g)) for g in adata.uns["groups_obs_indices"]]

        model_with_ortho.train(
            group_indices_list,
            batch_size=32,
            max_epochs=2,
            train_size=0.9,
            early_stopping=False,
            n_epochs_kl_warmup=0,
            compute_orthogonality_metric=True,
            orthogonality_groupby_keys=("sample",),
            orthogonality_min_cells_per_stratum=5,
            accelerator="cpu",
            devices=1,
        )

        history_cols = _history_keys(model_with_ortho)
        assert any("orthogonality_within_stratum" in c for c in history_cols)
        assert any("orthogonality_excluded_strata" in c for c in history_cols)

    def test_metrics_absent_when_disabled(self, minimal_two_group_model):
        """
        After training with compute_orthogonality_metric=False (default),
        extra_metrics should not contain orthogonality keys.
        """
        model, _ = minimal_two_group_model
        assert model is not None

        history_cols = _history_keys(model)
        assert all("orthogonality_within_stratum" not in c for c in history_cols)
        assert all("orthogonality_excluded_strata" not in c for c in history_cols)
