"""Tests for spVIPESmulti.utils, spVIPESmulti.pl, and spVIPESmulti.metrics.

These tests are unit tests that do NOT require a trained model or scvi-tools
integration. They use synthetic numpy arrays and mock objects.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import modules under test directly (avoids full spVIPESmulti import chain)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent / "src"


def _load(name: str):
    path = _ROOT / "spVIPESmulti" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"spVIPESmulti.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"spVIPESmulti.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


utils = _load("utils")
metrics = _load("metrics")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CELLS = 120
N_DIMS_SHARED = 6
N_DIMS_PRIVATE = 4
N_GENES = 20
N_GROUPS = 3
GENE_NAMES = [f"Gene{i}" for i in range(N_GENES)]


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def adata(rng):
    """Minimal AnnData with synthetic obsm entries."""
    import anndata as ad

    n = N_CELLS
    X = rng.random((n, N_GENES)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "group": np.repeat(["g0", "g1", "g2"], [40, 40, 40]),
            "cell_type": np.tile(["A", "B", "C", "D"], 30),
        }
    )
    var = pd.DataFrame(index=GENE_NAMES)
    ad_obj = ad.AnnData(X=X, obs=obs, var=var)
    ad_obj.obsm["X_spVIPESmulti_shared"] = rng.random((n, N_DIMS_SHARED)).astype(np.float32)
    # Full-length private array (zeros for cells outside group 0) — matches store_latents output
    ad_obj.obsm["X_spVIPESmulti_private_g0"] = rng.random((n, N_DIMS_PRIVATE)).astype(np.float32)
    return ad_obj


@pytest.fixture
def loadings_df_shared():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((N_GENES, N_DIMS_SHARED)).astype(np.float32)
    cols = [f"Z_shared_{i}" for i in range(N_DIMS_SHARED)]
    return pd.DataFrame(data, index=GENE_NAMES, columns=cols)


@pytest.fixture
def loadings_df_private():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((N_GENES, N_DIMS_PRIVATE)).astype(np.float32)
    cols = [f"Z_private_{i}" for i in range(N_DIMS_PRIVATE)]
    return pd.DataFrame(data, index=GENE_NAMES, columns=cols)


@pytest.fixture
def mock_model(loadings_df_shared, loadings_df_private):
    """Minimal mock that mimics model.get_loadings()."""
    m = MagicMock()
    m.get_loadings.return_value = {
        (0, "shared"): loadings_df_shared,
        (0, "private"): loadings_df_private,
    }
    return m


@pytest.fixture
def group_indices_list():
    return [np.arange(0, 40), np.arange(40, 80), np.arange(80, 120)]


@pytest.fixture
def fake_latents(rng):
    """Synthetic latents dict mimicking get_latent_representation() output."""
    shared = {i: rng.random((40, N_DIMS_SHARED)).astype(np.float32) for i in range(3)}
    private = {i: rng.random((40, N_DIMS_PRIVATE)).astype(np.float32) for i in range(3)}
    shared_r = {i: v.copy() for i, v in shared.items()}
    private_r = {i: v.copy() for i, v in private.items()}
    return {
        "shared": shared,
        "private": private,
        "shared_reordered": shared_r,
        "private_reordered": private_r,
    }


# ===========================================================================
# utils._validate_loadings_df
# ===========================================================================


class TestValidateLoadingsDf:
    def test_valid_shared(self, loadings_df_shared):
        utils._validate_loadings_df(loadings_df_shared, "shared")  # no exception

    def test_valid_private(self, loadings_df_private):
        utils._validate_loadings_df(loadings_df_private, "private")

    def test_not_dataframe_raises(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            utils._validate_loadings_df(np.ones((5, 3)), "shared")

    def test_nan_raises(self, loadings_df_shared):
        df = loadings_df_shared.copy()
        df.iloc[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            utils._validate_loadings_df(df, "shared")

    def test_wrong_prefix_raises(self, loadings_df_shared):
        df = loadings_df_shared.rename(columns={c: c.replace("Z_shared_", "Z_private_") for c in loadings_df_shared.columns})
        with pytest.raises(ValueError, match="Z_shared_"):
            utils._validate_loadings_df(df, "shared")

    def test_non_integer_suffix_raises(self):
        df = pd.DataFrame(
            np.ones((5, 2)),
            columns=["Z_shared_a", "Z_shared_b"],
            index=[f"g{i}" for i in range(5)],
        )
        with pytest.raises(ValueError, match="integers"):
            utils._validate_loadings_df(df, "shared")

    def test_non_contiguous_indices_raises(self):
        df = pd.DataFrame(
            np.ones((5, 2)),
            columns=["Z_shared_0", "Z_shared_2"],  # missing 1
            index=[f"g{i}" for i in range(5)],
        )
        with pytest.raises(ValueError, match="contiguous"):
            utils._validate_loadings_df(df, "shared")


# ===========================================================================
# utils._resolve_loadings
# ===========================================================================


class TestResolveLoadings:
    def test_uses_provided_df(self, loadings_df_shared, mock_model):
        result = utils._resolve_loadings(loadings_df_shared, mock_model, 0, "shared")
        assert result is loadings_df_shared
        mock_model.get_loadings.assert_not_called()

    def test_fetches_from_model_when_none(self, mock_model):
        result = utils._resolve_loadings(None, mock_model, 0, "shared")
        mock_model.get_loadings.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_raises_when_both_none(self):
        with pytest.raises(ValueError, match="loadings_df.*model"):
            utils._resolve_loadings(None, None, 0, "shared")


# ===========================================================================
# utils.store_latents
# ===========================================================================


class TestStoreLatents:
    def test_shared_written(self, adata, fake_latents, group_indices_list):
        import anndata as ad
        fresh = ad.AnnData(X=adata.X, obs=adata.obs.copy(), var=adata.var.copy())
        result = utils.store_latents(fresh, fake_latents, group_indices_list)
        assert "X_spVIPESmulti_shared" in result.obsm
        assert result.obsm["X_spVIPESmulti_shared"].shape == (N_CELLS, N_DIMS_SHARED)

    def test_private_per_group_written(self, adata, fake_latents, group_indices_list):
        import anndata as ad
        fresh = ad.AnnData(X=adata.X, obs=adata.obs.copy(), var=adata.var.copy())
        utils.store_latents(fresh, fake_latents, group_indices_list)
        for gi in range(3):
            key = f"X_spVIPESmulti_private_g{gi}"
            assert key in fresh.obsm, f"Missing key {key}"
            assert fresh.obsm[key].shape == (N_CELLS, N_DIMS_PRIVATE)

    def test_custom_prefix(self, adata, fake_latents, group_indices_list):
        import anndata as ad
        fresh = ad.AnnData(X=adata.X, obs=adata.obs.copy(), var=adata.var.copy())
        utils.store_latents(fresh, fake_latents, group_indices_list, obsm_prefix="X_custom")
        assert "X_custom_shared" in fresh.obsm

    def test_multimodal_keys_written(self, rng, group_indices_list):
        import anndata as ad
        n = N_CELLS
        fresh = ad.AnnData(X=np.zeros((n, 5)))
        latents = {
            "shared_reordered": {i: rng.random((40, N_DIMS_SHARED)).astype(np.float32) for i in range(3)},
            "private_multimodal_reordered": {
                (0, "rna"): rng.random((40, N_DIMS_PRIVATE)).astype(np.float32),
                (0, "protein"): rng.random((40, 3)).astype(np.float32),
                (1, "rna"): rng.random((40, N_DIMS_PRIVATE)).astype(np.float32),
                (1, "protein"): rng.random((40, 3)).astype(np.float32),
            },
        }
        utils.store_latents(fresh, latents, group_indices_list)
        assert "X_spVIPESmulti_private_0_rna" in fresh.obsm
        assert "X_spVIPESmulti_private_0_protein" in fresh.obsm

    def test_cell_order_preserved(self, rng, group_indices_list):
        """Cells at specific indices should receive the correct latent vector."""
        import anndata as ad
        n = N_CELLS
        fresh = ad.AnnData(X=np.zeros((n, 5)))
        # Build latents where each group's values are distinguishable
        sentinel = {i: np.full((40, N_DIMS_SHARED), float(i), dtype=np.float32) for i in range(3)}
        latents = {"shared_reordered": sentinel}
        utils.store_latents(fresh, latents, group_indices_list)
        arr = fresh.obsm["X_spVIPESmulti_shared"]
        for gi, idxs in enumerate(group_indices_list):
            assert np.allclose(arr[idxs], float(gi))


# ===========================================================================
# utils.add_latent_dims_to_obs
# ===========================================================================


class TestAddLatentDimsToObs:
    def test_columns_created(self, adata):
        utils.add_latent_dims_to_obs(adata, "X_spVIPESmulti_shared")
        for i in range(N_DIMS_SHARED):
            assert f"spVIPESmulti_shared_{i}" in adata.obs.columns

    def test_max_dims(self, adata):
        utils.add_latent_dims_to_obs(adata, "X_spVIPESmulti_shared", max_dims=2)
        assert "spVIPESmulti_shared_0" in adata.obs.columns
        assert "spVIPESmulti_shared_1" in adata.obs.columns
        assert "spVIPESmulti_shared_2" not in adata.obs.columns

    def test_custom_prefix(self, adata):
        utils.add_latent_dims_to_obs(adata, "X_spVIPESmulti_shared", prefix="myZ")
        assert "myZ_0" in adata.obs.columns

    def test_missing_key_raises(self, adata):
        with pytest.raises(KeyError, match="not found in adata.obsm"):
            utils.add_latent_dims_to_obs(adata, "X_nonexistent")

    def test_values_correct(self, adata):
        utils.add_latent_dims_to_obs(adata, "X_spVIPESmulti_shared")
        np.testing.assert_allclose(
            adata.obs["spVIPESmulti_shared_0"].values,
            adata.obsm["X_spVIPESmulti_shared"][:, 0],
        )


# ===========================================================================
# utils.get_top_genes
# ===========================================================================


class TestGetTopGenes:
    def test_signed_true_shape(self, loadings_df_shared):
        result = utils.get_top_genes(loadings_df_shared, n_top=3)
        assert list(result.columns) == ["dim", "pos_genes", "neg_genes"]
        assert len(result) == N_DIMS_SHARED

    def test_signed_false_shape(self, loadings_df_shared):
        result = utils.get_top_genes(loadings_df_shared, n_top=3, signed=False)
        assert list(result.columns) == ["dim", "top_genes"]

    def test_n_top_genes_returned(self, loadings_df_shared):
        n_top = 4
        result = utils.get_top_genes(loadings_df_shared, n_top=n_top)
        for _, row in result.iterrows():
            assert len(row["pos_genes"]) == n_top
            assert len(row["neg_genes"]) == n_top

    def test_uses_model_when_df_none(self, mock_model):
        result = utils.get_top_genes(model=mock_model, n_top=2)
        assert len(result) == N_DIMS_SHARED

    def test_dim_column_matches_loadings_columns(self, loadings_df_shared):
        result = utils.get_top_genes(loadings_df_shared, n_top=2)
        assert result["dim"].tolist() == loadings_df_shared.columns.tolist()


# ===========================================================================
# utils.score_cells_on_factor
# ===========================================================================


class TestScoreCellsOnFactor:
    def test_default_col_name(self, adata):
        utils.score_cells_on_factor(adata, dim_idx=0, obsm_key="X_spVIPESmulti_shared")
        assert "spVIPESmulti_shared_0" in adata.obs.columns

    def test_custom_col_name(self, adata):
        utils.score_cells_on_factor(adata, dim_idx=1, obsm_key="X_spVIPESmulti_shared", col_name="myFactor")
        assert "myFactor" in adata.obs.columns

    def test_values_correct(self, adata):
        utils.score_cells_on_factor(adata, dim_idx=2, obsm_key="X_spVIPESmulti_shared")
        np.testing.assert_allclose(
            adata.obs["spVIPESmulti_shared_2"].values,
            adata.obsm["X_spVIPESmulti_shared"][:, 2],
        )

    def test_missing_key_raises(self, adata):
        with pytest.raises(KeyError):
            utils.score_cells_on_factor(adata, dim_idx=0, obsm_key="X_missing")

    def test_out_of_range_dim_raises(self, adata):
        with pytest.raises(ValueError, match="out of range"):
            utils.score_cells_on_factor(adata, dim_idx=999, obsm_key="X_spVIPESmulti_shared")


# ===========================================================================
# metrics (pure numpy, no scvi dependency)
# ===========================================================================


@pytest.fixture
def z_shared_mixed(rng):
    """Two groups that are well-mixed in latent space."""
    return rng.standard_normal((200, 8)).astype(np.float32)


@pytest.fixture
def groups_balanced():
    return np.array(["g0"] * 100 + ["g1"] * 100)


@pytest.fixture
def labels_clustered(rng):
    """Cells with clear label structure."""
    return np.repeat(["A", "B", "C", "D"], 50)


class TestIlisi:
    def test_returns_float(self, z_shared_mixed, groups_balanced):
        score = metrics.ilisi(z_shared_mixed, groups_balanced, k=10)
        assert isinstance(score, float)

    def test_range(self, z_shared_mixed, groups_balanced):
        score = metrics.ilisi(z_shared_mixed, groups_balanced, k=10)
        assert 1.0 <= score <= len(np.unique(groups_balanced)) + 0.1

    def test_perfect_mixing_near_n_groups(self, rng):
        # Random normal ~ perfectly mixed in high-dim space
        z = rng.standard_normal((400, 10))
        groups = np.tile(["g0", "g1", "g2", "g3"], 100)
        score = metrics.ilisi(z, groups, k=20)
        assert score > 2.0  # should be close to 4 (n_groups)

    def test_perfect_segregation_near_1(self):
        # Each group occupies a completely different region
        z0 = np.zeros((50, 2)) + np.array([0, 0])
        z1 = np.zeros((50, 2)) + np.array([100, 100])
        z = np.vstack([z0, z1])
        groups = np.array(["g0"] * 50 + ["g1"] * 50)
        score = metrics.ilisi(z, groups, k=10)
        assert score < 1.5


class TestClisi:
    def test_is_alias_of_ilisi(self, z_shared_mixed, labels_clustered):
        s1 = metrics.clisi(z_shared_mixed, labels_clustered, k=10)
        s2 = metrics.ilisi(z_shared_mixed, labels_clustered, k=10)
        assert s1 == s2


class TestKbet:
    def test_returns_float(self, z_shared_mixed, groups_balanced):
        score = metrics.kbet(z_shared_mixed, groups_balanced, k=10)
        assert isinstance(score, float)

    def test_range(self, z_shared_mixed, groups_balanced):
        score = metrics.kbet(z_shared_mixed, groups_balanced, k=10)
        assert 0.0 <= score <= 1.0

    def test_perfect_mixing_close_to_1(self, rng):
        z = rng.standard_normal((400, 10))
        groups = np.tile(["g0", "g1"], 200)
        score = metrics.kbet(z, groups, k=20)
        assert score > 0.5

    def test_segregated_lower_than_mixed(self, rng):
        z_mixed = rng.standard_normal((200, 5))
        z_seg = np.vstack([
            rng.standard_normal((100, 5)) + np.array([10, 0, 0, 0, 0]),
            rng.standard_normal((100, 5)) + np.array([-10, 0, 0, 0, 0]),
        ])
        g = np.array(["g0"] * 100 + ["g1"] * 100)
        assert metrics.kbet(z_mixed, g, k=10) > metrics.kbet(z_seg, g, k=10)


class TestKnnPurity:
    def test_returns_float(self, z_shared_mixed, labels_clustered):
        score = metrics.knn_purity(z_shared_mixed, labels_clustered, k=10)
        assert isinstance(score, float)

    def test_range(self, z_shared_mixed, labels_clustered):
        score = metrics.knn_purity(z_shared_mixed, labels_clustered, k=10)
        assert 0.0 <= score <= 1.0

    def test_perfect_clustering_gives_1(self):
        # Each label in its own corner
        z = np.vstack([
            np.zeros((50, 2)) + [0, 0],
            np.zeros((50, 2)) + [100, 0],
            np.zeros((50, 2)) + [0, 100],
            np.zeros((50, 2)) + [100, 100],
        ])
        labels = np.repeat(["A", "B", "C", "D"], 50)
        score = metrics.knn_purity(z, labels, k=5)
        assert score == pytest.approx(1.0)


class TestPerGroupSilhouette:
    def test_returns_float(self, z_shared_mixed, groups_balanced):
        score = metrics.per_group_silhouette(z_shared_mixed, groups_balanced)
        assert isinstance(score, float)

    def test_range(self, z_shared_mixed, groups_balanced):
        score = metrics.per_group_silhouette(z_shared_mixed, groups_balanced)
        assert -1.0 <= score <= 1.0

    def test_single_group_returns_nan(self):
        z = np.ones((50, 4))
        g = np.array(["g0"] * 50)
        score = metrics.per_group_silhouette(z, g)
        assert np.isnan(score)

    def test_well_separated_groups_positive(self):
        z = np.vstack([
            np.zeros((100, 2)) + [0, 0],
            np.zeros((100, 2)) + [100, 100],
        ])
        g = np.array(["g0"] * 100 + ["g1"] * 100)
        score = metrics.per_group_silhouette(z, g)
        assert score > 0.9


class TestIntegrationReport:
    def test_returns_dataframe(self, rng):
        z = rng.standard_normal((200, 8)).astype(np.float32)
        groups = np.array(["g0"] * 100 + ["g1"] * 100)
        labels = np.tile(["A", "B", "C", "D"], 50)
        df = metrics.integration_report(z, groups, labels)
        assert isinstance(df, pd.DataFrame)
        assert "z_shared" in df["latent"].values

    def test_expected_columns(self, rng):
        z = rng.standard_normal((200, 8)).astype(np.float32)
        groups = np.array(["g0"] * 100 + ["g1"] * 100)
        labels = np.tile(["A", "B"], 100)
        df = metrics.integration_report(z, groups, labels)
        expected_cols = {"latent", "ilisi", "clisi", "kbet", "knn_purity", "leiden_ari", "silhouette"}
        assert expected_cols.issubset(set(df.columns))

    def test_shared_silhouette_is_nan(self, rng):
        z = rng.standard_normal((200, 8)).astype(np.float32)
        groups = np.array(["g0"] * 100 + ["g1"] * 100)
        labels = np.tile(["A", "B"], 100)
        df = metrics.integration_report(z, groups, labels)
        shared_row = df[df["latent"] == "z_shared"].iloc[0]
        assert np.isnan(shared_row["silhouette"])

    def test_with_private_dict(self, rng):
        z = rng.standard_normal((200, 8)).astype(np.float32)
        groups = np.array(["g0"] * 100 + ["g1"] * 100)
        labels = np.tile(["A", "B"], 100)
        z_priv = {
            "g0": rng.standard_normal((100, 4)).astype(np.float32),
            "g1": rng.standard_normal((100, 4)).astype(np.float32),
        }
        df = metrics.integration_report(z, groups, labels, z_private_dict=z_priv)
        assert len(df) == 3  # 1 shared + 2 private groups
        private_rows = df[df["latent"].str.startswith("z_private")]
        assert len(private_rows) == 2
        # silhouette should be non-nan for private rows
        assert not private_rows["silhouette"].isna().all()

    def test_metric_ranges(self, rng):
        z = rng.standard_normal((200, 8)).astype(np.float32)
        groups = np.array(["g0"] * 100 + ["g1"] * 100)
        labels = np.tile(["A", "B"], 100)
        df = metrics.integration_report(z, groups, labels)
        row = df[df["latent"] == "z_shared"].iloc[0]
        assert row["ilisi"] >= 1.0
        assert 0.0 <= row["kbet"] <= 1.0
        assert 0.0 <= row["knn_purity"] <= 1.0
        # leiden_ari is nan when igraph is not installed — that is acceptable
        assert np.isnan(row["leiden_ari"]) or -1.0 <= row["leiden_ari"] <= 1.0
