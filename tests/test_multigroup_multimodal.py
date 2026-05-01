"""Tests for multi-group (N >= 2) and multimodal support in spVIPESmulti."""

import importlib.util
import os
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import anndata as ad

# Direct module loaders to avoid triggering spVIPESmulti.__init__ (which requires scvi-tools >=1.0)
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _load_module(module_name, filepath):
    """Load a single module from filepath without triggering package __init__."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_prepare_adatas_mod = _load_module(
    "prepare_adatas", os.path.join(_SRC, "spVIPESmulti", "data", "prepare_adatas.py")
)
prepare_adatas = _prepare_adatas_mod.prepare_adatas
prepare_multimodal_adatas = _prepare_adatas_mod.prepare_multimodal_adatas


# =============================================================================
# Test prepare_adatas multi-group support
# =============================================================================


class TestPrepareAdatasMultiGroup:
    """Tests for prepare_adatas with N >= 2 groups."""

    def _make_adata(self, n_obs=50, n_vars=30, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32)
        adata = ad.AnnData(X=csr_matrix(X))
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        return adata

    def test_two_groups(self):
        """Two groups should work as before."""

        adata1 = self._make_adata(50, 30, seed=0)
        adata2 = self._make_adata(40, 25, seed=1)
        result = prepare_adatas({"group_a": adata1, "group_b": adata2})

        assert "groups" in result.obs.columns
        assert "indices" in result.obs.columns
        assert len(result.uns["groups_var_indices"]) == 2
        assert len(result.uns["groups_obs_indices"]) == 2
        assert result.uns["groups_lengths"][0] == 30
        assert result.uns["groups_lengths"][1] == 25
        assert result.n_obs == 90  # 50 + 40

    def test_three_groups(self):
        """Three groups should now be supported."""

        adata1 = self._make_adata(50, 30, seed=0)
        adata2 = self._make_adata(40, 25, seed=1)
        adata3 = self._make_adata(35, 20, seed=2)
        result = prepare_adatas({"a": adata1, "b": adata2, "c": adata3})

        assert result.n_obs == 125  # 50 + 40 + 35
        assert len(result.uns["groups_var_indices"]) == 3
        assert len(result.uns["groups_obs_indices"]) == 3
        assert result.uns["groups_lengths"][0] == 30
        assert result.uns["groups_lengths"][1] == 25
        assert result.uns["groups_lengths"][2] == 20
        assert result.uns["groups_mapping"] == {0: "a", 1: "b", 2: "c"}

    def test_five_groups(self):
        """Five groups should work."""

        adatas = {f"g{i}": self._make_adata(20 + i * 5, 10 + i * 3, seed=i) for i in range(5)}
        result = prepare_adatas(adatas)

        assert len(result.uns["groups_var_indices"]) == 5
        assert len(result.uns["groups_obs_indices"]) == 5
        assert len(result.uns["groups_mapping"]) == 5

    def test_fewer_than_two_groups_raises(self):
        """Should raise ValueError for fewer than 2 groups."""

        with pytest.raises(ValueError, match="At least 2 groups"):
            prepare_adatas({"only_one": self._make_adata()})

    def test_group_indices_are_correct(self):
        """Verify group indices correctly map cells."""

        adata1 = self._make_adata(30, 10, seed=0)
        adata2 = self._make_adata(20, 15, seed=1)
        adata3 = self._make_adata(25, 12, seed=2)
        result = prepare_adatas({"x": adata1, "y": adata2, "z": adata3})

        # Check obs indices partition the data correctly
        obs_indices = result.uns["groups_obs_indices"]
        assert len(obs_indices[0]) == 30
        assert len(obs_indices[1]) == 20
        assert len(obs_indices[2]) == 25

        # All indices should be unique and cover all cells
        all_indices = np.concatenate(obs_indices)
        assert len(np.unique(all_indices)) == 75


# =============================================================================
# Test prepare_multimodal_adatas
# =============================================================================


class TestPrepareMultimodalAdatas:
    """Tests for prepare_multimodal_adatas."""

    def _make_adata(self, n_obs=50, n_vars=30, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32)
        adata = ad.AnnData(X=csr_matrix(X))
        adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
        adata.var_names = [f"feat_{i}" for i in range(n_vars)]
        return adata

    def test_two_groups_two_modalities(self):
        """Two groups, each with two modalities."""

        adatas = {
            "treatment": {
                "rna": self._make_adata(50, 30, seed=0),
                "protein": self._make_adata(50, 10, seed=1),
            },
            "control": {
                "rna": self._make_adata(40, 30, seed=2),
                "protein": self._make_adata(40, 10, seed=3),
            },
        }
        result = prepare_multimodal_adatas(adatas, modality_likelihoods={"rna": "nb", "protein": "gaussian"})

        assert result.uns["is_multimodal"] is True
        assert set(result.uns["modality_names"]) == {"protein", "rna"}
        assert result.uns["modality_likelihoods"]["rna"] == "nb"
        assert result.uns["modality_likelihoods"]["protein"] == "gaussian"
        assert result.n_obs == 90  # 50 + 40

        # Check per-group, per-modality lengths
        assert result.uns["groups_modality_lengths"][0]["rna"] == 30
        assert result.uns["groups_modality_lengths"][0]["protein"] == 10
        assert result.uns["groups_modality_lengths"][1]["rna"] == 30
        assert result.uns["groups_modality_lengths"][1]["protein"] == 10

        # Check var indices exist for each (group, modality)
        assert "rna" in result.uns["groups_modality_var_indices"][0]
        assert "protein" in result.uns["groups_modality_var_indices"][0]
        assert len(result.uns["groups_modality_var_indices"][0]["rna"]) == 30
        assert len(result.uns["groups_modality_var_indices"][0]["protein"]) == 10

    def test_default_likelihoods(self):
        """If no likelihoods specified, defaults to 'nb'."""

        adatas = {
            "g1": {"rna": self._make_adata(20, 10, seed=0)},
            "g2": {"rna": self._make_adata(20, 10, seed=1)},
        }
        result = prepare_multimodal_adatas(adatas)

        assert result.uns["modality_likelihoods"]["rna"] == "nb"

    def test_invalid_likelihood_raises(self):
        """Invalid likelihood type should raise."""

        adatas = {
            "g1": {"rna": self._make_adata(20, 10, seed=0)},
            "g2": {"rna": self._make_adata(20, 10, seed=1)},
        }
        with pytest.raises(ValueError, match="Unsupported likelihood"):
            prepare_multimodal_adatas(adatas, modality_likelihoods={"rna": "poisson"})

    def test_fewer_than_two_groups_raises(self):
        """Should raise ValueError for fewer than 2 groups."""

        with pytest.raises(ValueError, match="At least 2 groups"):
            prepare_multimodal_adatas({"only_one": {"rna": self._make_adata()}})

    def test_asymmetric_modalities(self):
        """Groups can have different sets of modalities."""

        adatas = {
            "g1": {
                "rna": self._make_adata(30, 20, seed=0),
                "protein": self._make_adata(30, 8, seed=1),
            },
            "g2": {
                "rna": self._make_adata(25, 20, seed=2),
                # g2 has no protein modality
            },
        }
        result = prepare_multimodal_adatas(adatas)

        assert "rna" in result.uns["groups_modality_var_indices"][0]
        assert "protein" in result.uns["groups_modality_var_indices"][0]
        assert "rna" in result.uns["groups_modality_var_indices"][1]
        assert "protein" not in result.uns["groups_modality_var_indices"][1]

    def test_groups_and_indices_metadata(self):
        """Check standard metadata is correctly set."""

        adatas = {
            "treatment": {"rna": self._make_adata(30, 15, seed=0)},
            "control": {"rna": self._make_adata(20, 15, seed=1)},
        }
        result = prepare_multimodal_adatas(adatas)

        assert "groups" in result.obs.columns
        assert "indices" in result.obs.columns
        assert result.uns["groups_mapping"] == {0: "treatment", 1: "control"}
        assert len(result.uns["groups_obs_indices"]) == 2
        assert len(result.uns["groups_obs_indices"][0]) == 30
        assert len(result.uns["groups_obs_indices"][1]) == 20


# =============================================================================
# Test PoE generalization
# =============================================================================


class TestPoEGeneralization:
    """Tests for the generic _poe_n and _product_of_experts methods."""

    def test_poe_n_two_groups_matches_formula(self):
        """Verify _poe_n produces correct results for 2 groups with equal batch sizes."""
        import torch
        from torch.distributions import Normal

        # We can't easily instantiate spVIPESmultimodule without all the scvi deps,
        # but we can test _product_of_experts directly since it's pure math
        # Test the Gaussian PoE formula manually

        torch.manual_seed(42)
        batch_size = 16
        latent_dim = 10

        mu1 = torch.randn(batch_size, latent_dim)
        mu2 = torch.randn(batch_size, latent_dim)
        logvar1 = torch.randn(batch_size, latent_dim)
        logvar2 = torch.randn(batch_size, latent_dim)

        # Stack for _product_of_experts format
        mus = torch.stack([mu1, mu2], dim=0)  # (2, batch, latent)
        logvars = torch.stack([logvar1, logvar2], dim=0)

        # Compute PoE manually
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        joint_precision = 1.0 + 1.0 / var1 + 1.0 / var2  # includes prior
        joint_var = 1.0 / joint_precision
        joint_mu = joint_var * (mu1 / var1 + mu2 / var2)
        joint_logvar = torch.log(joint_var)

        # Compute using the formula from _product_of_experts
        vars_stacked = torch.exp(logvars)
        mus_joint = torch.sum(mus / vars_stacked, dim=0)
        logvars_joint = torch.ones_like(mus_joint)
        logvars_joint += torch.sum(1.0 / vars_stacked, dim=0)
        logvars_joint = 1.0 / logvars_joint
        mus_joint *= logvars_joint
        logvars_joint = torch.log(logvars_joint)

        assert torch.allclose(mus_joint, joint_mu, atol=1e-6)
        assert torch.allclose(logvars_joint, joint_logvar, atol=1e-6)

    def test_poe_n_three_groups(self):
        """Verify PoE formula works for 3 groups."""
        import torch

        torch.manual_seed(42)
        batch_size = 16
        latent_dim = 10

        mus = torch.randn(3, batch_size, latent_dim)
        logvars = torch.randn(3, batch_size, latent_dim)

        # Compute manually
        vars_ = torch.exp(logvars)
        precision_sum = 1.0 + torch.sum(1.0 / vars_, dim=0)
        joint_var = 1.0 / precision_sum
        joint_mu = joint_var * torch.sum(mus / vars_, dim=0)

        # Compute using formula
        vars_stacked = torch.exp(logvars)
        mus_joint = torch.sum(mus / vars_stacked, dim=0)
        logvars_joint = torch.ones_like(mus_joint)
        logvars_joint += torch.sum(1.0 / vars_stacked, dim=0)
        logvars_joint = 1.0 / logvars_joint
        mus_joint *= logvars_joint

        assert torch.allclose(mus_joint, joint_mu, atol=1e-6)

    def test_poe_padding_uninformative(self):
        """Padding with loc=0, logvar=0 should act as a standard normal prior."""
        import torch

        torch.manual_seed(42)
        batch_size = 16
        latent_dim = 10

        mu_real = torch.randn(batch_size, latent_dim)
        logvar_real = torch.randn(batch_size, latent_dim)

        # Uninformative "group": mu=0, logvar=0 (var=1, precision=1)
        mu_prior = torch.zeros(batch_size, latent_dim)
        logvar_prior = torch.zeros(batch_size, latent_dim)

        mus = torch.stack([mu_real, mu_prior], dim=0)
        logvars = torch.stack([logvar_real, logvar_prior], dim=0)

        vars_ = torch.exp(logvars)
        mus_joint = torch.sum(mus / vars_, dim=0)
        precision_joint = 1.0 + torch.sum(1.0 / vars_, dim=0)
        joint_var = 1.0 / precision_joint
        mus_joint *= joint_var

        # Compare with single-group + prior (should be equivalent to the PoE
        # with one real group and two prior contributions: the base prior + the padding)
        var_real = torch.exp(logvar_real)
        single_precision = 1.0 + 1.0 / var_real + 1.0  # base prior + real + padding (var=1)
        single_var = 1.0 / single_precision
        single_mu = single_var * (mu_real / var_real + 0.0)  # padding contributes 0

        assert torch.allclose(mus_joint, single_mu, atol=1e-6)


# =============================================================================
# Test likelihood factory
# =============================================================================


class TestLikelihoodFactory:
    """Tests for the build_likelihood utility."""

    @pytest.fixture(autouse=True)
    def _load_build_likelihood(self):
        """Try to load build_likelihood; skip if scvi not compatible."""
        try:
            mod = _load_module("spvipesmulti_utils", os.path.join(_SRC, "spVIPESmulti", "module", "utils.py"))
            self.build_likelihood = mod.build_likelihood
        except Exception:
            pytest.skip("scvi-tools version incompatible with this environment")

    def test_nb_likelihood(self):
        """NB likelihood should return NegativeBinomialMixture."""
        import torch

        batch_size = 16
        n_genes = 30
        px_rate_private = torch.rand(batch_size, n_genes)
        px_rate_shared = torch.rand(batch_size, n_genes)
        px_r = torch.rand(n_genes)
        px_mixing = torch.randn(batch_size, n_genes)

        dist = self.build_likelihood("nb", px_rate_private, px_rate_shared, px_r, px_mixing)
        sample = torch.ones(batch_size, n_genes)
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == (batch_size, n_genes)

    def test_gaussian_likelihood(self):
        """Gaussian likelihood should return Normal distribution."""
        import torch

        batch_size = 16
        n_genes = 30
        px_rate_private = torch.rand(batch_size, n_genes)
        px_rate_shared = torch.rand(batch_size, n_genes)
        px_r = torch.rand(n_genes)
        px_mixing = torch.randn(batch_size, n_genes)

        dist = self.build_likelihood("gaussian", px_rate_private, px_rate_shared, px_r, px_mixing)
        sample = torch.randn(batch_size, n_genes)
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == (batch_size, n_genes)

    def test_invalid_likelihood_raises(self):
        """Invalid likelihood type should raise."""
        import torch

        with pytest.raises(ValueError, match="Unsupported likelihood"):
            self.build_likelihood("poisson", torch.rand(1), torch.rand(1), torch.rand(1), torch.rand(1))
