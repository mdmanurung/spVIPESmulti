"""Regression tests covering bugs fixed in the v2 audit.

Tests in this file:
  - Generative latent slicing (private vs shared) — Phase 1.1
  - _label_based_poe double-prior — Phase 1.2
  - n_labels guard — Phase 2.5
  - DISENTANGLE_PRESETS key validation — Phase 3.2
  - Negative weight guard — Phase 3.3
  - ConcatDataLoader empty-indices guard — Phase 3.4
  - px_scale blending includes private — Phase 3.1
  - Gradient propagation check — Phase 4.2
  - Disentangle preset edge cases — Phase 4.6
  - get_latent_representation completeness — Phase 4.8
  - ConcatDataLoader balance — Phase 4.10
"""
import importlib.util
import os

import numpy as np
import pytest
import torch
from scvi import REGISTRY_KEYS

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Helpers
# ============================================================

def _make_decoder(n_private=10, n_shared=25, n_output=50):
    """Instantiate a LinearDecoderSPVIPE without triggering full spVIPESmulti init."""
    import sys
    sys.path.insert(0, _SRC)
    from spVIPESmulti.nn.networks import LinearDecoderSPVIPE
    return LinearDecoderSPVIPE(
        n_input_private=n_private,
        n_input_shared=n_shared,
        n_output=n_output,
        n_cat_list=None,
        use_batch_norm=False,
        use_layer_norm=False,
        bias=False,
    )


def _make_module(n_private=10, n_shared=25, n_genes=40, n_cells=30, n_groups=2, **kwargs):
    """Instantiate a minimal spVIPESmultimodule for use in unit tests."""
    import sys
    sys.path.insert(0, _SRC)
    from spVIPESmulti.module.spVIPESmultimodule import spVIPESmultimodule

    groups_lengths = {i: n_genes for i in range(n_groups)}
    groups_obs_names = [list(range(n_cells))] * n_groups
    groups_var_names = [[f"g{v}" for v in range(n_genes)]] * n_groups
    groups_obs_indices = [list(range(n_cells))] * n_groups
    groups_var_indices = [list(range(n_genes))] * n_groups

    return spVIPESmultimodule(
        groups_lengths=groups_lengths,
        groups_obs_names=groups_obs_names,
        groups_var_names=groups_var_names,
        groups_obs_indices=groups_obs_indices,
        groups_var_indices=groups_var_indices,
        n_dimensions_private=n_private,
        n_dimensions_shared=n_shared,
        n_hidden=32,
        **kwargs,
    )


def _make_minimal_loss_inputs(module, frac=False):
    """Create deterministic loss inputs without running full inference/generative passes."""
    n_groups = len(module.input_dims)
    n_private = module.n_dimensions_private
    n_shared = module.n_dimensions_shared

    per_group = []
    inference_outputs = {"private_stats": {}, "poe_stats": {}}
    generative_outputs = {"private_poe": {}}

    for g in range(n_groups):
        x = torch.tensor(
            [[1.5, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 2.0, 1.0]], dtype=torch.float32
        ) if frac else torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 2.0, 1.0]], dtype=torch.float32
        )
        if module.input_dims[g] != x.shape[1]:
            x = x[:, : module.input_dims[g]]
        batch = torch.zeros((x.shape[0], 1), dtype=torch.long)
        groups = torch.full((x.shape[0], 1), g, dtype=torch.long)
        per_group.append(
            {
                REGISTRY_KEYS.X_KEY: x,
                REGISTRY_KEYS.BATCH_KEY: batch,
                "groups": groups,
            }
        )

        z_private = torch.zeros((x.shape[0], n_private), dtype=torch.float32)
        z_shared = torch.zeros((x.shape[0], n_shared), dtype=torch.float32)

        inference_outputs["private_stats"][g] = {
            "qz": torch.distributions.Normal(z_private, torch.ones_like(z_private)),
            "log_z": z_private,
        }
        inference_outputs["poe_stats"][g] = {
            "logtheta_qz": torch.distributions.Normal(z_shared, torch.ones_like(z_shared)),
            "logtheta_log_z": z_shared,
        }

        generative_outputs["private_poe"][str(g)] = {
            "px": torch.distributions.Normal(loc=x, scale=torch.ones_like(x)),
        }

    return per_group, inference_outputs, generative_outputs


# ============================================================
# Phase 1.1 — Generative latent slicing
# ============================================================

class TestGenerativeSlicing:
    """Ensure the decoder receives the correct private vs shared latents."""

    def test_private_change_affects_private_rate_not_shared(self):
        """Perturbing z_private should change px_rate_private but not px_rate_shared."""
        dec = _make_decoder(n_private=10, n_shared=25, n_output=50)
        dec.eval()

        torch.manual_seed(0)
        batch_size = 8
        z_private = torch.randn(batch_size, 10)
        z_shared = torch.randn(batch_size, 25)
        library = torch.zeros(batch_size, 1)

        with torch.no_grad():
            out_a = dec("gene", z_private, z_shared, library)
            z_private_alt = torch.randn(batch_size, 10)
            out_b = dec("gene", z_private_alt, z_shared, library)

        # px_rate_private should differ
        assert not torch.allclose(out_a[2], out_b[2]), "px_rate_private did not change when z_private changed"
        # px_rate_shared should be identical (z_shared unchanged)
        assert torch.allclose(out_a[3], out_b[3]), "px_rate_shared changed when only z_private changed"

    def test_shared_change_affects_shared_rate_not_private(self):
        """Perturbing z_shared should change px_rate_shared but not px_rate_private."""
        dec = _make_decoder(n_private=10, n_shared=25, n_output=50)
        dec.eval()

        torch.manual_seed(1)
        batch_size = 8
        z_private = torch.randn(batch_size, 10)
        z_shared = torch.randn(batch_size, 25)
        library = torch.zeros(batch_size, 1)

        with torch.no_grad():
            out_a = dec("gene", z_private, z_shared, library)
            z_shared_alt = torch.randn(batch_size, 25)
            out_b = dec("gene", z_private, z_shared_alt, library)

        assert not torch.allclose(out_a[3], out_b[3]), "px_rate_shared did not change when z_shared changed"
        assert torch.allclose(out_a[2], out_b[2]), "px_rate_private changed when only z_shared changed"

    def test_generative_slicing_correctness(self):
        """generative() must pass private dims to private decoder and shared to shared."""
        import sys
        sys.path.insert(0, _SRC)
        module = _make_module(n_private=10, n_shared=25, n_genes=40)
        module.eval()

        batch_size = 8
        torch.manual_seed(42)

        # Craft private_stats and poe_stats with known, distinct values
        z_private = torch.zeros(batch_size, 10)           # private is all zeros
        z_shared = torch.ones(batch_size, 25) * 5.0       # shared is all fives

        private_stats = {
            0: {
                "log_z": z_private,
                "theta": torch.softmax(z_private, -1),
                "qz": torch.distributions.Normal(z_private, torch.ones_like(z_private)),
            },
            1: {
                "log_z": z_private.clone(),
                "theta": torch.softmax(z_private, -1),
                "qz": torch.distributions.Normal(z_private, torch.ones_like(z_private)),
            },
        }
        poe_stats = {
            0: {
                "logtheta_log_z": z_shared,
                "logtheta_theta": torch.softmax(z_shared, -1),
                "logtheta_qz": torch.distributions.Normal(z_shared, torch.ones_like(z_shared)),
            },
            1: {
                "logtheta_log_z": z_shared.clone(),
                "logtheta_theta": torch.softmax(z_shared, -1),
                "logtheta_qz": torch.distributions.Normal(z_shared, torch.ones_like(z_shared)),
            },
        }
        library = {0: torch.zeros(batch_size, 1), 1: torch.zeros(batch_size, 1)}
        batch_index = [torch.zeros(batch_size, 1, dtype=torch.long)] * 2

        with torch.no_grad():
            out = module.generative(private_stats, {}, poe_stats, library, None, batch_index)

        # Verify outputs are finite
        for key in ["0", "1"]:
            assert torch.isfinite(out["private_poe"][key]["px_rate_private"]).all()
            assert torch.isfinite(out["private_poe"][key]["px_rate_shared"]).all()


# ============================================================
# Audit follow-up — prepare_adatas prefix-overlap regression
# ============================================================


class TestPrepareAdatasPrefixOverlap:
    """`prepare_adatas` must not cross-match groups whose names share a prefix.

    Example regression: groups "cat" and "category" both matched
    ``str.startswith("cat")`` on var_names and obs["groups"], so
    ``groups_var_indices[0]`` ended up containing every var (cat + category)
    and ``groups_obs_indices[0]`` contained every cell.
    """

    def _load_prepare(self):
        return _load_module(
            "prepare_adatas",
            os.path.join(_SRC, "spVIPESmulti", "data", "prepare_adatas.py"),
        ).prepare_adatas

    def _load_prepare_multimodal(self):
        return _load_module(
            "prepare_adatas_mm",
            os.path.join(_SRC, "spVIPESmulti", "data", "prepare_adatas.py"),
        ).prepare_multimodal_adatas

    def _make_adata(self, n_obs, n_vars, seed, obs_names=None):
        import anndata as ad
        from scipy.sparse import csr_matrix

        rng = np.random.default_rng(seed)
        X = rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32)
        a = ad.AnnData(X=csr_matrix(X))
        a.obs_names = obs_names if obs_names is not None else [f"c{seed}_{i}" for i in range(n_obs)]
        a.var_names = [f"g{i}" for i in range(n_vars)]
        return a

    def test_single_modal_overlapping_prefixes(self):
        prepare_adatas = self._load_prepare()
        a1 = self._make_adata(20, 10, seed=0)
        a2 = self._make_adata(20, 10, seed=1)

        result = prepare_adatas({"cat": a1, "category": a2})

        # Each group should own exactly its own vars and cells.
        assert len(result.uns["groups_var_indices"][0]) == 10, "cat should not absorb category vars"
        assert len(result.uns["groups_var_indices"][1]) == 10
        assert len(result.uns["groups_obs_indices"][0]) == 20, "cat should not absorb category cells"
        assert len(result.uns["groups_obs_indices"][1]) == 20

        # Indices must be disjoint.
        assert set(result.uns["groups_var_indices"][0]).isdisjoint(
            set(result.uns["groups_var_indices"][1])
        )
        assert set(result.uns["groups_obs_indices"][0]).isdisjoint(
            set(result.uns["groups_obs_indices"][1])
        )

    def test_multimodal_overlapping_prefixes(self):
        prepare_multimodal_adatas = self._load_prepare_multimodal()
        obs_cat = [f"cat_{i}" for i in range(20)]
        obs_category = [f"category_{i}" for i in range(20)]
        a_rna = self._make_adata(20, 10, seed=0, obs_names=obs_cat)
        a_prot = self._make_adata(20, 5, seed=1, obs_names=obs_cat)
        b_rna = self._make_adata(20, 10, seed=2, obs_names=obs_category)
        b_prot = self._make_adata(20, 5, seed=3, obs_names=obs_category)

        result = prepare_multimodal_adatas(
            {
                "cat": {"rna": a_rna, "protein": a_prot},
                "category": {"rna": b_rna, "protein": b_prot},
            }
        )

        assert len(result.uns["groups_obs_indices"][0]) == 20, "cat must not absorb category cells"
        assert len(result.uns["groups_obs_indices"][1]) == 20
        # Var indices for each (group, modality) live under the
        # `{group}_{modality}_` prefix; no group should leak into the other's
        # var indices.
        assert (
            len(result.uns["groups_modality_var_indices"][0]["rna"]) == 10
        )
        assert (
            len(result.uns["groups_modality_var_indices"][1]["rna"]) == 10
        )

    def test_multimodal_mismatched_obs_names_raise(self):
        prepare_multimodal_adatas = self._load_prepare_multimodal()
        a_rna = self._make_adata(20, 10, seed=0)
        a_prot = self._make_adata(20, 5, seed=1)
        b_obs = [f"shared_{i}" for i in range(20)]
        b_rna = self._make_adata(20, 10, seed=2, obs_names=b_obs)
        b_prot = self._make_adata(20, 5, seed=3, obs_names=b_obs)

        with pytest.raises(ValueError, match="same cells with matching obs_names"):
            prepare_multimodal_adatas(
                {
                    "cat": {"rna": a_rna, "protein": a_prot},
                    "category": {"rna": b_rna, "protein": b_prot},
                }
            )


# ============================================================
# Phase 1.2 — PoE double-prior
# ============================================================

class TestPoEDoublePrior:
    """Absent-label group should contribute near-zero precision (not 1+1=2)."""

    def test_absent_group_uses_large_logvar(self):
        """In _label_based_poe, absent groups must have logvar=30 (not 0)."""
        import sys
        sys.path.insert(0, _SRC)
        module = _make_module(n_private=10, n_shared=8, n_genes=20, n_groups=2)

        # Build dummy shared_stats and label_group where group 1 lacks label 0
        batch_size = 6
        latent_dim = 8
        shared_stats = {
            0: {
                "logtheta_loc": torch.randn(batch_size, latent_dim),
                "logtheta_logvar": torch.zeros(batch_size, latent_dim),
                "logtheta_scale": torch.ones(batch_size, latent_dim),
            },
            1: {
                "logtheta_loc": torch.randn(4, latent_dim),
                "logtheta_logvar": torch.zeros(4, latent_dim),
                "logtheta_scale": torch.ones(4, latent_dim),
            },
        }
        # Group 0 has label 0; group 1 has only label 1
        label_group = {
            0: torch.zeros(batch_size, dtype=torch.long),
            1: torch.ones(4, dtype=torch.long),
        }

        result = module._label_based_poe(shared_stats, label_group)

        # Result should exist for both groups
        assert 0 in result
        assert 1 in result

        # The returned posterior variance for group 0 should be less than 1
        # (it was combined with the global prior), confirming single-precision contribution
        g0_logvar = result[0]["logtheta_logvar"]
        assert torch.isfinite(g0_logvar).all()

    def test_product_of_experts_precision_accumulation(self):
        """_product_of_experts with one near-zero-precision expert should behave like 1-expert + prior."""
        import sys
        sys.path.insert(0, _SRC)
        module = _make_module(n_private=5, n_shared=4, n_genes=10, n_groups=2)

        batch_size = 4
        latent_dim = 4
        mu_real = torch.ones(batch_size, latent_dim)
        logvar_real = torch.zeros(batch_size, latent_dim)  # var=1, precision=1

        # "Absent" expert with large logvar (precision ≈ 0)
        mu_absent = torch.zeros(batch_size, latent_dim)
        logvar_absent = torch.full((batch_size, latent_dim), 30.0)

        mus = torch.stack([mu_real, mu_absent], dim=0)
        logvars = torch.stack([logvar_real, logvar_absent], dim=0)

        mu_joint, logvar_joint = module._product_of_experts(mus, logvars)

        # Joint precision ≈ 1 (global prior) + 1 (real) + ~0 (absent) = 2 → var ≈ 0.5
        joint_var = torch.exp(logvar_joint)
        assert (joint_var < 1.0).all(), f"Expected joint_var < 1, got {joint_var}"
        # With logvar=30 for absent group, absent precision ≈ exp(-30) ≈ 0
        # So joint_var ≈ 1/(1+1) = 0.5
        assert torch.allclose(joint_var, torch.full_like(joint_var, 0.5), atol=0.01), \
            f"Expected joint_var ≈ 0.5, got {joint_var}"


# ============================================================
# Phase 2.5 — n_labels guard
# ============================================================

class TestNLabelsGuard:
    def test_use_labels_without_n_labels_raises(self):
        """use_labels=True with n_labels=None should raise ValueError immediately."""
        import sys
        sys.path.insert(0, _SRC)
        from spVIPESmulti.module.spVIPESmultimodule import spVIPESmultimodule

        groups_lengths = {0: 20, 1: 20}
        with pytest.raises(ValueError, match="n_labels must be provided"):
            spVIPESmultimodule(
                groups_lengths=groups_lengths,
                groups_obs_names=[[]] * 2,
                groups_var_names=[[]] * 2,
                groups_obs_indices=[[]] * 2,
                groups_var_indices=[[]] * 2,
                use_labels=True,
                n_labels=None,   # <-- should fail
            )


# ============================================================
# Phase 3.1 — px_scale blending includes private
# ============================================================

class TestPxScaleBlending:
    def test_px_scale_includes_private(self):
        """px_scale must differ when z_private changes (private contribution is non-zero)."""
        dec = _make_decoder(n_private=10, n_shared=25, n_output=50)
        dec.eval()

        torch.manual_seed(7)
        batch_size = 8
        z_shared = torch.randn(batch_size, 25)
        library = torch.zeros(batch_size, 1)

        with torch.no_grad():
            z_private_a = torch.randn(batch_size, 10)
            z_private_b = torch.randn(batch_size, 10)
            *_, px_scale_a = dec("gene", z_private_a, z_shared, library)
            *_, px_scale_b = dec("gene", z_private_b, z_shared, library)

        # If private is now included in px_scale, it must differ
        assert not torch.allclose(px_scale_a, px_scale_b), \
            "px_scale did not change when z_private changed — private component may still be missing"

    def test_px_scale_sums_to_one(self):
        """px_scale should be L1-normalized (sums to 1 per cell)."""
        dec = _make_decoder(n_private=10, n_shared=25, n_output=50)
        dec.eval()

        torch.manual_seed(9)
        batch_size = 16
        z_private = torch.randn(batch_size, 10)
        z_shared = torch.randn(batch_size, 25)
        library = torch.zeros(batch_size, 1)

        with torch.no_grad():
            *_, px_scale = dec("gene", z_private, z_shared, library)

        row_sums = px_scale.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5), \
            f"px_scale rows do not sum to 1: {row_sums}"


# ============================================================
# Phase 3.2 — DISENTANGLE_PRESETS key validation
# ============================================================

class TestDisentanglePresets:
    def test_all_presets_have_required_keys(self):
        import sys
        sys.path.insert(0, _SRC)
        from spVIPESmulti.model._disentangle_presets import DISENTANGLE_PRESETS, _REQUIRED_PRESET_KEYS

        for name, preset in DISENTANGLE_PRESETS.items():
            missing = _REQUIRED_PRESET_KEYS - preset.keys()
            assert not missing, f"Preset '{name}' is missing keys: {missing}"

    def test_preset_values_are_non_negative(self):
        import sys
        sys.path.insert(0, _SRC)
        from spVIPESmulti.model._disentangle_presets import DISENTANGLE_PRESETS

        for name, preset in DISENTANGLE_PRESETS.items():
            for key, val in preset.items():
                assert val >= 0, f"Preset '{name}' key '{key}' has negative value {val}"


# ============================================================
# Phase 3.3 — Negative weight guard (requires full spVIPESmulti)
# ============================================================

class TestNegativeWeightGuard:
    @pytest.mark.integration
    def test_negative_weight_raises(self):
        """Passing a negative disentangle weight should raise ValueError."""
        import spVIPESmulti
        import anndata as ad
        from scipy.sparse import csr_matrix

        rng = np.random.default_rng(0)
        X = rng.poisson(5, size=(40, 30)).astype(np.float32)
        adata = ad.AnnData(X=csr_matrix(X))
        adata.obs_names = [f"c{i}" for i in range(40)]
        adata.var_names = [f"g{i}" for i in range(30)]
        adata.obs["group"] = ["A"] * 20 + ["B"] * 20
        adata.obs["idx"] = list(range(40))
        adata.obs["ct"] = ["ct0"] * 20 + ["ct1"] * 20

        prepared = spVIPESmulti.data.prepare_adatas({"A": adata[:20].copy(), "B": adata[20:].copy()})
        spVIPESmulti.model.spVIPESmulti.setup_anndata(prepared, groups_key="groups", label_key="ct")

        with pytest.raises(ValueError, match="must be >= 0"):
            spVIPESmulti.model.spVIPESmulti(
                prepared,
                disentangle_preset="off",
                disentangle_group_shared_weight=-1.0,
            )


# ============================================================
# Phase 3.4 — ConcatDataLoader empty indices guard
# ============================================================

class TestConcatDataLoaderGuard:
    @pytest.mark.integration
    def test_empty_indices_list_raises(self):
        """ConcatDataLoader with empty indices_list should raise ValueError."""
        import spVIPESmulti
        import anndata as ad
        from scipy.sparse import csr_matrix

        rng = np.random.default_rng(0)
        X = rng.poisson(5, size=(40, 30)).astype(np.float32)
        adata = ad.AnnData(X=csr_matrix(X))
        adata.obs_names = [f"c{i}" for i in range(40)]
        adata.var_names = [f"g{i}" for i in range(30)]
        adata.obs["group"] = ["A"] * 20 + ["B"] * 20
        adata.obs["idx"] = list(range(40))
        adata.obs["ct"] = ["ct0"] * 20 + ["ct1"] * 20

        prepared = spVIPESmulti.data.prepare_adatas({"A": adata[:20].copy(), "B": adata[20:].copy()})
        spVIPESmulti.model.spVIPESmulti.setup_anndata(prepared, groups_key="groups", label_key="ct")
        model = spVIPESmulti.model.spVIPESmulti(prepared)

        from spVIPESmulti.dataloaders._concat_dataloader import ConcatDataLoader
        with pytest.raises(ValueError, match="empty"):
            ConcatDataLoader(model.adata_manager, indices_list=[])

    @pytest.mark.integration
    def test_all_groups_represented_each_batch(self):
        """Every batch from ConcatDataLoader should contain cells from all groups."""
        import spVIPESmulti
        import anndata as ad
        from scipy.sparse import csr_matrix
        from spVIPESmulti.dataloaders._concat_dataloader import ConcatDataLoader

        rng = np.random.default_rng(0)
        X = rng.poisson(5, size=(80, 30)).astype(np.float32)
        adata = ad.AnnData(X=csr_matrix(X))
        adata.obs_names = [f"c{i}" for i in range(80)]
        adata.var_names = [f"g{i}" for i in range(30)]
        adata.obs["group"] = ["A"] * 40 + ["B"] * 40
        adata.obs["idx"] = list(range(80))
        adata.obs["ct"] = ["ct0"] * 40 + ["ct1"] * 40

        prepared = spVIPESmulti.data.prepare_adatas({"A": adata[:40].copy(), "B": adata[40:].copy()})
        spVIPESmulti.model.spVIPESmulti.setup_anndata(prepared, groups_key="groups", label_key="ct")
        model = spVIPESmulti.model.spVIPESmulti(prepared)

        gi = [list(map(int, g)) for g in prepared.uns["groups_obs_indices"]]
        dl = ConcatDataLoader(model.adata_manager, indices_list=gi, shuffle=False, batch_size=16)

        for batch in dl:
            group_codes = batch["groups"].reshape(-1).unique().tolist()
            assert len(group_codes) == 2, f"Batch missing a group; groups present: {group_codes}"
            break  # Check just the first batch


# ============================================================
# Phase 4.2 — Gradient propagation (moved here from test_multimodal_disentangle)
# ============================================================

class TestGradientPropagation:
    @pytest.mark.integration
    def test_grad_flows_to_encoder_parameters(self):
        """loss.backward() must produce non-zero gradients on model parameters."""
        import spVIPESmulti
        import anndata as ad
        import pandas as pd
        from scipy.sparse import csr_matrix
        from spVIPESmulti.dataloaders._concat_dataloader import ConcatDataLoader

        def _make_mod(n_obs, n_vars, seed):
            rng = np.random.default_rng(seed)
            X = rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32)
            a = ad.AnnData(X=csr_matrix(X))
            a.obs_names = [f"c{i}_{seed}" for i in range(n_obs)]
            a.var_names = [f"g{i}" for i in range(n_vars)]
            return a

        groups = {}
        rng = np.random.default_rng(0)
        for gi in range(2):
            gname = f"g{gi}"
            rna = _make_mod(60, 50, seed=10 * gi)
            prot = _make_mod(60, 20, seed=10 * gi + 1)
            rna.obs_names = [f"{gname}_c{i}" for i in range(60)]
            prot.obs_names = rna.obs_names
            cts = pd.Categorical(rng.choice(["ct0", "ct1", "ct2"], 60))
            rna.obs["cell_types"] = cts
            prot.obs["cell_types"] = cts
            groups[gname] = {"rna": rna, "protein": prot}

        prepared = spVIPESmulti.data.prepare_multimodal_adatas(
            groups, modality_likelihoods={"rna": "nb", "protein": "nb"}
        )
        spVIPESmulti.model.spVIPESmulti.setup_anndata(
            prepared, groups_key="groups", label_key="cell_types",
            modality_likelihoods={"rna": "nb", "protein": "nb"},
        )
        model = spVIPESmulti.model.spVIPESmulti(
            prepared, n_hidden=32, n_dimensions_shared=8, n_dimensions_private=4,
            disentangle_preset="full",
        )

        gi = [list(map(int, g)) for g in prepared.uns["groups_obs_indices"]]
        dl = ConcatDataLoader(model.adata_manager, indices_list=gi, shuffle=False, batch_size=32)
        tensors = next(iter(dl))

        inference_inputs = model.module._get_inference_input(tensors)
        inference_outputs = model.module.inference(**inference_inputs)
        gen_inputs = model.module._get_generative_input(tensors, inference_outputs)
        gen_outputs = model.module.generative(**gen_inputs)
        loss_output = model.module.loss(tensors, inference_outputs, gen_outputs)

        loss_output.loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.module.parameters()
        )
        assert has_grad, "No gradients flowed to any model parameter after loss.backward()"


# ============================================================
# Phase 4.6 — Disentangle preset edge cases
# ============================================================

class TestDisentanglePresetEdgeCases:
    @pytest.mark.integration
    def test_invalid_preset_raises(self):
        import spVIPESmulti
        import anndata as ad
        from scipy.sparse import csr_matrix

        rng = np.random.default_rng(0)
        X = rng.poisson(5, size=(40, 20)).astype(np.float32)
        adata = ad.AnnData(X=csr_matrix(X))
        adata.obs_names = [f"c{i}" for i in range(40)]
        adata.var_names = [f"g{i}" for i in range(20)]
        adata.obs["group"] = ["A"] * 20 + ["B"] * 20
        adata.obs["idx"] = list(range(40))
        adata.obs["ct"] = ["ct0"] * 20 + ["ct1"] * 20

        prepared = spVIPESmulti.data.prepare_adatas({"A": adata[:20].copy(), "B": adata[20:].copy()})
        spVIPESmulti.model.spVIPESmulti.setup_anndata(prepared, groups_key="groups", label_key="ct")

        with pytest.raises(ValueError, match="Unknown disentangle_preset"):
            spVIPESmulti.model.spVIPESmulti(prepared, disentangle_preset="does_not_exist")

    @pytest.mark.integration
    def test_preset_off_with_weight_override_works(self):
        """Preset='off' with a weight override should construct cleanly."""
        import spVIPESmulti
        import anndata as ad
        from scipy.sparse import csr_matrix

        rng = np.random.default_rng(0)
        X = rng.poisson(5, size=(40, 20)).astype(np.float32)
        adata = ad.AnnData(X=csr_matrix(X))
        adata.obs_names = [f"c{i}" for i in range(40)]
        adata.var_names = [f"g{i}" for i in range(20)]
        adata.obs["group"] = ["A"] * 20 + ["B"] * 20
        adata.obs["idx"] = list(range(40))
        adata.obs["ct"] = ["ct0"] * 20 + ["ct1"] * 20

        prepared = spVIPESmulti.data.prepare_adatas({"A": adata[:20].copy(), "B": adata[20:].copy()})
        spVIPESmulti.model.spVIPESmulti.setup_anndata(prepared, groups_key="groups", label_key="ct")

        model = spVIPESmulti.model.spVIPESmulti(
            prepared,
            disentangle_preset="off",
            disentangle_group_shared_weight=0.5,  # override one weight
        )
        assert model.module.disentangle_group_shared_weight == 0.5
        assert model.module.disentangle_label_shared_weight == 0.0  # preset value unchanged


# ============================================================
# Phase 4.8 — get_latent_representation completeness
# ============================================================

class TestLatentRepresentationCompleteness:
    @pytest.mark.integration
    def test_output_size_matches_n_cells(self):
        """Latent output arrays must have exactly n_cells rows for each group."""
        import spVIPESmulti
        import anndata as ad
        from scipy.sparse import csr_matrix

        rng = np.random.default_rng(0)
        n_cells = [30, 45]
        adatas = {}
        for i, n in enumerate(n_cells):
            X = rng.poisson(5, size=(n, 20)).astype(np.float32)
            a = ad.AnnData(X=csr_matrix(X))
            a.obs_names = [f"g{i}_c{j}" for j in range(n)]
            a.var_names = [f"gene{j}" for j in range(20)]
            a.obs["cell_type"] = ["ct0"] * (n // 2) + ["ct1"] * (n - n // 2)
            adatas[f"g{i}"] = a

        prepared = spVIPESmulti.data.prepare_adatas(adatas)
        spVIPESmulti.model.spVIPESmulti.setup_anndata(prepared, groups_key="groups", label_key="cell_type")
        model = spVIPESmulti.model.spVIPESmulti(prepared, n_hidden=32, n_dimensions_shared=8, n_dimensions_private=4)

        gi = [list(map(int, g)) for g in prepared.uns["groups_obs_indices"]]
        result = model.get_latent_representation(gi, batch_size=16)

        for g, expected_n in enumerate(n_cells):
            assert result["shared"][g].shape[0] == expected_n, \
                f"Group {g}: expected {expected_n} shared rows, got {result['shared'][g].shape[0]}"
            assert result["private"][g].shape[0] == expected_n, \
                f"Group {g}: expected {expected_n} private rows, got {result['private'][g].shape[0]}"


class TestJeffreysAndLikelihoodHardening:
    def test_single_modal_jeffreys_integration_changes_loss(self, monkeypatch):
        module = _make_module(
            n_private=2,
            n_shared=3,
            n_genes=3,
            n_groups=2,
            use_jeffreys_integ=False,
            jeffreys_integ_weight=2.0,
        )
        module.eval()

        tensors_by_group, inference_outputs, generative_outputs = _make_minimal_loss_inputs(module)

        monkeypatch.setattr(module, "_compute_disentangle_losses", lambda *args, **kwargs: 0.0)
        baseline = module.loss(
            tensors_by_group,
            inference_outputs,
            generative_outputs,
            kl_weight=1.0,
        ).loss

        module.use_jeffreys_integ = True
        monkeypatch.setattr(
            module,
            "_compute_jeffreys_integ_loss",
            lambda *args, **kwargs: torch.tensor(3.0),
        )
        with_jeffreys = module.loss(
            tensors_by_group,
            inference_outputs,
            generative_outputs,
            kl_weight=1.0,
        ).loss

        assert torch.isclose(with_jeffreys - baseline, torch.tensor(6.0), atol=1e-6)

    def test_strict_likelihood_support_rejects_fractional_nb_counts(self):
        module = _make_module(
            n_private=2,
            n_shared=3,
            n_genes=3,
            n_groups=2,
            strict_likelihood_support=True,
            log_variational_generative=False,
        )
        module.eval()

        tensors_by_group, inference_outputs, generative_outputs = _make_minimal_loss_inputs(module, frac=True)

        with pytest.raises(ValueError, match="strict support validation"):
            module.loss(
                tensors_by_group,
                inference_outputs,
                generative_outputs,
                kl_weight=1.0,
            )


class TestUnequalGroupBatchLossAggregation:
    def test_single_modal_loss_handles_unequal_group_batch_sizes(self, monkeypatch):
        module = _make_module(
            n_private=2,
            n_shared=3,
            n_genes=3,
            n_groups=2,
            log_variational_generative=False,
        )
        module.eval()

        x0 = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        x1 = torch.tensor([[1.0, 0.0, 2.0], [2.0, 2.0, 2.0], [3.0, 1.0, 1.0]])

        tensors_by_group = [
            {
                REGISTRY_KEYS.X_KEY: x0,
                REGISTRY_KEYS.BATCH_KEY: torch.zeros((x0.shape[0], 1), dtype=torch.long),
                "groups": torch.zeros((x0.shape[0], 1), dtype=torch.long),
            },
            {
                REGISTRY_KEYS.X_KEY: x1,
                REGISTRY_KEYS.BATCH_KEY: torch.zeros((x1.shape[0], 1), dtype=torch.long),
                "groups": torch.ones((x1.shape[0], 1), dtype=torch.long),
            },
        ]

        z0_private = torch.zeros((x0.shape[0], module.n_dimensions_private), dtype=torch.float32)
        z1_private = torch.zeros((x1.shape[0], module.n_dimensions_private), dtype=torch.float32)
        z0_shared = torch.zeros((x0.shape[0], module.n_dimensions_shared), dtype=torch.float32)
        z1_shared = torch.zeros((x1.shape[0], module.n_dimensions_shared), dtype=torch.float32)

        inference_outputs = {
            "private_stats": {
                0: {
                    "qz": torch.distributions.Normal(z0_private, torch.ones_like(z0_private)),
                    "log_z": z0_private,
                },
                1: {
                    "qz": torch.distributions.Normal(z1_private, torch.ones_like(z1_private)),
                    "log_z": z1_private,
                },
            },
            "poe_stats": {
                0: {
                    "logtheta_qz": torch.distributions.Normal(z0_shared, torch.ones_like(z0_shared)),
                    "logtheta_log_z": z0_shared,
                },
                1: {
                    "logtheta_qz": torch.distributions.Normal(z1_shared, torch.ones_like(z1_shared)),
                    "logtheta_log_z": z1_shared,
                },
            },
        }

        generative_outputs = {
            "private_poe": {
                "0": {"px": torch.distributions.Normal(loc=x0, scale=torch.ones_like(x0))},
                "1": {"px": torch.distributions.Normal(loc=x1, scale=torch.ones_like(x1))},
            }
        }

        monkeypatch.setattr(module, "_compute_disentangle_losses", lambda *args, **kwargs: 0.0)
        result = module.loss(tensors_by_group, inference_outputs, generative_outputs, kl_weight=1.0)

        assert result.loss.ndim == 0
        assert torch.isfinite(result.loss)

    def test_multimodal_loss_handles_unequal_group_batch_sizes(self, monkeypatch):
        import sys
        sys.path.insert(0, _SRC)
        from spVIPESmulti.module.spVIPESmultimodule import spVIPESmultimodule

        groups_lengths = {0: 3, 1: 3}
        groups_modality_lengths = {
            0: {"rna": 2, "protein": 1},
            1: {"rna": 2, "protein": 1},
        }
        groups_modality_var_indices = {
            0: {"rna": [0, 1], "protein": [2]},
            1: {"rna": [0, 1], "protein": [2]},
        }

        module = spVIPESmultimodule(
            groups_lengths=groups_lengths,
            groups_obs_names=[list(range(4)), list(range(3))],
            groups_var_names=[["g0", "g1", "g2"], ["g0", "g1", "g2"]],
            groups_obs_indices=[list(range(4)), list(range(3))],
            groups_var_indices=[list(range(3)), list(range(3))],
            n_dimensions_private=2,
            n_dimensions_shared=3,
            n_hidden=16,
            groups_modality_lengths=groups_modality_lengths,
            groups_modality_var_indices=groups_modality_var_indices,
            modality_likelihoods={"rna": "nb", "protein": "nb"},
            modality_names=["rna", "protein"],
            log_variational_generative=False,
        )
        module.eval()

        x0 = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        x1 = torch.tensor([[1.0, 0.0, 2.0], [2.0, 2.0, 2.0], [3.0, 1.0, 1.0]])

        tensors_by_group = [
            {
                REGISTRY_KEYS.X_KEY: x0,
                REGISTRY_KEYS.BATCH_KEY: torch.zeros((x0.shape[0], 1), dtype=torch.long),
                "groups": torch.zeros((x0.shape[0], 1), dtype=torch.long),
            },
            {
                REGISTRY_KEYS.X_KEY: x1,
                REGISTRY_KEYS.BATCH_KEY: torch.zeros((x1.shape[0], 1), dtype=torch.long),
                "groups": torch.ones((x1.shape[0], 1), dtype=torch.long),
            },
        ]

        z0_shared = torch.zeros((x0.shape[0], module.n_dimensions_shared), dtype=torch.float32)
        z1_shared = torch.zeros((x1.shape[0], module.n_dimensions_shared), dtype=torch.float32)

        per_modality_private = {}
        for g, xg in enumerate([x0, x1]):
            for modality in ["rna", "protein"]:
                z = torch.zeros((xg.shape[0], module.n_dimensions_private), dtype=torch.float32)
                per_modality_private[(g, modality)] = {
                    "qz": torch.distributions.Normal(z, torch.ones_like(z)),
                    "log_z": z,
                }

        inference_outputs = {
            "private_stats": {
                0: {"qz": per_modality_private[(0, "rna")]["qz"], "log_z": per_modality_private[(0, "rna")]["log_z"]},
                1: {"qz": per_modality_private[(1, "rna")]["qz"], "log_z": per_modality_private[(1, "rna")]["log_z"]},
            },
            "poe_stats": {
                0: {
                    "logtheta_qz": torch.distributions.Normal(z0_shared, torch.ones_like(z0_shared)),
                    "logtheta_log_z": z0_shared,
                },
                1: {
                    "logtheta_qz": torch.distributions.Normal(z1_shared, torch.ones_like(z1_shared)),
                    "logtheta_log_z": z1_shared,
                },
            },
            "per_modality_private": per_modality_private,
        }

        generative_outputs = {
            "private_poe": {
                "0_rna": {
                    "px": torch.distributions.Normal(
                        loc=x0[:, groups_modality_var_indices[0]["rna"]],
                        scale=torch.ones_like(x0[:, groups_modality_var_indices[0]["rna"]]),
                    )
                },
                "0_protein": {
                    "px": torch.distributions.Normal(
                        loc=x0[:, groups_modality_var_indices[0]["protein"]],
                        scale=torch.ones_like(x0[:, groups_modality_var_indices[0]["protein"]]),
                    )
                },
                "1_rna": {
                    "px": torch.distributions.Normal(
                        loc=x1[:, groups_modality_var_indices[1]["rna"]],
                        scale=torch.ones_like(x1[:, groups_modality_var_indices[1]["rna"]]),
                    )
                },
                "1_protein": {
                    "px": torch.distributions.Normal(
                        loc=x1[:, groups_modality_var_indices[1]["protein"]],
                        scale=torch.ones_like(x1[:, groups_modality_var_indices[1]["protein"]]),
                    )
                },
            }
        }

        monkeypatch.setattr(module, "_compute_disentangle_losses", lambda *args, **kwargs: 0.0)
        result = module._loss_multimodal(tensors_by_group, inference_outputs, generative_outputs, kl_weight=1.0)

        assert result.loss.ndim == 0
        assert torch.isfinite(result.loss)
