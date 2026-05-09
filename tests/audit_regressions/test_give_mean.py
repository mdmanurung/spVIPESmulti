"""W-001 / F-GIVE-MEAN: ``get_latent_representation(give_mean=True, normalized=False)``
must return the posterior mean, not a single rsample.

Surface: src/spVIPESmulti/model/spvipesmulti.py
Verification: V2.5 (primary audit §5)
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.audit_regression]

# ---------------------------------------------------------------------------
# Lightweight unit test — no scvi-tools import needed.
# We test the branch directly in the helper that populates per-group outputs.
# ---------------------------------------------------------------------------


def _load_model_mod():
    """Import model module without triggering full package __init__."""
    src = Path(__file__).parents[2] / "src" / "spVIPESmulti" / "model" / "spvipesmulti.py"
    # Build a minimal stub environment so the module can be imported
    # without real scvi-tools being present.
    pytest.importorskip("scvi")  # skip entire test if scvi not available
    import spVIPESmulti.model.spvipesmulti as mod
    return mod


@pytest.mark.integration
def test_give_mean_unnormalized_is_deterministic():
    """Two calls with identical args must produce bit-identical embeddings (no rsample)."""
    pytest.importorskip("scvi")
    import anndata
    import numpy as np
    import spVIPESmulti
    from spVIPESmulti.data import prepare_adatas

    rng = np.random.default_rng(0)
    n_cells_per = 60
    n_genes = 30
    X_a = rng.negative_binomial(5, 0.5, size=(n_cells_per, n_genes)).astype(np.float32)
    X_b = rng.negative_binomial(5, 0.5, size=(n_cells_per, n_genes)).astype(np.float32)
    ad_a = anndata.AnnData(X_a)
    ad_b = anndata.AnnData(X_b)
    ad_a.obs_names = [f"A_{i}" for i in range(n_cells_per)]
    ad_b.obs_names = [f"B_{i}" for i in range(n_cells_per)]
    ad_a.var_names = [f"g{i}" for i in range(n_genes)]
    ad_b.var_names = [f"g{i}" for i in range(n_genes)]
    ad_a.obs["cell_type"] = ["A"] * (n_cells_per // 2) + ["B"] * (n_cells_per - n_cells_per // 2)
    ad_b.obs["cell_type"] = ["A"] * (n_cells_per // 2) + ["B"] * (n_cells_per - n_cells_per // 2)

    adata = prepare_adatas({"A": ad_a, "B": ad_b})
    spVIPESmulti.model.spVIPESmulti.setup_anndata(adata, groups_key="groups", label_key="cell_type")
    model = spVIPESmulti.model.spVIPESmulti(adata, n_dimensions_shared=4, n_dimensions_private=4)
    model.train(max_epochs=1, batch_size=120, accelerator="cpu", devices=1)

    out1 = model.get_latent_representation(give_mean=True, normalized=False)
    out2 = model.get_latent_representation(give_mean=True, normalized=False)
    for g in out1["shared_reordered"]:
        assert np.allclose(out1["shared_reordered"][g], out2["shared_reordered"][g], atol=0.0, rtol=0.0), (
            f"give_mean=True must be deterministic for shared group {g}; got different outputs"
        )
    for g in out1["private_reordered"]:
        assert np.allclose(out1["private_reordered"][g], out2["private_reordered"][g], atol=0.0, rtol=0.0), (
            f"give_mean=True must be deterministic for private group {g}; got different outputs"
        )
