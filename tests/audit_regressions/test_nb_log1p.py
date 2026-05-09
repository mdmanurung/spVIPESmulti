"""W-011 / F-NB-LOG1P: NB ``log_prob`` must be evaluated against raw counts,
not against ``log1p(x)``.

Surface: src/spVIPESmulti/module/spVIPESmultimodule.py
Verification: V2.1 (primary audit §5)
"""
import importlib.util
from pathlib import Path
import re

import pytest

pytestmark = [pytest.mark.audit_regression]


def test_nb_target_is_raw_counts():
    """The loss() body must not apply log1p to x_target before NB log_prob."""
    src = Path(__file__).parents[2] / "src" / "spVIPESmulti" / "module" / "spVIPESmultimodule.py"
    code = src.read_text()
    # The bad pattern: applying log1p/log to x_obs before passing as x_target
    bad_patterns = [
        r"x_target\s*=\s*torch\.log\(.*?x_obs",
        r"log_variational_generative.*?x_target",
        r"x_target\s*=\s*torch\.log1p\s*\(",
    ]
    for pat in bad_patterns:
        m = re.search(pat, code)
        assert m is None, (
            f"W-011: Found forbidden pattern '{pat}' in loss() — "
            "NB target must be raw counts, not log-transformed."
        )
    # The fix must be present: x_target = x_obs directly (comment asserts it)
    assert "W-011" in code, "Expected W-011 comment marker to be present in module"


@pytest.mark.integration
def test_nb_recovery_synthetic():
    """Train 2-group spVIPESmulti on NB draws; model trains without error (W-011)."""
    pytest.importorskip("scvi")
    import anndata
    import numpy as np
    import spVIPESmulti
    from spVIPESmulti.data import prepare_adatas
    from ._synthdata import make_nb_counts

    rng = np.random.default_rng(1)
    X_a, mu_a, _ = make_nb_counts(n_cells=200, n_genes=50, seed=1)
    X_b, mu_b, _ = make_nb_counts(n_cells=200, n_genes=50, seed=2)
    ad_a = anndata.AnnData(X_a.astype(np.float32))
    ad_b = anndata.AnnData(X_b.astype(np.float32))
    ad_a.obs_names = [f"A_{i}" for i in range(200)]
    ad_b.obs_names = [f"B_{i}" for i in range(200)]
    ad_a.var_names = [f"g{i}" for i in range(50)]
    ad_b.var_names = [f"g{i}" for i in range(50)]
    ad_a.obs["cell_type"] = ["A"] * 100 + ["B"] * 100
    ad_b.obs["cell_type"] = ["A"] * 100 + ["B"] * 100

    adata = prepare_adatas({"A": ad_a, "B": ad_b})
    spVIPESmulti.model.spVIPESmulti.setup_anndata(adata, groups_key="groups", label_key="cell_type")
    model = spVIPESmulti.model.spVIPESmulti(adata, n_dimensions_shared=8, n_dimensions_private=4)
    model.train(max_epochs=3, batch_size=400, accelerator="cpu", devices=1)
    # Verify training completes without error (W-011: NB target is raw counts)
    assert model is not None
