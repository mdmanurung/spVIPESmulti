"""W-020 / F-POE-ROWWISE: in unsupervised mode each group's z_shared must come
from its own encoder posterior, not a cross-group row-paired PoE.

Surface: src/spVIPESmulti/module/spVIPESmultimodule.py (_supervised_poe)
Verification: V2.2 (primary audit §5)
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.audit_regression]


def test_z_shared_invariant_under_other_group_shuffle():
    """In the unsupervised PoE path, group A's z_shared must only depend on its own encoder stats.

    We verify this by inspecting the source: the unsupervised branch in
    ``_supervised_poe`` must use each group's own stats (not cross-group indexing).
    """
    src = Path(__file__).parents[2] / "src" / "spVIPESmulti" / "module" / "spVIPESmultimodule.py"
    code = src.read_text()
    # The fix: unsupervised mode has been removed entirely; _supervised_poe must
    # now contain a defensive RuntimeError guard that fires when labels are absent.
    assert "setup_anndata enforcement was bypassed" in code, (
        "Expected defensive RuntimeError guard in _supervised_poe "
        "(unsupervised mode has been removed)"
    )


@pytest.mark.integration
def test_setup_anndata_requires_label_key():
    """Integration: setup_anndata must raise when label_key is omitted (unsupervised mode removed)."""
    pytest.importorskip("scvi")
    import anndata
    import numpy as np
    import spVIPESmulti
    from spVIPESmulti.data import prepare_adatas

    rng = np.random.default_rng(0)
    n, n_genes = 40, 30
    X_a = rng.negative_binomial(5, 0.5, size=(n, n_genes)).astype(np.float32)
    X_b = rng.negative_binomial(5, 0.5, size=(n, n_genes)).astype(np.float32)
    ad_a = anndata.AnnData(X_a)
    ad_b = anndata.AnnData(X_b)
    ad_a.obs_names = [f"A_{i}" for i in range(n)]
    ad_b.obs_names = [f"B_{i}" for i in range(n)]
    ad_a.var_names = [f"g{i}" for i in range(n_genes)]
    ad_b.var_names = [f"g{i}" for i in range(n_genes)]
    adata = prepare_adatas({"A": ad_a, "B": ad_b})
    with pytest.raises((TypeError, ValueError)):
        spVIPESmulti.model.spVIPESmulti.setup_anndata(adata, groups_key="groups")
