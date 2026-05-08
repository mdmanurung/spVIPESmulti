"""W-030 / F-DA-NOTEST: ``differential_abundance`` must return calibrated p-values.

Surface: src/spVIPESmulti/model/spvipesmulti.py (differential_abundance)
Verification: V2.4 (primary audit §5)
"""
import pytest

pytestmark = [pytest.mark.audit_regression]


@pytest.mark.integration
def test_da_returns_pvalue_column():
    """The return value of ``differential_abundance`` must include ``p_value`` and ``q_value``."""
    pytest.importorskip("scvi")
    import anndata
    import numpy as np
    import spVIPESmulti
    from spVIPESmulti.data import prepare_adatas

    rng = np.random.default_rng(42)
    n = 60
    n_genes = 30
    X_a = rng.negative_binomial(5, 0.5, size=(n, n_genes)).astype(np.float32)
    X_b = rng.negative_binomial(5, 0.5, size=(n, n_genes)).astype(np.float32)
    ad_a = anndata.AnnData(X_a)
    ad_b = anndata.AnnData(X_b)
    ad_a.obs_names = [f"A_{i}" for i in range(n)]
    ad_b.obs_names = [f"B_{i}" for i in range(n)]
    ad_a.var_names = [f"g{i}" for i in range(n_genes)]
    ad_b.var_names = [f"g{i}" for i in range(n_genes)]
    ad_a.obs["sample"] = "s1"
    ad_b.obs["sample"] = "s1"
    adata = prepare_adatas({"A": ad_a, "B": ad_b})
    spVIPESmulti.model.spVIPESmulti.setup_anndata(adata, groups_key="groups", sample_key="sample")
    model = spVIPESmulti.model.spVIPESmulti(adata, n_dimensions_shared=4, n_dimensions_private=4)
    model.train(max_epochs=1, batch_size=120, accelerator="cpu", devices=1)

    result = model.differential_abundance(n_permutations=10)
    df = result["scores"]
    assert "p_value" in df.columns, "p_value column missing from differential_abundance output"
    assert "q_value" in df.columns, "q_value column missing from differential_abundance output"
    assert df["p_value"].between(0.0, 1.0).all(), "p_values must be in [0, 1]"
    assert df["q_value"].between(0.0, 1.0).all(), "q_values must be in [0, 1]"


@pytest.mark.slow
@pytest.mark.integration
def test_da_pvalue_uniform_under_null():
    """Under identical-composition two-group input, p-values should be ~Uniform.
    This is a slow calibration test; run with --runaudit.
    """
    pytest.importorskip("scvi")
    import anndata
    import numpy as np
    from scipy.stats import kstest
    import spVIPESmulti
    from spVIPESmulti.data import prepare_adatas
    from ._synthdata import make_paired_two_group

    data = make_paired_two_group(share_frac=1.0, n_cells_per=300, n_genes=100, seed=7)
    ad_a = anndata.AnnData(data["A"].astype(np.float32))
    ad_b = anndata.AnnData(data["B"].astype(np.float32))
    ad_a.obs["sample"] = "s1"
    ad_b.obs["sample"] = "s1"
    adata = prepare_adatas({"A": ad_a, "B": ad_b})
    spVIPESmulti.spVIPESmulti.setup_anndata(adata, sample_key="sample")
    model = spVIPESmulti.spVIPESmulti(adata, n_dimensions_shared=8, n_dimensions_private=4)
    model.train(max_epochs=10, batch_size=600, accelerator="cpu", devices=1)

    result = model.differential_abundance(n_permutations=200)
    p_vals = result["scores"]["p_value"].values
    stat, pval = kstest(p_vals, "uniform")
    assert pval > 0.01, f"KS test failed: p_vals not ~Uniform (KS stat={stat:.3f}, p={pval:.4f})"
