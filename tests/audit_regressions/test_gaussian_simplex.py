"""W-012 / F-GAUSSIAN-SIMPLEX: ``setup_anndata`` must warn when a modality uses
'gaussian' likelihood that data must be pre-normalised to [0, 1].

Surface: src/spVIPESmulti/model/spvipesmulti.py (setup_anndata)
Verification: V2.3 (primary audit §5)
"""
import warnings

import pytest

pytestmark = [pytest.mark.audit_regression]


@pytest.mark.integration
def test_gaussian_mean_can_match_lognorm_range():
    """setup_anndata warns when gaussian modality is present (W-012)."""
    pytest.importorskip("scvi")
    import warnings
    import anndata
    import numpy as np
    import spVIPESmulti

    rng = np.random.default_rng(0)
    n_cells = 40
    n_rna = 20
    n_prot = 8

    # Build group A
    X_rna_a = rng.negative_binomial(5, 0.5, size=(n_cells, n_rna)).astype(np.float32)
    X_prot_a = rng.normal(0, 1, size=(n_cells, n_prot)).astype(np.float32)
    ad_rna_a = anndata.AnnData(X_rna_a)
    ad_prot_a = anndata.AnnData(X_prot_a)
    ad_rna_a.obs_names = [f"A_{i}" for i in range(n_cells)]
    ad_prot_a.obs_names = ad_rna_a.obs_names
    ad_rna_a.var_names = [f"gene{i}" for i in range(n_rna)]
    ad_prot_a.var_names = [f"prot{i}" for i in range(n_prot)]

    # Build group B
    X_rna_b = rng.negative_binomial(5, 0.5, size=(n_cells, n_rna)).astype(np.float32)
    X_prot_b = rng.normal(0, 1, size=(n_cells, n_prot)).astype(np.float32)
    ad_rna_b = anndata.AnnData(X_rna_b)
    ad_prot_b = anndata.AnnData(X_prot_b)
    ad_rna_b.obs_names = [f"B_{i}" for i in range(n_cells)]
    ad_prot_b.obs_names = ad_rna_b.obs_names
    ad_rna_b.var_names = [f"gene{i}" for i in range(n_rna)]
    ad_prot_b.var_names = [f"prot{i}" for i in range(n_prot)]
    for _a in (ad_rna_a, ad_prot_a):
        _a.obs["cell_type"] = ["ct0"] * (n_cells // 2) + ["ct1"] * (n_cells - n_cells // 2)
    for _a in (ad_rna_b, ad_prot_b):
        _a.obs["cell_type"] = ["ct0"] * (n_cells // 2) + ["ct1"] * (n_cells - n_cells // 2)

    adata = spVIPESmulti.data.prepare_multimodal_adatas(
        {"A": {"rna": ad_rna_a, "protein": ad_prot_a},
         "B": {"rna": ad_rna_b, "protein": ad_prot_b}},
        modality_likelihoods={"rna": "nb", "protein": "gaussian"},
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spVIPESmulti.model.spVIPESmulti.setup_anndata(
            adata,
            groups_key="groups",
            label_key="cell_type",
            modality_likelihoods={"rna": "nb", "protein": "gaussian"},
        )
    messages = [str(wi.message) for wi in w if issubclass(wi.category, UserWarning)]
    assert any("gaussian" in m.lower() or "simplex" in m.lower() for m in messages), (
        f"Expected a UserWarning about Gaussian/simplex; got: {messages}"
    )
