import pytest
import numpy as np
import anndata as ad
from spVIPES.model.spvipes import spVIPES
from spVIPES.model.base.training_mixin import MultiGroupTrainingMixin

def make_dummy_adata(n_obs=20, n_vars=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32)
    obs = {"celltype": ["A"] * (n_obs // 2) + ["B"] * (n_obs - n_obs // 2)}
    var = {"gene_symbols": [f"gene{i}" for i in range(n_vars)]}
    return ad.AnnData(X=X, obs=obs, var=var)

def test_multigroup_training_runs():
    adata1 = make_dummy_adata(20, 10, seed=1)
    adata2 = make_dummy_adata(18, 10, seed=2)
    from spVIPES.data.prepare_adatas import prepare_adatas
    adata = prepare_adatas({"g1": adata1, "g2": adata2})
    spVIPES.setup_anndata(adata, groups_key="groups")
    model = spVIPES(adata, n_hidden=8, n_dimensions_shared=2, n_dimensions_private=2, dropout_rate=0.1)
    group_indices_list = adata.uns["groups_obs_indices"]
    # Should not raise TypeError
    model.train(group_indices_list=group_indices_list, max_epochs=1, batch_size=4)

if __name__ == "__main__":
    test_multigroup_training_runs()
    print("Test passed.")
