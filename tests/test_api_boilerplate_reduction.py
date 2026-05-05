import warnings

import anndata as ad
import numpy as np
import pytest

import spVIPESmulti


@pytest.mark.integration
def test_get_latent_representation_auto_infers_groups_warns_once():
    rng = np.random.default_rng(0)
    a1 = ad.AnnData(X=rng.poisson(5, size=(16, 10)).astype(np.float32))
    a2 = ad.AnnData(X=rng.poisson(5, size=(14, 10)).astype(np.float32))
    a1.var_names = [f"g{i}" for i in range(10)]
    a2.var_names = [f"g{i}" for i in range(10)]

    combined = spVIPESmulti.data.prepare_adatas({"group-1": a1, "group 2": a2})
    spVIPESmulti.model.spVIPESmulti.setup_anndata(combined, groups_key="groups")
    model = spVIPESmulti.model.spVIPESmulti(
        combined,
        n_hidden=16,
        n_dimensions_shared=4,
        n_dimensions_private=3,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result1 = model.get_latent_representation(batch_size=8)
        result2 = model.get_latent_representation(batch_size=8)

    infer_msgs = [w for w in caught if "inferred from adata.uns['groups_obs_indices']" in str(w.message)]
    assert len(infer_msgs) == 1
    assert result1["shared"][0].shape[0] == 16
    assert result2["private"][1].shape[0] == 14


@pytest.mark.integration
def test_embed_writes_expected_keys_and_returns_payload():
    rng = np.random.default_rng(1)
    a1 = ad.AnnData(X=rng.poisson(5, size=(12, 8)).astype(np.float32))
    a2 = ad.AnnData(X=rng.poisson(5, size=(10, 8)).astype(np.float32))
    a1.var_names = [f"g{i}" for i in range(8)]
    a2.var_names = [f"g{i}" for i in range(8)]

    combined = spVIPESmulti.data.prepare_adatas({"group-1": a1, "group 2": a2})
    spVIPESmulti.model.spVIPESmulti.setup_anndata(combined, groups_key="groups")
    model = spVIPESmulti.model.spVIPESmulti(
        combined,
        n_hidden=16,
        n_dimensions_shared=5,
        n_dimensions_private=2,
    )

    payload = model.embed(batch_size=8)

    assert payload["keys"]["shared"] == "X_spvm_shared"
    assert "group_1" in payload["keys"]["private"]
    assert "group_2" in payload["keys"]["private"]

    shared_key = payload["keys"]["shared"]
    assert shared_key in combined.obsm
    assert payload["shared"].shape == (combined.n_obs, 5)


@pytest.mark.integration
def test_embed_overwrite_guard_is_transactional():
    rng = np.random.default_rng(2)
    a1 = ad.AnnData(X=rng.poisson(5, size=(10, 8)).astype(np.float32))
    a2 = ad.AnnData(X=rng.poisson(5, size=(10, 8)).astype(np.float32))
    a1.var_names = [f"g{i}" for i in range(8)]
    a2.var_names = [f"g{i}" for i in range(8)]

    combined = spVIPESmulti.data.prepare_adatas({"g1": a1, "g2": a2})
    spVIPESmulti.model.spVIPESmulti.setup_anndata(combined, groups_key="groups")
    model = spVIPESmulti.model.spVIPESmulti(
        combined,
        n_hidden=16,
        n_dimensions_shared=4,
        n_dimensions_private=2,
    )

    model.embed(batch_size=8)
    before = set(combined.obsm.keys())

    with pytest.raises(ValueError, match="overwrite"):
        model.embed(batch_size=8, overwrite=False)

    after = set(combined.obsm.keys())
    assert before == after


@pytest.mark.integration
def test_train_auto_infers_group_indices(monkeypatch):
    rng = np.random.default_rng(3)
    a1 = ad.AnnData(X=rng.poisson(5, size=(12, 6)).astype(np.float32))
    a2 = ad.AnnData(X=rng.poisson(5, size=(12, 6)).astype(np.float32))
    a1.var_names = [f"g{i}" for i in range(6)]
    a2.var_names = [f"g{i}" for i in range(6)]

    combined = spVIPESmulti.data.prepare_adatas({"g1": a1, "g2": a2})
    spVIPESmulti.model.spVIPESmulti.setup_anndata(combined, groups_key="groups")
    model = spVIPESmulti.model.spVIPESmulti(
        combined,
        n_hidden=8,
        n_dimensions_shared=3,
        n_dimensions_private=2,
    )

    from spVIPESmulti.model.base import training_mixin

    monkeypatch.setattr(training_mixin.PatchedTrainRunner, "__call__", lambda self: None)
    model.train(max_epochs=1, batch_size=4, accelerator="cpu", devices=1)
