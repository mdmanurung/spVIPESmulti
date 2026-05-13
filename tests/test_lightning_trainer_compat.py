import anndata as ad
import numpy as np
import torch

from spVIPESmulti.model.spvipesmulti import spVIPESmulti


def make_dummy_adata(n_obs=20, n_vars=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(5, size=(n_obs, n_vars)).astype(np.float32)
    obs = {"celltype": ["A"] * (n_obs // 2) + ["B"] * (n_obs - n_obs // 2)}
    var = {"gene_symbols": [f"gene{i}" for i in range(n_vars)]}
    return ad.AnnData(X=X, obs=obs, var=var)


def test_multigroup_training_runs(monkeypatch):
    # Keep this compatibility test independent of local CUDA driver state.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    adata1 = make_dummy_adata(20, 10, seed=1)
    adata2 = make_dummy_adata(18, 10, seed=2)
    from spVIPESmulti.data.prepare_adatas import prepare_adatas

    adata = prepare_adatas({"g1": adata1, "g2": adata2})
    spVIPESmulti.setup_anndata(adata, groups_key="groups")
    model = spVIPESmulti(adata, n_hidden=8, n_dimensions_shared=2, n_dimensions_private=2, dropout_rate=0.1)
    group_indices_list = adata.uns["groups_obs_indices"]
    # Should not raise TypeError
    model.train(
        group_indices_list=group_indices_list,
        max_epochs=1,
        batch_size=4,
        train_size=0.75,
        validation_size=0.25,
        accelerator="cpu",
        devices=1,
    )
    assert "elbo_validation" in model.history
    assert "reconstruction_loss_validation" in model.history


def test_cosine_lr_scheduler(monkeypatch):
    """CosineAnnealingLR decays LR and completes training without errors."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    adata1 = make_dummy_adata(20, 10, seed=3)
    adata2 = make_dummy_adata(18, 10, seed=4)
    from spVIPESmulti.data.prepare_adatas import prepare_adatas

    adata = prepare_adatas({"g1": adata1, "g2": adata2})
    spVIPESmulti.setup_anndata(adata, groups_key="groups")
    model = spVIPESmulti(adata, n_hidden=8, n_dimensions_shared=2, n_dimensions_private=2, dropout_rate=0.1)
    group_indices_list = adata.uns["groups_obs_indices"]

    MAX_EPOCHS = 3
    INITIAL_LR = 1e-3
    model.train(
        group_indices_list=group_indices_list,
        max_epochs=MAX_EPOCHS,
        batch_size=4,
        train_size=0.75,
        validation_size=0.25,
        accelerator="cpu",
        devices=1,
        plan_kwargs={
            "lr": INITIAL_LR,
            "lr_scheduler_type": "cosine",
            "lr_min": 1e-5,
        },
    )

    # Cosine schedule must have decayed the LR by the final epoch
    optimizer = model.trainer.optimizers[0]
    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr < INITIAL_LR, f"Expected LR to decay below {INITIAL_LR} with cosine schedule, got {final_lr}"
    # Training history should still be populated
    assert "elbo_train" in model.history


if __name__ == "__main__":
    test_multigroup_training_runs()
    print("Test passed.")
