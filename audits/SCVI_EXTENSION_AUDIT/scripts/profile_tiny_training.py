"""Run a tiny CPU training profile through Lightning's profiler."""

from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData

from spVIPESmulti.data.prepare_adatas import prepare_adatas
from spVIPESmulti.model.spvipesmulti import spVIPESmulti


def _make_adata(n_obs: int, n_vars: int, seed: int) -> AnnData:
    """Create a tiny count AnnData for profiler smoke testing."""
    rng = np.random.default_rng(seed)
    x = rng.poisson(3, size=(n_obs, n_vars)).astype(np.float32)
    adata = AnnData(X=x)
    adata.obs["celltype"] = ["A" if i % 2 == 0 else "B" for i in range(n_obs)]
    adata.obs_names = [f"c{seed}_{i}" for i in range(n_obs)]
    adata.var_names = [f"g{i}" for i in range(n_vars)]
    return adata


def main() -> None:
    """Profile one tiny training epoch with Lightning's simple profiler."""
    torch.cuda.is_available = lambda: False
    torch.manual_seed(0)
    np.random.seed(0)

    adata = prepare_adatas(
        {
            "g1": _make_adata(16, 8, 1),
            "g2": _make_adata(16, 8, 2),
        }
    )
    spVIPESmulti.setup_anndata(adata, groups_key="groups", label_key="celltype")
    model = spVIPESmulti(
        adata,
        n_hidden=8,
        n_dimensions_shared=2,
        n_dimensions_private=2,
        dropout_rate=0.0,
    )
    model.train(
        group_indices_list=adata.uns["groups_obs_indices"],
        max_epochs=1,
        batch_size=4,
        train_size=0.75,
        validation_size=0.25,
        accelerator="cpu",
        devices=1,
        profiler="simple",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


if __name__ == "__main__":
    main()
