"""Probe whether get_latent_representation forwards the indices argument."""

from __future__ import annotations

import importlib

import numpy as np
from anndata import AnnData

model_mod = importlib.import_module("spVIPESmulti.model.spvipesmulti")
Model = model_mod.spVIPESmulti


class _FakeConcatDataLoader:
    """Capture constructor arguments without loading AnnData tensors."""

    def __init__(self, adata_manager, indices_list, **kwargs):
        self.adata_manager = adata_manager
        self.indices_list = indices_list
        self.kwargs = kwargs


def main() -> None:
    """Run the probe and print captured loader indices."""
    adata = AnnData(np.zeros((6, 2), dtype=np.float32))
    adata.uns["groups_obs_indices"] = [[0, 1, 2], [3, 4, 5]]

    model = object.__new__(Model)
    model._adata_manager = object()
    model._validate_anndata = lambda value: value
    model._warn_group_indices_auto_inferred = lambda caller: None
    model._process_batches = lambda loader, *args: {"indices_list": loader.indices_list}
    model._format_results = lambda results, n_per_group: {
        "loader_indices_list": results["indices_list"],
        "n_per_group": n_per_group,
    }

    original_loader = model_mod.ConcatDataLoader
    model_mod.ConcatDataLoader = _FakeConcatDataLoader
    try:
        result = Model.get_latent_representation(
            model,
            adata=adata,
            indices=[1, 4],
            batch_size=2,
        )
    finally:
        model_mod.ConcatDataLoader = original_loader

    print(result)


if __name__ == "__main__":
    main()
