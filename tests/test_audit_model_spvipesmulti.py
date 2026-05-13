from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from anndata import AnnData


class _FakeConcatDataLoader:
    """Capture loader inputs without touching AnnData storage."""

    def __init__(self, adata_manager: object, indices_list: list[list[int]], **kwargs: Any) -> None:
        self.adata_manager = adata_manager
        self.indices_list = indices_list
        self.kwargs = kwargs


def _make_registered_shape_adata() -> AnnData:
    """Create a tiny AnnData carrying the group index metadata expected by the model."""

    adata = AnnData(np.zeros((6, 2), dtype=np.float32))
    adata.uns["groups_obs_indices"] = [[0, 1, 2], [3, 4, 5]]
    return adata


@pytest.mark.xfail(strict=True, reason="INT-002 pending: get_latent_representation ignores indices")
def test_get_latent_representation_indices_subset_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """The indices argument should restrict the per-group loader indices."""

    model_mod = importlib.import_module("spVIPESmulti.model.spvipesmulti")
    model_cls = model_mod.spVIPESmulti
    adata = _make_registered_shape_adata()

    model = object.__new__(model_cls)
    model._adata_manager = object()
    model._validate_anndata = lambda value: value
    model._warn_group_indices_auto_inferred = lambda caller: None
    model._process_batches = lambda loader, *args: {"indices_list": loader.indices_list}
    model._format_results = lambda results, n_per_group: results

    monkeypatch.setattr(model_mod, "ConcatDataLoader", _FakeConcatDataLoader)

    result = model_cls.get_latent_representation(
        model,
        adata=adata,
        indices=[1, 4],
        batch_size=2,
    )

    assert result["indices_list"] == [[1], [4]]


@pytest.mark.xfail(
    strict=True,
    reason="INT-005 pending: get_latent_representation validates new AnnData but keeps self.adata_manager",
)
def test_get_latent_representation_uses_validated_adata_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """A posterior call with adata should load from that adata's validated manager."""

    model_mod = importlib.import_module("spVIPESmulti.model.spvipesmulti")
    model_cls = model_mod.spVIPESmulti
    adata = _make_registered_shape_adata()
    original_manager = object()
    validated_manager = object()

    model = object.__new__(model_cls)
    model._adata_manager = original_manager
    model._validate_anndata = lambda value: value
    model.get_anndata_manager = lambda value: validated_manager
    model._warn_group_indices_auto_inferred = lambda caller: None
    model._process_batches = lambda loader, *args: {"adata_manager": loader.adata_manager}
    model._format_results = lambda results, n_per_group: results

    monkeypatch.setattr(model_mod, "ConcatDataLoader", _FakeConcatDataLoader)

    result = model_cls.get_latent_representation(model, adata=adata, batch_size=2)

    assert result["adata_manager"] is validated_manager
