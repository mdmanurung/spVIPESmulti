"""Internal utilities for single-modal intervention helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy import sparse
from scvi import REGISTRY_KEYS, settings

from spVIPESmulti.dataloaders._concat_dataloader import ConcatDataLoader
from spVIPESmulti.utils import resolve_group_indices_list


def _as_dense(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _ensure_single_modal(model: Any) -> None:
    if bool(getattr(model.module, "is_multimodal", False)):
        raise NotImplementedError("F2 counterfactual interventions support single-modal models only.")


def _validate_group_idx(group_idx: int, n_groups: int) -> int:
    group_idx = int(group_idx)
    if group_idx < 0 or group_idx >= n_groups:
        raise ValueError(f"group_idx={group_idx} is invalid for n_groups={n_groups}.")
    return group_idx


def _prepare_adata(model: Any, adata: Any | None):
    adata = model.adata if adata is None else adata
    return model._validate_anndata(adata) if hasattr(model, "_validate_anndata") else adata


@torch.no_grad()
def _collect_encoded(model: Any, adata: Any | None = None, batch_size: int | None = None) -> dict[str, Any]:
    """Collect deterministic posterior means/scales in group-local order."""
    _ensure_single_modal(model)
    adata = _prepare_adata(model, adata)
    group_indices_list, inferred = resolve_group_indices_list(adata, None)
    if inferred and hasattr(model, "_warn_group_indices_auto_inferred"):
        model._warn_group_indices_auto_inferred("encode_cells")

    n_groups = len(group_indices_list)
    if batch_size is None:
        batch_size = settings.batch_size

    dl = ConcatDataLoader(
        model.adata_manager,
        indices_list=[list(map(int, idxs)) for idxs in group_indices_list],
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
    )

    module = model.module
    was_training = module.training
    module.eval()

    shared: dict[int, list[torch.Tensor]] = {g: [] for g in range(n_groups)}
    private: dict[int, list[torch.Tensor]] = {g: [] for g in range(n_groups)}
    shared_scale: dict[int, list[torch.Tensor]] = {g: [] for g in range(n_groups)}
    private_scale: dict[int, list[torch.Tensor]] = {g: [] for g in range(n_groups)}
    library: dict[int, list[torch.Tensor]] = {g: [] for g in range(n_groups)}
    batch_index: dict[int, list[torch.Tensor]] = {g: [] for g in range(n_groups)}
    original_indices: dict[int, list[torch.Tensor]] = {g: [] for g in range(n_groups)}

    try:
        for tensors_by_group in dl:
            inference_inputs = module._get_inference_input(tensors_by_group)
            outputs = module.inference(**inference_inputs)

            for g in range(n_groups):
                shared[g].append(outputs["poe_stats"][g]["logtheta_loc"].detach().cpu())
                private[g].append(outputs["private_stats"][g]["logtheta_loc"].detach().cpu())
                shared_scale[g].append(outputs["poe_stats"][g]["logtheta_scale"].detach().cpu())
                private_scale[g].append(outputs["private_stats"][g]["logtheta_scale"].detach().cpu())
                library[g].append(outputs["library"][g].detach().cpu())
                batch_index[g].append(inference_inputs["batch_index"][g].detach().cpu())
                idx = inference_inputs["global_indices"][g]
                if idx is None:
                    idx = torch.arange(outputs["poe_stats"][g]["logtheta_loc"].shape[0])
                original_indices[g].append(idx.detach().cpu())
    finally:
        module.train(was_training)

    out: dict[str, Any] = {
        "shared": {},
        "private": {},
        "shared_scale": {},
        "private_scale": {},
        "library": {},
        "batch_index": {},
        "obs_indices": {},
        "obs_names": {},
        "group_indices_list": [np.asarray(idxs, dtype=int) for idxs in group_indices_list],
    }

    for g in range(n_groups):
        n_g = len(group_indices_list[g])
        # ConcatDataLoader exposes group-local indices for some loaders; the
        # public intervention API must report global AnnData observation indices.
        obs_indices = np.asarray(group_indices_list[g], dtype=int)
        order = np.arange(n_g, dtype=int)
        out["shared"][g] = torch.cat(shared[g]).numpy()[:n_g][order].astype(np.float32, copy=False)
        out["private"][g] = torch.cat(private[g]).numpy()[:n_g][order].astype(np.float32, copy=False)
        out["shared_scale"][g] = torch.cat(shared_scale[g]).numpy()[:n_g][order].astype(np.float32, copy=False)
        out["private_scale"][g] = torch.cat(private_scale[g]).numpy()[:n_g][order].astype(np.float32, copy=False)
        out["library"][g] = torch.cat(library[g]).numpy()[:n_g][order].astype(np.float32, copy=False)
        out["batch_index"][g] = torch.cat(batch_index[g]).numpy()[:n_g][order].astype(np.int64, copy=False)
        out["obs_indices"][g] = obs_indices
        out["obs_names"][g] = adata.obs_names[obs_indices].astype(str).tolist()

    return out


def _cells_to_global_indices(adata: Any, cells: Sequence[int | str] | np.ndarray | None) -> np.ndarray | None:
    if cells is None:
        return None
    if isinstance(cells, np.ndarray) and cells.dtype == bool:
        if cells.shape[0] != adata.n_obs:
            raise ValueError("Boolean cells mask must have length adata.n_obs.")
        return np.flatnonzero(cells)

    values = list(cells) if not isinstance(cells, (str, bytes)) else [cells]
    if not values:
        return np.asarray([], dtype=int)
    if isinstance(values[0], str):
        name_to_pos = {str(name): i for i, name in enumerate(adata.obs_names)}
        missing = [name for name in values if str(name) not in name_to_pos]
        if missing:
            raise KeyError(f"Unknown cell names: {missing[:5]}")
        return np.asarray([name_to_pos[str(name)] for name in values], dtype=int)
    return np.asarray(values, dtype=int)


def _select_group_positions(
    adata: Any,
    encoded: dict[str, Any],
    group_idx: int,
    cells: Sequence[int | str] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    group_obs = np.asarray(encoded["obs_indices"][group_idx], dtype=int)
    if cells is None:
        positions = np.arange(group_obs.shape[0], dtype=int)
        return group_obs, positions

    requested = _cells_to_global_indices(adata, cells)
    lookup = {int(obs_idx): i for i, obs_idx in enumerate(group_obs)}
    outside = [int(idx) for idx in requested if int(idx) not in lookup]
    if outside:
        raise ValueError(f"cells contains observations outside group_idx={group_idx}: {outside[:5]}")
    positions = np.asarray([lookup[int(idx)] for idx in requested], dtype=int)
    return np.asarray(requested, dtype=int), positions


def _library_from_adata(adata: Any, cells: Sequence[int | str] | np.ndarray | None, n_cells: int) -> np.ndarray:
    global_idx = _cells_to_global_indices(adata, cells)
    if global_idx is None:
        return np.full((n_cells, 1), float(np.log(1e4)), dtype=np.float32)
    x = _as_dense(adata.X[global_idx])
    lib = np.maximum(np.asarray(x).sum(axis=1, keepdims=True), 1e-8)
    return np.log(lib).astype(np.float32, copy=False)


def _batch_from_adata(model: Any, adata: Any, cells: Sequence[int | str] | np.ndarray | None, n_cells: int) -> np.ndarray:
    if int(getattr(model.module, "n_batch", 1)) <= 1:
        return np.zeros((n_cells, 1), dtype=np.int64)
    global_idx = _cells_to_global_indices(adata, cells)
    if global_idx is None:
        return np.zeros((n_cells, 1), dtype=np.int64)
    batch = model.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
    return np.asarray(batch)[global_idx].reshape(-1, 1).astype(np.int64, copy=False)


def _get_group_decoder(model: Any, group_idx: int):
    _ensure_single_modal(model)
    n_groups = len(model.module.decoders)
    group_idx = _validate_group_idx(group_idx, n_groups)
    return model.module.decoders[group_idx]


def _decode_arrays(
    model: Any,
    z_shared: np.ndarray,
    z_private: np.ndarray,
    library: np.ndarray,
    batch_index: np.ndarray,
    group_idx: int,
) -> dict[str, np.ndarray]:
    decoder = _get_group_decoder(model, group_idx)
    module = model.module
    device = next(module.parameters()).device
    z_shared_t = torch.as_tensor(z_shared, dtype=torch.float32, device=device)
    z_private_t = torch.as_tensor(z_private, dtype=torch.float32, device=device)
    library_t = torch.as_tensor(library, dtype=torch.float32, device=device)
    batch_t = torch.as_tensor(batch_index, dtype=torch.long, device=device)
    cat_args = (batch_t,)

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            px_scale_private, px_scale_shared, px_rate_private, px_rate_shared, px_mixing, px_scale = decoder(
                module.dispersion,
                z_private_t,
                z_shared_t,
                library_t,
                *cat_args,
            )
            expected = torch.exp(library_t) * px_scale
    finally:
        module.train(was_training)

    return {
        "X": expected.detach().cpu().numpy().astype(np.float32, copy=False),
        "px_scale": px_scale.detach().cpu().numpy().astype(np.float32, copy=False),
        "px_scale_private": px_scale_private.detach().cpu().numpy().astype(np.float32, copy=False),
        "px_scale_shared": px_scale_shared.detach().cpu().numpy().astype(np.float32, copy=False),
        "px_rate_private": px_rate_private.detach().cpu().numpy().astype(np.float32, copy=False),
        "px_rate_shared": px_rate_shared.detach().cpu().numpy().astype(np.float32, copy=False),
        "px_mixing": px_mixing.detach().cpu().numpy().astype(np.float32, copy=False),
    }


def _target_var_names(model: Any, adata: Any, group_idx: int) -> list[str]:
    var_idx = np.asarray(model.module.groups_var_indices[group_idx], dtype=int)
    return adata.var_names[var_idx].astype(str).tolist()
