"""Donor and condition-aware counterfactual protocol helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .counterfactual import CounterfactualResult, decode_counterfactual
from .latent_operators import condition_centroid_shift
from .utils import _collect_encoded, _prepare_adata, _select_group_positions, _validate_group_idx


def _registered_obs_key(model: Any, registry_key: str, setup_arg: str) -> str:
    if registry_key not in model.adata_manager.data_registry:
        raise ValueError(
            f"{setup_arg} is required for this counterfactual protocol; "
            f"call setup_anndata(..., {setup_arg}='...') first."
        )
    return str(model.adata_manager.get_state_registry(registry_key).original_key)


def _default_source_group(target_group_idx: int, n_groups: int) -> int:
    if n_groups <= 1:
        return target_group_idx
    return (target_group_idx + 1) % n_groups


def _source_positions(
    encoded: dict[str, Any],
    source_group_idx: int,
    exclude_obs: np.ndarray | None = None,
) -> np.ndarray:
    source_obs = np.asarray(encoded["obs_indices"][source_group_idx], dtype=int)
    positions = np.arange(source_obs.shape[0], dtype=int)
    if exclude_obs is not None and source_obs.size > 1:
        keep = ~np.isin(source_obs, np.asarray(exclude_obs, dtype=int))
        if np.any(keep):
            positions = positions[keep]
    if positions.size == 0:
        raise ValueError("No source cells are available for private-latent swap.")
    return positions


def _choose_positions(candidates: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    replace = candidates.shape[0] < n
    return rng.choice(candidates, size=n, replace=replace).astype(int, copy=False)


def _decode_private_swap(
    model: Any,
    adata: Any,
    encoded: dict[str, Any],
    *,
    target_group_idx: int,
    source_group_idx: int,
    target_obs: np.ndarray,
    target_pos: np.ndarray,
    source_pos: np.ndarray,
    protocol: str,
    extra_info: dict[str, Any] | None = None,
    include_uncertainty: bool = True,
) -> CounterfactualResult:
    result = decode_counterfactual(
        model,
        encoded["shared"][target_group_idx][target_pos],
        encoded["private"][source_group_idx][source_pos],
        group_idx=target_group_idx,
        adata=adata,
        cells=target_obs,
        library=encoded["library"][target_group_idx][target_pos],
        include_uncertainty=include_uncertainty,
    )
    source_obs = np.asarray(encoded["obs_indices"][source_group_idx], dtype=int)[source_pos]
    result.info.update(
        {
            "protocol": protocol,
            "source_group_idx": source_group_idx,
            "target_group_idx": target_group_idx,
            "source_obs_indices": source_obs,
            "target_obs_indices": np.asarray(target_obs, dtype=int),
            "z_shared_source": "target",
            "z_private_source": "source",
        }
    )
    if extra_info:
        result.info.update(extra_info)
    return result


def private_swap_unmatched(
    model: Any,
    adata: Any | None,
    cells: Sequence[int | str] | np.ndarray | None,
    group_idx: int,
    source_group_idx: int | None = None,
    seed: int = 0,
    include_uncertainty: bool = True,
) -> CounterfactualResult:
    """P1: swap private latents from unmatched source cells into target cells."""
    adata = _prepare_adata(model, adata)
    encoded = _collect_encoded(model, adata=adata)
    n_groups = len(encoded["group_indices_list"])
    target_group_idx = _validate_group_idx(group_idx, n_groups)
    source_group_idx = _validate_group_idx(
        _default_source_group(target_group_idx, n_groups) if source_group_idx is None else source_group_idx,
        n_groups,
    )
    target_obs, target_pos = _select_group_positions(adata, encoded, target_group_idx, cells)
    candidates = _source_positions(
        encoded,
        source_group_idx,
        exclude_obs=target_obs if source_group_idx == target_group_idx else None,
    )
    source_pos = _choose_positions(candidates, target_pos.shape[0], seed)
    return _decode_private_swap(
        model,
        adata,
        encoded,
        target_group_idx=target_group_idx,
        source_group_idx=source_group_idx,
        target_obs=target_obs,
        target_pos=target_pos,
        source_pos=source_pos,
        protocol="private_swap_unmatched",
        include_uncertainty=include_uncertainty,
    )


def private_swap_label_matched(
    model: Any,
    adata: Any | None,
    cells: Sequence[int | str] | np.ndarray | None,
    group_idx: int,
    source_group_idx: int | None = None,
    seed: int = 0,
    include_uncertainty: bool = True,
) -> CounterfactualResult:
    """P2: swap private latents from source cells with matching labels."""
    adata = _prepare_adata(model, adata)
    label_key = _registered_obs_key(model, "labels", "label_key")
    encoded = _collect_encoded(model, adata=adata)
    n_groups = len(encoded["group_indices_list"])
    target_group_idx = _validate_group_idx(group_idx, n_groups)
    source_group_idx = _validate_group_idx(
        _default_source_group(target_group_idx, n_groups) if source_group_idx is None else source_group_idx,
        n_groups,
    )
    target_obs, target_pos = _select_group_positions(adata, encoded, target_group_idx, cells)
    source_candidates = _source_positions(encoded, source_group_idx)
    labels = adata.obs[label_key].astype(str).to_numpy()
    source_obs = np.asarray(encoded["obs_indices"][source_group_idx], dtype=int)

    source_pos = []
    rng = np.random.default_rng(seed)
    for obs_idx in target_obs:
        matches = source_candidates[labels[source_obs[source_candidates]] == labels[int(obs_idx)]]
        if matches.size == 0:
            raise ValueError(f"No label-matched source cells found for label={labels[int(obs_idx)]!r}.")
        source_pos.append(int(rng.choice(matches)))

    return _decode_private_swap(
        model,
        adata,
        encoded,
        target_group_idx=target_group_idx,
        source_group_idx=source_group_idx,
        target_obs=target_obs,
        target_pos=target_pos,
        source_pos=np.asarray(source_pos, dtype=int),
        protocol="private_swap_label_matched",
        extra_info={"label_key": label_key},
        include_uncertainty=include_uncertainty,
    )


def private_swap_stratified(
    model: Any,
    adata: Any | None,
    cells: Sequence[int | str] | np.ndarray | None,
    group_idx: int,
    source_group_idx: int | None = None,
    timepoint_key: str | None = None,
    seed: int = 0,
    include_uncertainty: bool = True,
) -> CounterfactualResult:
    """P3: prefer label + donor/timepoint private swaps with deterministic fallbacks."""
    adata = _prepare_adata(model, adata)
    label_key = _registered_obs_key(model, "labels", "label_key")
    donor_key = _registered_obs_key(model, "donor", "donor_key")
    if timepoint_key is not None and timepoint_key not in adata.obs:
        raise ValueError(f"timepoint_key={timepoint_key!r} is not present in adata.obs.")

    encoded = _collect_encoded(model, adata=adata)
    n_groups = len(encoded["group_indices_list"])
    target_group_idx = _validate_group_idx(group_idx, n_groups)
    source_group_idx = _validate_group_idx(
        _default_source_group(target_group_idx, n_groups) if source_group_idx is None else source_group_idx,
        n_groups,
    )
    target_obs, target_pos = _select_group_positions(adata, encoded, target_group_idx, cells)
    source_candidates = _source_positions(encoded, source_group_idx)
    source_obs = np.asarray(encoded["obs_indices"][source_group_idx], dtype=int)
    labels = adata.obs[label_key].astype(str).to_numpy()
    donors = adata.obs[donor_key].astype(str).to_numpy()
    timepoints = adata.obs[timepoint_key].astype(str).to_numpy() if timepoint_key is not None else None

    fallback_counts = {
        "label_donor_timepoint": 0,
        "label_donor": 0,
        "label": 0,
        "unmatched": 0,
    }
    source_pos = []
    match_tiers = []
    rng = np.random.default_rng(seed)
    for obs_idx in target_obs:
        obs_idx = int(obs_idx)
        label_mask = labels[source_obs[source_candidates]] == labels[obs_idx]
        donor_mask = donors[source_obs[source_candidates]] == donors[obs_idx]
        strict = source_candidates[label_mask & donor_mask]
        if timepoints is not None:
            time_mask = timepoints[source_obs[source_candidates]] == timepoints[obs_idx]
            strict = source_candidates[label_mask & donor_mask & time_mask]
        if strict.size:
            tier = "label_donor_timepoint" if timepoints is not None else "label_donor"
            matches = strict
        else:
            label_donor = source_candidates[label_mask & donor_mask]
            if label_donor.size:
                tier = "label_donor"
                matches = label_donor
            else:
                label_only = source_candidates[label_mask]
                if label_only.size:
                    tier = "label"
                    matches = label_only
                else:
                    tier = "unmatched"
                    matches = source_candidates
        fallback_counts[tier] += 1
        match_tiers.append(tier)
        source_pos.append(int(rng.choice(matches)))

    return _decode_private_swap(
        model,
        adata,
        encoded,
        target_group_idx=target_group_idx,
        source_group_idx=source_group_idx,
        target_obs=target_obs,
        target_pos=target_pos,
        source_pos=np.asarray(source_pos, dtype=int),
        protocol="private_swap_stratified",
        extra_info={
            "label_key": label_key,
            "donor_key": donor_key,
            "timepoint_key": timepoint_key,
            "fallback_counts": fallback_counts,
            "match_tiers": match_tiers,
        },
        include_uncertainty=include_uncertainty,
    )


def donor_condition_shift(
    model: Any,
    adata: Any | None,
    cells: Sequence[int | str] | np.ndarray | None,
    condition_from: str,
    condition_to: str,
    group_idx: int,
    target_group_idx: int | None = None,
    donor_aware: bool = True,
    include_uncertainty: bool = True,
) -> CounterfactualResult:
    """Apply a condition centroid shift, using within-donor deltas when available."""
    adata = _prepare_adata(model, adata)
    condition_key = _registered_obs_key(model, "condition", "condition_key")
    donor_key = _registered_obs_key(model, "donor", "donor_key") if donor_aware else None
    encoded = _collect_encoded(model, adata=adata)
    n_groups = len(encoded["group_indices_list"])
    source_group_idx = _validate_group_idx(group_idx, n_groups)
    target_group_idx = _validate_group_idx(source_group_idx if target_group_idx is None else target_group_idx, n_groups)
    target_obs, target_pos = _select_group_positions(adata, encoded, source_group_idx, cells)

    z_shared_all = np.zeros((adata.n_obs, encoded["shared"][0].shape[1]), dtype=np.float32)
    for group, arr in encoded["shared"].items():
        z_shared_all[np.asarray(encoded["obs_indices"][group], dtype=int)] = arr

    conditions = adata.obs[condition_key].astype(str).to_numpy()
    from_mask_global = conditions == str(condition_from)
    to_mask_global = conditions == str(condition_to)
    if not np.any(from_mask_global) or not np.any(to_mask_global):
        raise ValueError(f"Both condition_from={condition_from!r} and condition_to={condition_to!r} must be present.")
    global_direction = z_shared_all[to_mask_global].mean(axis=0) - z_shared_all[from_mask_global].mean(axis=0)

    donors = adata.obs[donor_key].astype(str).to_numpy() if donor_key is not None else None
    directions = []
    fallback_counts = {"donor_specific": 0, "global": 0}
    for obs_idx in target_obs:
        if donors is not None:
            donor_mask = donors == donors[int(obs_idx)]
            from_mask = from_mask_global & donor_mask
            to_mask = to_mask_global & donor_mask
            if np.any(from_mask) and np.any(to_mask):
                directions.append(z_shared_all[to_mask].mean(axis=0) - z_shared_all[from_mask].mean(axis=0))
                fallback_counts["donor_specific"] += 1
                continue
        directions.append(global_direction)
        fallback_counts["global"] += 1

    direction_arr = np.asarray(directions, dtype=np.float32)
    z_shared = condition_centroid_shift(encoded["shared"][source_group_idx][target_pos], direction_arr, alpha=1.0)
    result = decode_counterfactual(
        model,
        z_shared,
        encoded["private"][source_group_idx][target_pos],
        group_idx=target_group_idx,
        adata=adata,
        cells=target_obs,
        library=encoded["library"][source_group_idx][target_pos],
        include_uncertainty=include_uncertainty,
    )
    result.info.update(
        {
            "protocol": "donor_condition_shift",
            "condition_key": condition_key,
            "donor_key": donor_key,
            "condition_from": condition_from,
            "condition_to": condition_to,
            "source_group_idx": source_group_idx,
            "target_group_idx": target_group_idx,
            "source_obs_indices": np.asarray(target_obs, dtype=int),
            "target_obs_indices": np.asarray(target_obs, dtype=int),
            "direction": direction_arr,
            "fallback_counts": fallback_counts,
        }
    )
    return result


__all__ = [
    "donor_condition_shift",
    "private_swap_label_matched",
    "private_swap_stratified",
    "private_swap_unmatched",
]
