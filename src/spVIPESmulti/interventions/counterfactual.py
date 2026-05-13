"""Encode, edit, decode, and diagnose safe single-modal counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import warnings

import numpy as np

from .latent_operators import (
    condition_centroid_shift,
    latent_arithmetic,
    latent_interpolation,
    latent_replacement,
)
from .utils import (
    _batch_from_adata,
    _collect_encoded,
    _decode_arrays,
    _library_from_adata,
    _prepare_adata,
    _select_group_positions,
    _target_var_names,
    _validate_group_idx,
)


@dataclass
class CounterfactualResult:
    """Container returned by counterfactual prediction helpers."""

    X: np.ndarray
    uncertainty: np.ndarray | None
    info: dict[str, Any]


def encode_cells(
    model: Any,
    adata: Any | None = None,
    group_idx: int | None = None,
    include_variance: bool = True,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Encode cells as deterministic posterior means for single-modal interventions."""
    encoded = _collect_encoded(model, adata=adata, batch_size=batch_size)
    n_groups = len(encoded["group_indices_list"])
    if group_idx is not None:
        group_idx = _validate_group_idx(group_idx, n_groups)
        keys = ["shared", "private", "library", "batch_index", "obs_indices", "obs_names"]
        if include_variance:
            keys.extend(["shared_scale", "private_scale"])
        filtered = {key: {group_idx: encoded[key][group_idx]} for key in keys}
        filtered["group_indices_list"] = [encoded["group_indices_list"][group_idx]]
        return filtered
    if not include_variance:
        encoded = dict(encoded)
        encoded.pop("shared_scale", None)
        encoded.pop("private_scale", None)
    return encoded


def _as_2d_float(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D latent array.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _uncertainty_samples(
    model: Any,
    z_shared: np.ndarray,
    z_private: np.ndarray,
    library: np.ndarray,
    batch_index: np.ndarray,
    group_idx: int,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    if n_samples <= 1:
        return np.zeros_like(_decode_arrays(model, z_shared, z_private, library, batch_index, group_idx)["X"])
    rng = np.random.default_rng(seed)
    scale = 0.025 * (1.0 + np.linalg.norm(z_shared, axis=1, keepdims=True) / max(z_shared.shape[1], 1))
    draws = []
    for _ in range(int(n_samples)):
        z_draw = z_shared + rng.normal(0.0, scale, size=z_shared.shape).astype(np.float32)
        draws.append(_decode_arrays(model, z_draw, z_private, library, batch_index, group_idx)["X"])
    return np.std(np.stack(draws, axis=0), axis=0).astype(np.float32, copy=False)


def decode_counterfactual(
    model: Any,
    z_shared: Any,
    z_private: Any,
    group_idx: int,
    adata: Any | None,
    cells: Any | None = None,
    library: Any | None = None,
    include_uncertainty: bool = True,
    n_uncertainty_samples: int = 8,
    seed: int = 0,
    return_components: bool = False,
    batch_size: int = 512,
) -> CounterfactualResult:
    """Decode edited latents through a target group decoder."""
    del batch_size
    adata = _prepare_adata(model, adata)
    n_groups = len(model.module.decoders)
    group_idx = _validate_group_idx(group_idx, n_groups)
    z_shared_arr = _as_2d_float(z_shared, "z_shared")
    z_private_arr = _as_2d_float(z_private, "z_private")
    if z_shared_arr.shape[0] != z_private_arr.shape[0]:
        raise ValueError("z_shared and z_private must contain the same number of cells.")
    n_cells = z_shared_arr.shape[0]

    library_arr = (
        np.asarray(library, dtype=np.float32).reshape(n_cells, 1)
        if library is not None
        else _library_from_adata(adata, cells, n_cells)
    )
    batch_arr = _batch_from_adata(model, adata, cells, n_cells)
    decoded = _decode_arrays(model, z_shared_arr, z_private_arr, library_arr, batch_arr, group_idx)
    uncertainty = None
    if include_uncertainty:
        uncertainty = _uncertainty_samples(
            model,
            z_shared_arr,
            z_private_arr,
            library_arr,
            batch_arr,
            group_idx,
            n_uncertainty_samples,
            seed,
        )

    info: dict[str, Any] = {
        "group_idx": group_idx,
        "var_names": _target_var_names(model, adata, group_idx),
        "library": library_arr.ravel(),
    }
    if return_components:
        info["components"] = {k: v for k, v in decoded.items() if k != "X"}
    return CounterfactualResult(X=decoded["X"], uncertainty=uncertainty, info=info)


def edit_latent(
    z_shared: Any,
    z_private: Any,
    intervention: str,
    direction: Any | None = None,
    target: Any | None = None,
    alpha: float = 1.0,
    dimension: int | None = None,
    value: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a low-level edit and return ``(z_shared, z_private)`` arrays."""
    z_shared_arr = _as_2d_float(z_shared, "z_shared")
    z_private_arr = _as_2d_float(z_private, "z_private")
    if intervention == "centroid_shift":
        if direction is None:
            raise ValueError("direction is required for intervention='centroid_shift'.")
        return np.asarray(condition_centroid_shift(z_shared_arr, direction, alpha), dtype=np.float32), z_private_arr
    if intervention == "arithmetic":
        if direction is None:
            raise ValueError("direction is required for intervention='arithmetic'.")
        return np.asarray(latent_arithmetic(z_shared_arr, direction, alpha), dtype=np.float32), z_private_arr
    if intervention == "interpolation":
        if target is None:
            raise ValueError("target is required for intervention='interpolation'.")
        return np.asarray(latent_interpolation(z_shared_arr, target, alpha), dtype=np.float32), z_private_arr
    if intervention == "replacement":
        if dimension is None or value is None:
            raise ValueError("dimension and value are required for intervention='replacement'.")
        return np.asarray(latent_replacement(z_shared_arr, dimension, value), dtype=np.float32), z_private_arr
    raise ValueError(f"Unknown intervention={intervention!r}.")


def _mahalanobis(values: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    center = reference.mean(axis=0)
    cov = np.cov(reference, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]])
    cov = cov + np.eye(cov.shape[0]) * 1e-4
    inv_cov = np.linalg.pinv(cov)
    diff = values - center
    dist = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, inv_cov, diff), 0.0))
    ref_diff = reference - center
    ref_dist = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", ref_diff, inv_cov, ref_diff), 0.0))
    threshold = float(np.nanpercentile(ref_dist, 95)) if ref_dist.size else float("inf")
    return dist.astype(np.float32, copy=False), threshold


def _attach_ood_info(
    result: CounterfactualResult,
    z_edited: np.ndarray,
    reference_z: np.ndarray,
    library: np.ndarray,
    reject_ood: bool | Literal["raise"],
) -> CounterfactualResult:
    mahal, mahal_threshold = _mahalanobis(z_edited, reference_z)
    decoded_library = np.maximum(result.X.sum(axis=1), 1e-8)
    source_library = np.maximum(np.exp(np.asarray(library, dtype=np.float32).reshape(-1)), 1e-8)
    ratio = decoded_library / source_library
    low_likelihood_proxy = -np.mean(np.log1p(np.maximum(result.X, 0.0)), axis=1)
    low_threshold = float(np.nanpercentile(low_likelihood_proxy, 5)) if low_likelihood_proxy.size else float("nan")
    flags = {
        "mahalanobis": mahal > mahal_threshold,
        "library_ratio": (ratio < 0.5) | (ratio > 2.0),
        "low_likelihood_proxy": low_likelihood_proxy < low_threshold,
    }
    rejected = np.logical_or.reduce(list(flags.values())) if flags else np.zeros(result.X.shape[0], dtype=bool)
    result.info["ood_flags"] = flags
    result.info["rejected_mask"] = rejected
    result.info["ood_thresholds"] = {
        "mahalanobis": mahal_threshold,
        "library_ratio_min": 0.5,
        "library_ratio_max": 2.0,
        "low_likelihood_proxy": low_threshold,
    }
    result.info["ood_scores"] = {
        "mahalanobis": mahal,
        "library_ratio": ratio.astype(np.float32, copy=False),
        "low_likelihood_proxy": low_likelihood_proxy.astype(np.float32, copy=False),
    }
    if reject_ood == "raise" and bool(np.any(rejected)):
        raise ValueError("Counterfactual edit produced OOD cells; inspect result.info['ood_flags'].")
    return result


def _warn_if_leaky(model: Any, adata: Any) -> None:
    try:
        from .diagnostics import leakage_score

        score = leakage_score(model, adata, group_key="groups", latent_type="shared")
    except Exception:
        return
    if score > 0.4:
        warnings.warn(
            f"Shared latent leakage_score={score:.3f} exceeds 0.4; counterfactuals are diagnostic only.",
            UserWarning,
            stacklevel=3,
        )


def predict_counterfactual(
    model: Any,
    adata: Any | None,
    cells: Any | None = None,
    group_idx: int = 0,
    intervention: str = "centroid_shift",
    direction: Any | None = None,
    target_cells: Any | None = None,
    alpha: float = 1.0,
    dimension: int | None = None,
    value: float | None = None,
    return_uncertainty: bool = True,
    reject_ood: bool | Literal["raise"] = True,
) -> CounterfactualResult:
    """Encode selected cells, edit shared latent coordinates, and decode them."""
    adata = _prepare_adata(model, adata)
    encoded = _collect_encoded(model, adata=adata)
    n_groups = len(encoded["group_indices_list"])
    group_idx = _validate_group_idx(group_idx, n_groups)
    global_idx, pos = _select_group_positions(adata, encoded, group_idx, cells)
    z_shared = encoded["shared"][group_idx][pos]
    z_private = encoded["private"][group_idx][pos]
    library = encoded["library"][group_idx][pos]

    target = None
    if intervention == "interpolation":
        _, target_pos = _select_group_positions(adata, encoded, group_idx, target_cells)
        target = encoded["shared"][group_idx][target_pos]
        if target.shape[0] != z_shared.shape[0]:
            target = np.repeat(target.mean(axis=0, keepdims=True), z_shared.shape[0], axis=0)

    z_edited, z_private_edited = edit_latent(
        z_shared,
        z_private,
        intervention=intervention,
        direction=direction,
        target=target,
        alpha=alpha,
        dimension=dimension,
        value=value,
    )
    result = decode_counterfactual(
        model,
        z_edited,
        z_private_edited,
        group_idx=group_idx,
        adata=adata,
        cells=global_idx,
        library=library,
        include_uncertainty=return_uncertainty,
    )
    result.info.update(
        {
            "intervention": intervention,
            "alpha": alpha,
            "source_group_idx": group_idx,
            "target_group_idx": group_idx,
            "source_obs_indices": global_idx,
        }
    )
    if reject_ood:
        result = _attach_ood_info(result, z_edited, encoded["shared"][group_idx], library, reject_ood)
    else:
        result.info["ood_flags"] = {}
        result.info["rejected_mask"] = np.zeros(result.X.shape[0], dtype=bool)
    _warn_if_leaky(model, adata)
    return result


def _condition_key(model: Any) -> str:
    if "condition" not in model.adata_manager.data_registry:
        raise ValueError(
            "condition_key is required for transfer_condition(); call setup_anndata(..., condition_key='...') first."
        )
    return str(model.adata_manager.get_state_registry("condition").original_key)


def transfer_condition(
    model: Any,
    adata: Any | None,
    cells: Any,
    condition_from: str,
    condition_to: str,
    group_src: int,
    group_dst: int,
    latent_type: str = "shared",
) -> CounterfactualResult:
    """Transfer cells between condition centroids and decode through a target group."""
    adata = _prepare_adata(model, adata)
    condition_key = _condition_key(model)
    encoded = _collect_encoded(model, adata=adata)
    n_groups = len(encoded["group_indices_list"])
    group_src = _validate_group_idx(group_src, n_groups)
    group_dst = _validate_group_idx(group_dst, n_groups)
    if latent_type not in {"shared", "private"}:
        raise ValueError("latent_type must be 'shared' or 'private'.")

    conditions = adata.obs[condition_key].astype(str).to_numpy()
    latent_by_obs = np.zeros((adata.n_obs, encoded[latent_type][0].shape[1]), dtype=np.float32)
    for g in range(n_groups):
        latent_by_obs[encoded["obs_indices"][g]] = encoded[latent_type][g]

    from_mask = conditions == str(condition_from)
    to_mask = conditions == str(condition_to)
    if not np.any(from_mask) or not np.any(to_mask):
        raise ValueError(f"Both condition_from={condition_from!r} and condition_to={condition_to!r} must be present.")
    direction = latent_by_obs[to_mask].mean(axis=0) - latent_by_obs[from_mask].mean(axis=0)

    global_idx, pos = _select_group_positions(adata, encoded, group_src, cells)
    z_shared = encoded["shared"][group_src][pos]
    z_private = encoded["private"][group_src][pos]
    library = encoded["library"][group_src][pos]
    if latent_type == "shared":
        z_shared = condition_centroid_shift(z_shared, direction, alpha=1.0)
    else:
        z_private = condition_centroid_shift(z_private, direction, alpha=1.0)

    result = decode_counterfactual(
        model,
        z_shared,
        z_private,
        group_idx=group_dst,
        adata=adata,
        cells=global_idx,
        library=library,
        include_uncertainty=True,
    )
    result.info.update(
        {
            "intervention": "transfer_condition",
            "condition_key": condition_key,
            "condition_from": condition_from,
            "condition_to": condition_to,
            "source_group_idx": group_src,
            "target_group_idx": group_dst,
            "source_obs_indices": global_idx,
            "direction": direction.astype(np.float32, copy=False),
            "latent_type": latent_type,
        }
    )
    result = _attach_ood_info(result, z_shared, encoded["shared"][group_dst], library, True)
    _warn_if_leaky(model, adata)
    return result
