"""Low-level latent arithmetic operators used by counterfactual helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


def _to_backend(value: Any, like: ArrayLike) -> ArrayLike:
    if torch.is_tensor(like):
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return np.asarray(value, dtype=np.asarray(like).dtype)


def _latent_dim(x: ArrayLike) -> int:
    if x.ndim == 0:
        raise ValueError("Latent input must have at least one dimension.")
    return int(x.shape[-1])


def _validate_direction(z: ArrayLike, direction: Any) -> ArrayLike:
    d = _to_backend(direction, z)
    if d.ndim == 0:
        raise ValueError("direction must be a vector or array with a latent dimension.")
    if int(d.shape[-1]) != _latent_dim(z):
        raise ValueError(
            f"direction latent dimension {int(d.shape[-1])} does not match input dimension {_latent_dim(z)}."
        )
    return d


def condition_centroid_shift(z: ArrayLike, direction: Any, alpha: float = 1.0) -> ArrayLike:
    """Apply a scGen-style centroid shift to latent coordinates."""
    d = _validate_direction(z, direction)
    return z + float(alpha) * d


def latent_arithmetic(z: ArrayLike, direction: Any, weight: float = 1.0) -> ArrayLike:
    """Apply a weighted latent direction."""
    d = _validate_direction(z, direction)
    return z + float(weight) * d


def latent_interpolation(z_src: ArrayLike, z_tgt: Any, alpha: float) -> ArrayLike:
    """Interpolate linearly between source and target latent coordinates."""
    target = _validate_direction(z_src, z_tgt)
    return (1.0 - float(alpha)) * z_src + float(alpha) * target


def latent_replacement(z: ArrayLike, dimension: int, value: float) -> ArrayLike:
    """Replace one latent dimension with a diagnostic scalar value."""
    dim = int(dimension)
    n_dim = _latent_dim(z)
    if dim < 0 or dim >= n_dim:
        raise IndexError(f"dimension={dim} is out of bounds for latent dimension {n_dim}.")
    out = z.clone() if torch.is_tensor(z) else np.array(z, copy=True)
    out[..., dim] = value
    return out
