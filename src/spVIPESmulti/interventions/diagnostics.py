"""Lightweight latent diagnostics for intervention safety checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import _collect_encoded, _prepare_adata


def _nearest_centroid_score(x: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    unique = np.unique(labels)
    if unique.shape[0] < 2 or x.shape[0] == 0:
        return 0.0
    centroids = np.stack([x[labels == label].mean(axis=0) for label in unique], axis=0)
    dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    pred = unique[np.argmin(dist, axis=1)]
    return float(np.mean(pred == labels))


def _global_latent(encoded: dict[str, Any], latent_type: str, n_obs: int) -> np.ndarray:
    if latent_type not in {"shared", "private"}:
        raise ValueError("latent_type must be 'shared' or 'private'.")
    dim = encoded[latent_type][0].shape[1]
    out = np.zeros((n_obs, dim), dtype=np.float32)
    for g, arr in encoded[latent_type].items():
        out[encoded["obs_indices"][g]] = arr
    return out


def leakage_score(
    model: Any,
    adata: Any | None,
    group_key: str,
    label_key: str | None = None,
    latent_type: str = "shared",
) -> float:
    """Estimate how easily ``group_key`` can be recovered from a latent space."""
    adata = _prepare_adata(model, adata)
    encoded = _collect_encoded(model, adata=adata)
    latent = _global_latent(encoded, latent_type, adata.n_obs)
    groups = adata.obs[group_key].astype(str).to_numpy()
    if label_key is None:
        return _nearest_centroid_score(latent, groups)

    labels = adata.obs[label_key].astype(str).to_numpy()
    scores: list[float] = []
    weights: list[int] = []
    for label in np.unique(labels):
        mask = labels == label
        if np.unique(groups[mask]).shape[0] >= 2:
            scores.append(_nearest_centroid_score(latent[mask], groups[mask]))
            weights.append(int(mask.sum()))
    if not scores:
        return 0.0
    return float(np.average(scores, weights=weights))


def condition_separability(model: Any, adata: Any | None, label_key: str) -> float:
    """Nearest-centroid separability of an obs label in shared latent space."""
    adata = _prepare_adata(model, adata)
    encoded = _collect_encoded(model, adata=adata)
    latent = _global_latent(encoded, "shared", adata.n_obs)
    labels = adata.obs[label_key].astype(str).to_numpy()
    return _nearest_centroid_score(latent, labels)


def latent_variance_utilization(
    model_or_encoded: Any,
    adata: Any | None = None,
    latent_type: str = "shared",
    threshold: float = 1e-4,
) -> dict[str, Any]:
    """Report empirical latent variance and active dimensions."""
    if isinstance(model_or_encoded, dict) and latent_type in model_or_encoded:
        encoded = model_or_encoded
        if adata is None:
            values = np.concatenate(list(encoded[latent_type].values()), axis=0)
        else:
            values = _global_latent(encoded, latent_type, adata.n_obs)
    else:
        adata = _prepare_adata(model_or_encoded, adata)
        encoded = _collect_encoded(model_or_encoded, adata=adata)
        values = _global_latent(encoded, latent_type, adata.n_obs)
    variance = np.var(values, axis=0).astype(np.float32, copy=False)
    active = variance > float(threshold)
    return {
        "latent_type": latent_type,
        "threshold": float(threshold),
        "variances": variance,
        "active_dims": int(active.sum()),
        "total_dims": int(variance.shape[0]),
        "active_fraction": float(active.mean()) if variance.size else 0.0,
    }


def integration_report(
    model: Any,
    adata: Any | None,
    group_key: str,
    label_key: str | None = None,
) -> dict[str, Any]:
    """Return compact intervention-readiness diagnostics."""
    report = {
        "leakage_shared": leakage_score(model, adata, group_key=group_key, label_key=label_key, latent_type="shared"),
        "leakage_private": leakage_score(model, adata, group_key=group_key, label_key=label_key, latent_type="private"),
        "shared_variance": latent_variance_utilization(model, adata, latent_type="shared"),
        "private_variance": latent_variance_utilization(model, adata, latent_type="private"),
    }
    if label_key is not None:
        report["condition_separability"] = condition_separability(model, adata, label_key=label_key)
    return report
