"""Internal CellDISECT-style metric helpers for F10a audits."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _as_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected a 1D or 2D array.")
    return arr


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("Pearson inputs must have matching shapes.")
    if a.size == 0:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def counterfactual_pearson(x_pred: Any, x_true: Any) -> float:
    """Pearson correlation between predicted and true mean expression."""
    pred = _as_2d(x_pred).mean(axis=0)
    true = _as_2d(x_true).mean(axis=0)
    return _pearson(pred, true)


def delta_pearson(x_ctrl: Any, x_true: Any, x_pred: Any) -> float:
    """Pearson correlation between predicted and true expression deltas."""
    ctrl = _as_2d(x_ctrl).mean(axis=0)
    true_delta = _as_2d(x_true).mean(axis=0) - ctrl
    pred_delta = _as_2d(x_pred).mean(axis=0) - ctrl
    return _pearson(pred_delta, true_delta)


def select_top_de_genes(x_ctrl: Any, x_true: Any, n_top: int = 20) -> np.ndarray:
    """Select top genes by absolute true control-to-target mean delta."""
    ctrl = _as_2d(x_ctrl).mean(axis=0)
    true = _as_2d(x_true).mean(axis=0)
    n_top = min(int(n_top), ctrl.shape[0])
    return np.argsort(np.abs(true - ctrl))[-n_top:][::-1]


def top_de_cosine(x_ctrl: Any, x_true: Any, x_pred: Any, n_top: int = 20) -> float:
    """Cosine similarity between predicted and true deltas on top-DE genes."""
    top = select_top_de_genes(x_ctrl, x_true, n_top=n_top)
    ctrl = _as_2d(x_ctrl).mean(axis=0)
    true_delta = _as_2d(x_true).mean(axis=0)[top] - ctrl[top]
    pred_delta = _as_2d(x_pred).mean(axis=0)[top] - ctrl[top]
    denom = np.linalg.norm(true_delta) * np.linalg.norm(pred_delta)
    if denom == 0:
        return 0.0
    return float(np.dot(true_delta, pred_delta) / denom)


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    if a.size == 0 or b.size == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, max(a.size, b.size))
    aq = np.quantile(a, q)
    bq = np.quantile(b, q)
    return float(np.mean(np.abs(aq - bq)))


def wasserstein_gene_marginals(x_pred: Any, x_true: Any, top_idx: Any | None = None) -> dict[str, Any]:
    """Compute per-gene empirical Wasserstein distances."""
    pred = _as_2d(x_pred)
    true = _as_2d(x_true)
    if pred.shape[1] != true.shape[1]:
        raise ValueError("x_pred and x_true must have the same number of genes.")
    per_gene = np.asarray([_wasserstein_1d(pred[:, j], true[:, j]) for j in range(pred.shape[1])], dtype=np.float64)
    out: dict[str, Any] = {"per_gene": per_gene, "mean_all": float(np.nanmean(per_gene))}
    if top_idx is not None:
        top_idx = np.asarray(top_idx, dtype=int)
        out["mean_top"] = float(np.nanmean(per_gene[top_idx]))
    return out


def _nearest_centroid_accuracy(x: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(str)
    unique = np.unique(labels)
    if unique.shape[0] < 2:
        return 0.0
    centroids = np.stack([x[labels == label].mean(axis=0) for label in unique], axis=0)
    dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    pred = unique[np.argmin(dist, axis=1)]
    return float(np.mean(pred == labels))


def classifier_accuracy_gap(z_i: Any, z_minus_i: Any, labels: Any) -> dict[str, float]:
    """CellDISECT-style CAG using nearest-centroid probe accuracy."""
    acc_i = _nearest_centroid_accuracy(_as_2d(z_i), np.asarray(labels))
    acc_minus = _nearest_centroid_accuracy(_as_2d(z_minus_i), np.asarray(labels))
    return {"acc_i": acc_i, "acc_minus_i": acc_minus, "cag": float(acc_i - acc_minus)}


def mig_scores(z: Any, labels: Any) -> dict[str, float]:
    """Bounded MI-proxy scores based on absolute latent-label correlations."""
    x = _as_2d(z)
    labels = np.asarray(labels).astype(str)
    _, codes = np.unique(labels, return_inverse=True)
    if np.unique(codes).shape[0] < 2:
        return {"maxMIG": 0.0, "concatMIG": 0.0, "minMIG": 0.0}
    corrs = []
    for j in range(x.shape[1]):
        corrs.append(abs(_pearson(x[:, j], codes)))
    corrs_arr = np.clip(np.asarray(corrs, dtype=np.float64), 0.0, 1.0)
    return {
        "maxMIG": float(np.max(corrs_arr)),
        "concatMIG": float(np.mean(corrs_arr)),
        "minMIG": float(np.min(corrs_arr)),
    }


DEFAULT_ARTIFACT_FIELDS = [
    "run_id",
    "timestamp",
    "seed",
    "dataset",
    "model",
    "split",
    "metric",
    "value",
    "status",
    "notes",
]


def write_artifact_schema(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> Path:
    """Write F10 audit rows, preserving skipped external baselines explicitly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(DEFAULT_ARTIFACT_FIELDS)
        extras = sorted({key for row in rows for key in row if key not in fieldnames})
        fieldnames.extend(extras)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path
