"""Run F11 nonlinear shared/private dependence diagnostics on Kang IFNB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_DIR = ROOT / "audits" / "F11"
CORE_CV_METRICS = (
    "hsic_rbf",
    "partial_corr_mean_abs",
    "partial_corr_adjusted_mean_abs",
)
REQUIRED_METRICS = (
    "hsic_rbf",
    "hsic_null_p95",
    "partial_corr_mean_abs",
    "partial_corr_adjusted_mean_abs",
    "orthogonality_within_stratum",
)


@dataclass
class Config:
    """Runtime configuration for the F11 diagnostic benchmark."""

    run_id: str
    kang_h5ad_path: str
    seeds: list[int]
    max_epochs: int
    batch_size: int
    max_cells_per_condition: int
    n_top_genes: int
    n_shared: int
    n_private: int
    n_hidden: int
    min_cells_per_stratum: int
    hsic_max_samples: int
    hsic_permutations: int
    linear_hidden_threshold: float
    output_dir: str


def parse_args() -> Config:
    """Parse command-line arguments into a benchmark config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=f"f11_nonlinear_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    parser.add_argument("--kang-h5ad-path", default="docs/notebooks/data/kang_2018.h5ad")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells-per-condition", type=int, default=600)
    parser.add_argument("--n-top-genes", type=int, default=1000)
    parser.add_argument("--n-shared", type=int, default=16)
    parser.add_argument("--n-private", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--min-cells-per-stratum", type=int, default=16)
    parser.add_argument("--hsic-max-samples", type=int, default=2000)
    parser.add_argument("--hsic-permutations", type=int, default=20)
    parser.add_argument("--linear-hidden-threshold", type=float, default=0.10)
    parser.add_argument("--output-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.smoke:
        seeds = seeds[:1] or [0]
        args.max_epochs = min(args.max_epochs, 2)
        args.max_cells_per_condition = min(args.max_cells_per_condition, 80)
        args.n_top_genes = min(args.n_top_genes, 300)
        args.hsic_permutations = min(args.hsic_permutations, 5)

    return Config(
        run_id=args.run_id,
        kang_h5ad_path=args.kang_h5ad_path,
        seeds=seeds,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        max_cells_per_condition=args.max_cells_per_condition,
        n_top_genes=args.n_top_genes,
        n_shared=args.n_shared,
        n_private=args.n_private,
        n_hidden=args.n_hidden,
        min_cells_per_stratum=args.min_cells_per_stratum,
        hsic_max_samples=args.hsic_max_samples,
        hsic_permutations=args.hsic_permutations,
        linear_hidden_threshold=args.linear_hidden_threshold,
        output_dir=args.output_dir,
    )


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if _is_finite(row.get(key))]
    return sum(values) / len(values) if values else None


def _cv(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if _is_finite(row.get(key))]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if abs(mean) < 1e-8:
        return None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / abs(mean)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def recommend_f11_diagnostics(
    rows: list[dict[str, Any]],
    *,
    expected_seeds: Sequence[int] = (0, 1, 2),
    cv_threshold: float = 0.30,
) -> dict[str, Any]:
    """Apply F11 finite/reproducible/hidden-signal gates to benchmark rows."""
    ok_rows = [row for row in rows if row.get("notes") == "ok"]
    expected = {int(seed) for seed in expected_seeds}
    observed = {int(row["seed"]) for row in ok_rows if _is_finite(row.get("seed"))}

    if not ok_rows:
        return {
            "feature_id": "F11",
            "verdict": "iterate",
            "reason": "missing successful F11 rows",
            "hidden_nonlinear_signal": False,
            "seeds": sorted(observed),
            "cross_seed_cv": {},
            "means": {},
        }

    failures: list[str] = []
    if observed != expected:
        failures.append(f"seed coverage mismatch: expected {sorted(expected)}, observed {sorted(observed)}")

    missing_metrics = sorted(
        {metric for row in ok_rows for metric in REQUIRED_METRICS if not _is_finite(row.get(metric))}
    )
    if missing_metrics:
        failures.append(f"missing or nonfinite required metrics: {missing_metrics}")

    cv_by_metric = {metric: _cv(ok_rows, metric) for metric in CORE_CV_METRICS}
    cv_failures = [
        f"{metric} CV >{cv_threshold:.2f} or missing"
        for metric, cv in cv_by_metric.items()
        if cv is None or cv > cv_threshold
    ]
    failures.extend(cv_failures)

    means = {metric: _mean(ok_rows, metric) for metric in REQUIRED_METRICS}
    hidden_signal = any(_truthy(row.get("hidden_nonlinear_signal")) for row in ok_rows)

    if failures:
        verdict = "iterate"
        reason = "; ".join(failures)
    elif hidden_signal:
        verdict = "pass"
        reason = "finite, reproducible F11 metrics found nonlinear signal not visible in F1"
    else:
        verdict = "informational"
        reason = "finite, reproducible F11 metrics, but Kang did not show hidden nonlinear leakage"

    return {
        "feature_id": "F11",
        "verdict": verdict,
        "reason": reason,
        "hidden_nonlinear_signal": hidden_signal,
        "seeds": sorted(observed),
        "cross_seed_cv": cv_by_metric,
        "means": means,
    }


def _matrix_to_dense(x: Any):
    import numpy as np
    from scipy import sparse

    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def load_kang_subset(cfg: Config, seed: int):
    """Load and subsample the Kang IFNB benchmark dataset."""
    import numpy as np
    import scanpy as sc

    path = Path(cfg.kang_h5ad_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    adata = sc.read_h5ad(path)
    adata.obs_names_make_unique()

    required = {"label", "cell_type", "replicate"}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise RuntimeError(f"Kang file missing required obs columns: {sorted(missing)}")

    adata = adata[adata.obs["cell_type"].astype(str) != "Megakaryocytes"].copy()

    rng = np.random.default_rng(seed)
    keep: list[int] = []
    labels = adata.obs["label"].astype(str).to_numpy()
    for condition in sorted(np.unique(labels)):
        pos = np.where(labels == condition)[0]
        n_keep = min(cfg.max_cells_per_condition, len(pos))
        keep.extend(rng.choice(pos, size=n_keep, replace=False).tolist())
    adata = adata[np.array(sorted(keep))].copy()

    if adata.n_vars > cfg.n_top_genes:
        x = _matrix_to_dense(adata.X).astype("float32", copy=False)
        top_idx = np.argsort(np.var(x, axis=0))[-cfg.n_top_genes :]
        adata = adata[:, np.sort(top_idx)].copy()

    return adata


def prepare_model_input(adata):
    """Prepare Kang AnnData and register fields for spVIPESmulti."""
    import spVIPESmulti as sv

    groups = sorted(map(str, adata.obs["label"].unique()))
    adatas = {group: adata[adata.obs["label"].astype(str) == group].copy() for group in groups}
    prepared = sv.data.prepare_adatas(adatas)
    sv.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_type",
        condition_key="label",
        donor_key="replicate",
        sample_key="replicate",
    )
    group_indices_list = [list(map(int, group)) for group in prepared.uns["groups_obs_indices"]]
    return prepared, group_indices_list


def _stitch(latents_by_group: dict[int, Any], group_indices_list: list[list[int]], n_obs: int):
    import numpy as np

    sample = next(iter(latents_by_group.values()))
    out = np.zeros((n_obs, sample.shape[1]), dtype=np.float32)
    for group_idx, obs_idx in enumerate(group_indices_list):
        out[np.asarray(obs_idx)] = latents_by_group[group_idx]
    return out


def _numeric_covariates(prepared):
    """Build numeric covariates from categorical audit strata."""
    import numpy as np
    import pandas as pd

    frames = []
    for key in ("groups", "sample"):
        if key in prepared.obs:
            dummies = pd.get_dummies(prepared.obs[key].astype(str), prefix=key, drop_first=True)
            if not dummies.empty:
                frames.append(dummies)
    if not frames:
        return None
    return np.asarray(pd.concat(frames, axis=1), dtype=float)


def _permutation_null(z_shared, z_private, cfg: Config, seed: int) -> tuple[float, float]:
    import numpy as np

    from spVIPESmulti.metrics import hsic_rbf

    if cfg.hsic_permutations < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed + 7919)
    values = []
    for permutation_idx in range(cfg.hsic_permutations):
        permuted = z_private[rng.permutation(z_private.shape[0])]
        values.append(
            hsic_rbf(
                z_shared,
                permuted,
                max_samples=cfg.hsic_max_samples,
                seed=seed + permutation_idx,
            )
        )
    return float(np.percentile(values, 95)), float(np.mean(values))


def _compute_f11_metrics(model, prepared, group_indices_list, cfg: Config, seed: int) -> dict[str, Any]:
    import numpy as np

    from spVIPESmulti.metrics import hsic_rbf, partial_corr_residualized
    from spVIPESmulti.module.spVIPESmultimodule import _within_stratum_corr_norm

    latents = model.get_latent_representation(group_indices_list=group_indices_list, batch_size=cfg.batch_size)
    z_shared = _stitch(latents["shared"], group_indices_list, prepared.n_obs)
    z_private = _stitch(latents["private"], group_indices_list, prepared.n_obs)

    strata = (
        prepared.obs["sample"].astype(str).values
        if "sample" in prepared.obs
        else prepared.obs["groups"].astype(str).values
    )
    _, strata_ids = np.unique(strata, return_inverse=True)
    ortho_mean, ortho_worst, ortho_excluded = _within_stratum_corr_norm(
        z_shared,
        z_private,
        strata_ids,
        min_cells=cfg.min_cells_per_stratum,
    )

    hsic_value = hsic_rbf(z_shared, z_private, max_samples=cfg.hsic_max_samples, seed=seed)
    hsic_null_p95, hsic_null_mean = _permutation_null(z_shared, z_private, cfg, seed)
    partial = partial_corr_residualized(z_shared, z_private)
    adjusted = partial_corr_residualized(z_shared, z_private, covariates=_numeric_covariates(prepared))
    hidden_signal = (
        math.isfinite(hsic_null_p95)
        and hsic_value > hsic_null_p95
        and abs(float(ortho_mean)) <= cfg.linear_hidden_threshold
    )

    return {
        "hsic_rbf": float(hsic_value),
        "hsic_null_p95": float(hsic_null_p95),
        "hsic_null_mean": float(hsic_null_mean),
        "partial_corr_mean_abs": float(partial["mean_abs_partial_corr"]),
        "partial_corr_max_abs": float(partial["max_abs_partial_corr"]),
        "partial_corr_adjusted_mean_abs": float(adjusted["mean_abs_partial_corr"]),
        "partial_corr_adjusted_max_abs": float(adjusted["max_abs_partial_corr"]),
        "partial_corr_n_pairs": int(partial["n_pairs"]),
        "partial_corr_adjusted_n_covariates": int(adjusted["n_covariates"]),
        "orthogonality_within_stratum": float(ortho_mean),
        "orthogonality_worst_stratum": float(ortho_worst),
        "orthogonality_excluded_strata": float(ortho_excluded),
        "hidden_nonlinear_signal": bool(hidden_signal),
    }


def train_and_score(cfg: Config, prepared, group_indices_list, *, seed: int) -> dict[str, Any]:
    """Train a default model and return one F11 diagnostic row."""
    import numpy as np
    import torch

    import spVIPESmulti as sv

    np.random.seed(seed)
    torch.manual_seed(seed)
    row: dict[str, Any] = {
        "run_id": cfg.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_id": "F11",
        "dataset": "kang_ifnb",
        "seed": seed,
        "variant": "default_diagnostics",
        "n_cells": prepared.n_obs,
        "n_genes": prepared.n_vars,
        "max_epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "train_wall_time_sec": None,
        "hsic_rbf": None,
        "hsic_null_p95": None,
        "hsic_null_mean": None,
        "partial_corr_mean_abs": None,
        "partial_corr_max_abs": None,
        "partial_corr_adjusted_mean_abs": None,
        "partial_corr_adjusted_max_abs": None,
        "partial_corr_n_pairs": None,
        "partial_corr_adjusted_n_covariates": None,
        "orthogonality_within_stratum": None,
        "orthogonality_worst_stratum": None,
        "orthogonality_excluded_strata": None,
        "hidden_nonlinear_signal": None,
        "notes": "ok",
    }
    try:
        model = sv.model.spVIPESmulti(
            prepared,
            n_hidden=cfg.n_hidden,
            n_dimensions_shared=cfg.n_shared,
            n_dimensions_private=cfg.n_private,
            dropout_rate=0.1,
            disentangle_preset="off",
            use_nf_prior=False,
        )
        start = perf_counter()
        model.train(
            group_indices_list,
            batch_size=cfg.batch_size,
            max_epochs=cfg.max_epochs,
            train_size=0.9,
            early_stopping=False,
            n_epochs_kl_warmup=min(5, cfg.max_epochs),
            accelerator="cpu",
            devices=1,
        )
        row["train_wall_time_sec"] = round(perf_counter() - start, 4)
        row.update(_compute_f11_metrics(model, prepared, group_indices_list, cfg, seed))
    except Exception as exc:  # pragma: no cover - benchmark guardrail
        row["notes"] = f"failed: {type(exc).__name__}: {exc}"
    return row


def write_artifacts(rows: list[dict[str, Any]], recommendation: dict[str, Any], cfg: Config) -> dict[str, Path]:
    """Write F11 metrics, summary, and recommendation artifacts."""
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "metrics.csv"
    summary_md = output_dir / "summary.md"
    recommendation_json = output_dir / "recommendation.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with recommendation_json.open("w", encoding="utf-8") as handle:
        json.dump({**recommendation, "config": asdict(cfg)}, handle, indent=2)
        handle.write("\n")

    lines = [
        "# F11 Nonlinear Dependence Diagnostics Audit",
        "",
        f"- run_id: `{cfg.run_id}`",
        f"- verdict: `{recommendation['verdict']}`",
        f"- reason: {recommendation['reason']}",
        f"- hidden_nonlinear_signal: `{recommendation['hidden_nonlinear_signal']}`",
        "",
        "## Gates",
        "",
        f"- Complete seed coverage: `{recommendation['seeds']}`",
        "- Core cross-seed CV threshold: `<= 0.30`",
        f"- Cross-seed CV: `{recommendation['cross_seed_cv']}`",
        "",
        "## Mean Metrics",
        "",
    ]
    for key, value in recommendation.get("means", {}).items():
        lines.append(f"- {key}: `{value}`")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "metrics": metrics_csv,
        "summary": summary_md,
        "recommendation": recommendation_json,
    }


def main() -> None:
    """Run the F11 diagnostic benchmark."""
    cfg = parse_args()
    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        adata = load_kang_subset(cfg, seed)
        prepared, group_indices_list = prepare_model_input(adata)
        rows.append(train_and_score(cfg, prepared, group_indices_list, seed=seed))

    recommendation = recommend_f11_diagnostics(rows, expected_seeds=cfg.seeds)
    write_artifacts(rows, recommendation, cfg)


if __name__ == "__main__":
    main()
