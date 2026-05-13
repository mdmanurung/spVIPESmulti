"""Benchmark F3 optional shared-private orthogonality loss on Kang IFNB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_DIR = ROOT / "audits" / "F3"


@dataclass
class Config:
    """Runtime configuration for the F3 benchmark."""

    run_id: str
    kang_h5ad_path: str
    seeds: list[int]
    weights: list[float]
    max_epochs: int
    batch_size: int
    max_cells_per_condition: int
    n_top_genes: int
    n_shared: int
    n_private: int
    n_hidden: int
    min_cells_per_stratum: int
    output_dir: str


def parse_args() -> Config:
    """Parse command-line arguments into a benchmark config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=f"f3_orthogonality_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    parser.add_argument("--kang-h5ad-path", default="docs/notebooks/data/kang_2018.h5ad")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--weights", default="0.01,0.05,0.1,0.2")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells-per-condition", type=int, default=600)
    parser.add_argument("--n-top-genes", type=int, default=1000)
    parser.add_argument("--n-shared", type=int, default=16)
    parser.add_argument("--n-private", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--min-cells-per-stratum", type=int, default=16)
    parser.add_argument("--output-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    if args.smoke:
        seeds = seeds[:1] or [0]
        weights = weights[:1] or [0.1]
        args.max_epochs = min(args.max_epochs, 2)
        args.max_cells_per_condition = min(args.max_cells_per_condition, 80)
        args.n_top_genes = min(args.n_top_genes, 300)

    return Config(
        run_id=args.run_id,
        kang_h5ad_path=args.kang_h5ad_path,
        seeds=seeds,
        weights=weights,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        max_cells_per_condition=args.max_cells_per_condition,
        n_top_genes=args.n_top_genes,
        n_shared=args.n_shared,
        n_private=args.n_private,
        n_hidden=args.n_hidden,
        min_cells_per_stratum=args.min_cells_per_stratum,
        output_dir=args.output_dir,
    )


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if _is_finite(row.get(key))]
    return sum(vals) / len(vals) if vals else None


def _cv(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if _is_finite(row.get(key))]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    denom = abs(mean)
    if denom < 1e-8:
        return None
    variance = sum((val - mean) ** 2 for val in vals) / (len(vals) - 1)
    return math.sqrt(variance) / denom


def _relative_worse(candidate: float | None, baseline: float | None, *, higher_is_better: bool) -> float | None:
    if candidate is None or baseline is None or not _is_finite(candidate) or not _is_finite(baseline):
        return None
    denom = max(abs(float(baseline)), 1e-8)
    if higher_is_better:
        return max(0.0, (float(baseline) - float(candidate)) / denom)
    return max(0.0, (float(candidate) - float(baseline)) / denom)


_ROADMAP_GATE_METRICS = (
    "orthogonality_within_stratum",
    "reconstruction_loss_per_cell",
    "iLISI",
    "kBET",
    "cLISI",
    "knn_purity",
)


def _has_required_metrics(rows: list[dict[str, Any]]) -> bool:
    return all(all(_is_finite(row.get(metric)) for metric in _ROADMAP_GATE_METRICS) for row in rows)


def recommend_f3_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply F3 go/no-go gates to tidy benchmark rows."""
    ok_rows = [row for row in rows if row.get("notes") == "ok"]
    baseline_rows = [row for row in ok_rows if float(row.get("orthogonality_weight", -1.0)) == 0.0]
    candidate_weights = sorted(
        {float(row["orthogonality_weight"]) for row in ok_rows if float(row.get("orthogonality_weight", 0.0)) > 0.0}
    )

    if not baseline_rows or not candidate_weights:
        return {
            "feature_id": "F3",
            "verdict": "iterate",
            "recommended_weight": None,
            "reason": "missing successful baseline or candidate rows",
            "candidates": [],
        }

    baseline_seeds = {int(row["seed"]) for row in baseline_rows if _is_finite(row.get("seed"))}
    coverage_failures: list[str] = []
    if len(baseline_seeds) < 3:
        coverage_failures.append("baseline has fewer than 3 successful seeds")
    if not _has_required_metrics(baseline_rows):
        coverage_failures.append("baseline rows missing required F3 gate metrics")

    baseline = {
        "orthogonality_within_stratum": _mean(baseline_rows, "orthogonality_within_stratum"),
        "reconstruction_loss_per_cell": _mean(baseline_rows, "reconstruction_loss_per_cell"),
        "iLISI": _mean(baseline_rows, "iLISI"),
        "kBET": _mean(baseline_rows, "kBET"),
        "cLISI": _mean(baseline_rows, "cLISI"),
        "knn_purity": _mean(baseline_rows, "knn_purity"),
        "leiden_ari": _mean(baseline_rows, "leiden_ari"),
        "active_dims_shared": _mean(baseline_rows, "active_dims_shared"),
        "train_wall_time_sec": _mean(baseline_rows, "train_wall_time_sec"),
    }

    candidates: list[dict[str, Any]] = []
    for weight in candidate_weights:
        weight_rows = [row for row in ok_rows if float(row.get("orthogonality_weight", 0.0)) == weight]
        weight_seeds = {int(row["seed"]) for row in weight_rows if _is_finite(row.get("seed"))}
        means = {
            "orthogonality_within_stratum": _mean(weight_rows, "orthogonality_within_stratum"),
            "reconstruction_loss_per_cell": _mean(weight_rows, "reconstruction_loss_per_cell"),
            "iLISI": _mean(weight_rows, "iLISI"),
            "kBET": _mean(weight_rows, "kBET"),
            "cLISI": _mean(weight_rows, "cLISI"),
            "knn_purity": _mean(weight_rows, "knn_purity"),
            "leiden_ari": _mean(weight_rows, "leiden_ari"),
            "active_dims_shared": _mean(weight_rows, "active_dims_shared"),
            "train_wall_time_sec": _mean(weight_rows, "train_wall_time_sec"),
        }
        base_ortho = baseline["orthogonality_within_stratum"]
        cand_ortho = means["orthogonality_within_stratum"]
        ortho_reduction = None
        if base_ortho is not None and cand_ortho is not None:
            ortho_reduction = (base_ortho - cand_ortho) / max(abs(base_ortho), 1e-8)

        recon_worse = _relative_worse(
            means["reconstruction_loss_per_cell"],
            baseline["reconstruction_loss_per_cell"],
            higher_is_better=False,
        )
        integration_worse = {
            "iLISI": _relative_worse(means["iLISI"], baseline["iLISI"], higher_is_better=True),
            "kBET": _relative_worse(means["kBET"], baseline["kBET"], higher_is_better=True),
            "cLISI": _relative_worse(means["cLISI"], baseline["cLISI"], higher_is_better=False),
            "knn_purity": _relative_worse(means["knn_purity"], baseline["knn_purity"], higher_is_better=True),
        }
        cv_by_metric = {metric: _cv(weight_rows, metric) for metric in _ROADMAP_GATE_METRICS}
        diagnostics = {
            "active_dims_ok": (
                means["active_dims_shared"] is not None
                and baseline["active_dims_shared"] is not None
                and means["active_dims_shared"] >= baseline["active_dims_shared"] - 2
            ),
            "wall_overhead": _relative_worse(
                means["train_wall_time_sec"],
                baseline["train_wall_time_sec"],
                higher_is_better=False,
            ),
        }

        failures = []
        incomplete = []
        if weight_seeds != baseline_seeds or len(weight_seeds) < 3:
            incomplete.append("candidate seed coverage does not match the 3-seed baseline")
        if not _has_required_metrics(weight_rows):
            incomplete.append("candidate rows missing required F3 gate metrics")
        if ortho_reduction is None or ortho_reduction < 0.20:
            failures.append("orthogonality reduction <20%")
        if recon_worse is None or recon_worse > 0.05:
            failures.append("reconstruction NLL worsened >5% or missing")
        for metric in ("iLISI", "kBET"):
            worse = integration_worse[metric]
            if worse is None or worse > 0.10:
                failures.append(f"{metric} worsened >10% or missing")
        for metric in ("cLISI", "knn_purity"):
            worse = integration_worse[metric]
            if worse is None or worse > 0.05:
                failures.append(f"{metric} worsened >5% or missing")
        for metric, cv in cv_by_metric.items():
            if cv is None or cv > 0.20:
                failures.append(f"{metric} cross-seed CV >0.20 or missing")

        candidates.append(
            {
                "weight": weight,
                "passes": not failures and not incomplete,
                "orthogonality_reduction": ortho_reduction,
                "reconstruction_worse": recon_worse,
                "integration_worse": integration_worse,
                "cross_seed_cv": cv_by_metric,
                "incomplete": incomplete,
                "failures": failures,
                "diagnostics": diagnostics,
                "means": means,
            }
        )
        coverage_failures.extend(f"weight={weight}: {msg}" for msg in incomplete)

    if coverage_failures:
        verdict = "iterate"
        recommended_weight = None
        reason = "; ".join(dict.fromkeys(coverage_failures))
    elif passing := [candidate for candidate in candidates if candidate["passes"]]:
        best = min(passing, key=lambda item: item["weight"])
        verdict = "pass"
        recommended_weight = best["weight"]
        reason = f"weight={recommended_weight} passed all F3 gates"
    else:
        verdict = "reject"
        recommended_weight = None
        reason = "no tested F3 weight passed all gates"

    return {
        "feature_id": "F3",
        "verdict": verdict,
        "recommended_weight": recommended_weight,
        "reason": reason,
        "baseline": baseline,
        "candidates": candidates,
    }


def _extract_final(history: Any, key_fragment: str) -> float | None:
    if history is None:
        return None
    keys = history.keys() if hasattr(history, "keys") else history.columns
    matches = [key for key in keys if key_fragment in str(key)]
    if not matches:
        return None
    series = history[matches[0]]
    val = series.iloc[-1] if hasattr(series, "iloc") else series[-1]
    try:
        return float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
    except (TypeError, ValueError, IndexError, AttributeError):
        import numpy as np

        arr = np.asarray(val).ravel()
        return float(arr[0]) if arr.size else None


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


def _shared_metrics(model, prepared, group_indices_list, cfg: Config) -> dict[str, float | None]:
    import numpy as np

    from spVIPESmulti.metrics import clisi, ilisi, kbet, knn_purity, latent_dimension_stats, leiden_ari
    from spVIPESmulti.module.spVIPESmultimodule import _within_stratum_corr_norm

    latents = model.get_latent_representation(group_indices_list=group_indices_list, batch_size=cfg.batch_size)
    z_shared = _stitch(latents["shared"], group_indices_list, prepared.n_obs)
    z_private = _stitch(latents["private"], group_indices_list, prepared.n_obs)
    labels = prepared.obs["cell_type"].values
    groups = prepared.obs["groups"].values
    strata = prepared.obs["sample"].astype(str).values if "sample" in prepared.obs.columns else groups.astype(str)
    _, strata_ids = np.unique(strata, return_inverse=True)
    ortho_mean, ortho_worst, ortho_excluded = _within_stratum_corr_norm(
        z_shared,
        z_private,
        strata_ids,
        min_cells=cfg.min_cells_per_stratum,
    )
    dim_stats = latent_dimension_stats(z_shared)
    return {
        "iLISI": float(ilisi(z_shared, groups, k=20)),
        "kBET": float(kbet(z_shared, groups, k=20)),
        "cLISI": float(clisi(z_shared, labels, k=20)),
        "knn_purity": float(knn_purity(z_shared, labels, k=20)),
        "leiden_ari": float(leiden_ari(z_shared, labels, resolution=0.8)),
        "orthogonality_within_stratum": float(ortho_mean),
        "orthogonality_worst_stratum": float(ortho_worst),
        "orthogonality_excluded_strata": float(ortho_excluded),
        "active_dims_shared": float((~dim_stats["is_collapsed"]).sum()),
    }


def train_variant(
    cfg: Config, prepared, group_indices_list, *, seed: int, orthogonality_weight: float
) -> dict[str, Any]:
    """Train one baseline or F3-weighted variant and return one metrics row."""
    import numpy as np
    import torch

    import spVIPESmulti as sv

    np.random.seed(seed)
    torch.manual_seed(seed)
    variant = "baseline" if orthogonality_weight == 0 else f"orthogonality_{orthogonality_weight:g}"
    row: dict[str, Any] = {
        "run_id": cfg.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_id": "F3",
        "dataset": "kang_ifnb",
        "seed": seed,
        "variant": variant,
        "orthogonality_weight": orthogonality_weight,
        "n_cells": prepared.n_obs,
        "n_genes": prepared.n_vars,
        "max_epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "train_wall_time_sec": None,
        "reconstruction_loss_per_cell": None,
        "orthogonality_loss": None,
        "iLISI": None,
        "kBET": None,
        "cLISI": None,
        "knn_purity": None,
        "leiden_ari": None,
        "orthogonality_within_stratum": None,
        "orthogonality_worst_stratum": None,
        "orthogonality_excluded_strata": None,
        "active_dims_shared": None,
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
            orthogonality_weight=orthogonality_weight,
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
            compute_orthogonality_metric=True,
            orthogonality_groupby_keys=("sample",),
            orthogonality_min_cells_per_stratum=cfg.min_cells_per_stratum,
            accelerator="cpu",
            devices=1,
        )
        row["train_wall_time_sec"] = round(perf_counter() - start, 4)
        row["reconstruction_loss_per_cell"] = _extract_final(model.history, "reconstruction_loss_train")
        row["orthogonality_loss"] = _extract_final(model.history, "orthogonality_loss")
        row.update(_shared_metrics(model, prepared, group_indices_list, cfg))
    except Exception as exc:  # pragma: no cover - benchmark guardrail
        row["notes"] = f"failed: {type(exc).__name__}: {exc}"
    return row


def write_artifacts(rows: list[dict[str, Any]], recommendation: dict[str, Any], cfg: Config) -> None:
    """Write benchmark CSV, Markdown summary, and recommendation JSON."""
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "metrics.csv"
    summary_md = output_dir / "summary.md"
    recommendation_json = output_dir / "recommendation.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with metrics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with recommendation_json.open("w") as handle:
        json.dump({**recommendation, "config": asdict(cfg)}, handle, indent=2)

    lines = [
        "# F3 Orthogonality Loss Audit",
        "",
        f"- run_id: `{cfg.run_id}`",
        f"- verdict: `{recommendation['verdict']}`",
        f"- recommended_weight: `{recommendation['recommended_weight']}`",
        f"- reason: {recommendation['reason']}",
        "",
        "## Candidates",
        "",
    ]
    for candidate in recommendation.get("candidates", []):
        status = "PASS" if candidate["passes"] else "FAIL"
        reduction = candidate["orthogonality_reduction"]
        reduction_txt = "NA" if reduction is None else f"{100 * reduction:.2f}%"
        lines.append(f"- weight `{candidate['weight']}`: {status}, orthogonality reduction {reduction_txt}")
        if candidate["failures"]:
            lines.append(f"  failures: {', '.join(candidate['failures'])}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the F3 benchmark."""
    cfg = parse_args()
    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        adata = load_kang_subset(cfg, seed)
        prepared, group_indices_list = prepare_model_input(adata)
        for weight in [0.0, *cfg.weights]:
            rows.append(
                train_variant(
                    cfg,
                    prepared,
                    group_indices_list,
                    seed=seed,
                    orthogonality_weight=weight,
                )
            )

    recommendation = recommend_f3_variant(rows)
    write_artifacts(rows, recommendation, cfg)


if __name__ == "__main__":
    main()
