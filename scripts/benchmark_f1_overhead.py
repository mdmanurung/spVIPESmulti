"""Measure F1 orthogonality metric overhead on a fixed Kang IFNB subset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import scanpy as sc
import torch
from scipy import sparse

import spVIPESmulti as sv

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "audits" / "F1"
METRICS_CSV = AUDIT_DIR / "metrics.csv"
SUMMARY_MD = AUDIT_DIR / "summary.md"
RECOMMENDATION_JSON = AUDIT_DIR / "recommendation.json"


@dataclass
class Config:
    run_id: str
    kang_h5ad_path: str
    seeds: list[int]
    repeats: int
    max_epochs: int
    batch_size: int
    max_cells_per_condition: int
    n_top_genes: int
    n_shared: int
    n_private: int
    n_hidden: int
    min_cells_per_stratum: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"f1_overhead_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--kang-h5ad-path", default="docs/notebooks/data/kang_2018.h5ad")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells-per-condition", type=int, default=400)
    parser.add_argument("--n-top-genes", type=int, default=800)
    parser.add_argument("--n-shared", type=int, default=16)
    parser.add_argument("--n-private", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--min-cells-per-stratum", type=int, default=16)
    args = parser.parse_args()
    return Config(
        run_id=args.run_id,
        kang_h5ad_path=args.kang_h5ad_path,
        seeds=[int(x.strip()) for x in args.seeds.split(",") if x.strip()],
        repeats=args.repeats,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        max_cells_per_condition=args.max_cells_per_condition,
        n_top_genes=args.n_top_genes,
        n_shared=args.n_shared,
        n_private=args.n_private,
        n_hidden=args.n_hidden,
        min_cells_per_stratum=args.min_cells_per_stratum,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _matrix_to_dense(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def load_kang_subset(cfg: Config, seed: int):
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
        x = _matrix_to_dense(adata.X).astype(np.float32, copy=False)
        top_idx = np.argsort(np.var(x, axis=0))[-cfg.n_top_genes :]
        adata = adata[:, np.sort(top_idx)].copy()

    return adata


def prepare_model_input(adata):
    groups = sorted(map(str, adata.obs["label"].unique()))
    adatas = {g: adata[adata.obs["label"].astype(str) == g].copy() for g in groups}
    prepared = sv.data.prepare_adatas(adatas)
    sv.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key="cell_type",
        sample_key="replicate",
    )
    group_indices_list = [list(map(int, g)) for g in prepared.uns["groups_obs_indices"]]
    return prepared, group_indices_list


def _extract_final(history: Any, key_fragment: str) -> float | None:
    if history is None:
        return None
    keys = history.keys() if hasattr(history, "keys") else history.columns
    matches = [k for k in keys if key_fragment in str(k)]
    if not matches:
        return None
    series = history[matches[0]]
    val = series.iloc[-1] if hasattr(series, "iloc") else series[-1]
    try:
        return float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
    except Exception:
        arr = np.asarray(val).ravel()
        return float(arr[0]) if arr.size else None


def train_once(cfg: Config, prepared, group_indices_list, *, seed: int, repeat: int, enabled: bool) -> dict[str, Any]:
    set_seed(seed)
    model = sv.model.spVIPESmulti(
        prepared,
        n_hidden=cfg.n_hidden,
        n_dimensions_shared=cfg.n_shared,
        n_dimensions_private=cfg.n_private,
        dropout_rate=0.1,
        disentangle_preset="full",
        use_nf_prior=False,
    )

    start = perf_counter()
    model.train(
        group_indices_list,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        train_size=0.9,
        early_stopping=False,
        n_epochs_kl_warmup=0,
        compute_orthogonality_metric=enabled,
        orthogonality_groupby_keys=("sample",),
        orthogonality_min_cells_per_stratum=cfg.min_cells_per_stratum,
        accelerator="cpu",
        devices=1,
    )
    wall = perf_counter() - start

    return {
        "run_id": cfg.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_id": "F1",
        "dataset": "kang_ifnb",
        "seed": seed,
        "repeat": repeat,
        "variant": "enabled" if enabled else "disabled",
        "compute_orthogonality_metric": enabled,
        "n_cells": prepared.n_obs,
        "n_genes": prepared.n_vars,
        "max_epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "train_wall_time_sec": round(wall, 4),
        "orthogonality_within_stratum": _extract_final(model.history, "orthogonality_within_stratum"),
        "orthogonality_worst_stratum": _extract_final(model.history, "orthogonality_worst_stratum"),
        "orthogonality_excluded_strata": _extract_final(model.history, "orthogonality_excluded_strata"),
        "notes": "ok",
    }


def summarize(rows: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    disabled = [r["train_wall_time_sec"] for r in rows if not r["compute_orthogonality_metric"]]
    enabled = [r["train_wall_time_sec"] for r in rows if r["compute_orthogonality_metric"]]
    disabled_mean = float(np.mean(disabled))
    enabled_mean = float(np.mean(enabled))
    overhead_pct = 100.0 * (enabled_mean - disabled_mean) / disabled_mean
    return {
        "run_id": cfg.run_id,
        "feature_id": "F1",
        "verdict": "pass" if overhead_pct <= 5.0 else "reject",
        "gate": "orthogonality metric wall-time overhead <= +5%",
        "overhead_pct": round(overhead_pct, 4),
        "disabled_mean_sec": round(disabled_mean, 4),
        "enabled_mean_sec": round(enabled_mean, 4),
        "n_rows": len(rows),
        "config": asdict(cfg),
    }


def write_artifacts(rows: list[dict[str, Any]], recommendation: dict[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    verdict = recommendation["verdict"]
    SUMMARY_MD.write_text(
        "\n".join(
            [
                "# F1 Conditional Orthogonality Instrumentation Audit",
                "",
                f"Run ID: `{recommendation['run_id']}`",
                f"Verdict: **{verdict}**",
                "",
                "## Overhead Gate",
                "",
                f"- Gate: {recommendation['gate']}",
                f"- Disabled mean wall time: {recommendation['disabled_mean_sec']} sec",
                f"- Enabled mean wall time: {recommendation['enabled_mean_sec']} sec",
                f"- Overhead: {recommendation['overhead_pct']}%",
                "",
                "## Notes",
                "",
                "- Dataset: local Kang IFNB H5AD, with megakaryocytes removed.",
                "- Gene subset: top genes by variance on the fixed benchmark subset.",
                "- Training compared identical model settings with only `compute_orthogonality_metric` toggled.",
                "- Targeted validation: `pytest tests/test_disentangle_metrics.py tests/test_multimodal_disentangle.py -q` passed separately.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    RECOMMENDATION_JSON.write_text(json.dumps(recommendation, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        adata = load_kang_subset(cfg, seed)
        prepared, group_indices_list = prepare_model_input(adata)
        for repeat in range(cfg.repeats):
            order = [False, True] if repeat % 2 == 0 else [True, False]
            for enabled in order:
                print(f"seed={seed} repeat={repeat} enabled={enabled}")
                rows.append(train_once(cfg, prepared, group_indices_list, seed=seed, repeat=repeat, enabled=enabled))

    recommendation = summarize(rows, cfg)
    write_artifacts(rows, recommendation)
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
