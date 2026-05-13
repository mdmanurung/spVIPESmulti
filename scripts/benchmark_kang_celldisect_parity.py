"""Run F10b CellDISECT-style parity audits on Kang IFNB.

The runner always writes schema-complete F10 artifacts. External CellDISECT is
optional: unavailable or incompatible external packages are recorded as skipped rows
instead of failing the audit.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _ensure_torchvision_nms_schema() -> None:
    """Avoid import-time failures when torchvision lacks the optional nms op."""
    try:
        import torch

        if torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta"):
            return
    except RuntimeError:
        pass
    except Exception:
        return

    try:
        from torch.library import Library

        lib = Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        globals()["_TORCHVISION_COMPAT_LIB"] = lib
    except Exception:
        return


_ensure_torchvision_nms_schema()

import numpy as np
import scanpy as sc
from scipy import sparse

import spVIPESmulti as sv
from spVIPESmulti.interventions import metrics as f10_metrics
from spVIPESmulti.interventions import transfer_condition

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_DIR = ROOT / "audits" / "F10"
DEFAULT_KANG_NOTE_DIR = ROOT / "audits" / "kang_ifnb"
DATASET = "kang_ifnb"
METRIC_NAMES = [
    "counterfactual_pearson",
    "delta_pearson",
    "top_de_cosine",
    "wasserstein_mean_all",
    "wasserstein_mean_top",
]
ARTIFACT_FIELDS = [
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
SPLIT_ALIASES = {
    "cd14_mono": "CD14 Mono",
    "cd4_t": "CD4 T",
    "cd8_t": "CD8 T",
    "b_cells": "B",
    "b_cell": "B",
    "nk": "NK",
}


@dataclass
class Config:
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
    condition_key: str
    label_key: str
    donor_key: str
    splits: list[str]
    condition_from: str | None
    condition_to: str | None
    disentangle_preset: str
    audit_dir: str


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"f10b_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--kang-h5ad-path", default="docs/notebooks/data/kang_2018.h5ad")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells-per-condition", type=int, default=600)
    parser.add_argument("--n-top-genes", type=int, default=1000)
    parser.add_argument("--n-shared", type=int, default=16)
    parser.add_argument("--n-private", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--condition-key", default="label")
    parser.add_argument("--label-key", default="cell_type")
    parser.add_argument("--donor-key", default="replicate")
    parser.add_argument("--splits", default="cd14_mono")
    parser.add_argument("--condition-from", default=None)
    parser.add_argument("--condition-to", default=None)
    parser.add_argument("--disentangle-preset", default="full")
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    args = parser.parse_args(argv)

    return Config(
        run_id=args.run_id,
        kang_h5ad_path=args.kang_h5ad_path,
        seeds=[int(x.strip()) for x in args.seeds.split(",") if x.strip()],
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        max_cells_per_condition=args.max_cells_per_condition,
        n_top_genes=args.n_top_genes,
        n_shared=args.n_shared,
        n_private=args.n_private,
        n_hidden=args.n_hidden,
        condition_key=args.condition_key,
        label_key=args.label_key,
        donor_key=args.donor_key,
        splits=[x.strip() for x in args.splits.split(",") if x.strip()],
        condition_from=args.condition_from,
        condition_to=args.condition_to,
        disentangle_preset=args.disentangle_preset,
        audit_dir=args.audit_dir,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _matrix_to_dense(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _normalize_label(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def strip_group_prefixes(var_names: list[str], group_name: str) -> list[str]:
    prefix = f"{group_name}_"
    stripped = []
    for name in var_names:
        name = str(name)
        if not name.startswith(prefix):
            raise ValueError(f"Expected target variable {name!r} to start with prefix {prefix!r}.")
        stripped.append(name[len(prefix) :])
    return stripped


def _resolve_path(path: str) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    return out


def _row_base(cfg: Config, seed: int, model_name: str, split: str, metric: str) -> dict[str, Any]:
    return {
        "run_id": cfg.run_id,
        "timestamp": _now(),
        "seed": seed,
        "dataset": DATASET,
        "model": model_name,
        "split": split,
        "metric": metric,
        "value": "",
        "status": "",
        "notes": "",
    }


def empty_metric_rows(
    cfg: Config,
    seed: int,
    model_name: str,
    split: str,
    status: str,
    notes: str,
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for metric in metrics or METRIC_NAMES:
        row = _row_base(cfg, seed, model_name, split, metric)
        row.update({"status": status, "notes": notes})
        rows.append(row)
    return rows


def metric_rows_for_prediction(
    cfg: Config,
    seed: int,
    model_name: str,
    split: str,
    x_ctrl: Any,
    x_true: Any,
    x_pred: Any,
    notes: str = "ok",
) -> list[dict[str, Any]]:
    top_idx = f10_metrics.select_top_de_genes(x_ctrl, x_true, n_top=20)
    wasserstein = f10_metrics.wasserstein_gene_marginals(x_pred, x_true, top_idx=top_idx)
    values = {
        "counterfactual_pearson": f10_metrics.counterfactual_pearson(x_pred, x_true),
        "delta_pearson": f10_metrics.delta_pearson(x_ctrl, x_true, x_pred),
        "top_de_cosine": f10_metrics.top_de_cosine(x_ctrl, x_true, x_pred, n_top=20),
        "wasserstein_mean_all": wasserstein["mean_all"],
        "wasserstein_mean_top": wasserstein.get("mean_top", float("nan")),
    }
    rows = []
    for metric in METRIC_NAMES:
        row = _row_base(cfg, seed, model_name, split, metric)
        value = float(values[metric])
        row.update({"value": value, "status": "ok", "notes": notes})
        rows.append(row)
    return rows


def load_kang_subset(cfg: Config, seed: int):
    path = _resolve_path(cfg.kang_h5ad_path)
    if not path.exists():
        raise RuntimeError(f"Kang file not found: {path}")
    adata = sc.read_h5ad(path)
    adata.obs_names_make_unique()

    required = {cfg.condition_key, cfg.label_key, cfg.donor_key}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise RuntimeError(f"Kang file missing required obs columns: {sorted(missing)}")

    adata = adata[adata.obs[cfg.label_key].astype(str) != "Megakaryocytes"].copy()

    rng = np.random.default_rng(seed)
    keep: list[int] = []
    conditions = adata.obs[cfg.condition_key].astype(str).to_numpy()
    for condition in sorted(np.unique(conditions)):
        pos = np.where(conditions == condition)[0]
        n_keep = min(cfg.max_cells_per_condition, len(pos))
        keep.extend(rng.choice(pos, size=n_keep, replace=False).tolist())
    adata = adata[np.array(sorted(keep))].copy()

    if adata.n_vars > cfg.n_top_genes:
        x = _matrix_to_dense(adata.X).astype(np.float32, copy=False)
        top_idx = np.argsort(np.var(x, axis=0))[-cfg.n_top_genes :]
        adata = adata[:, np.sort(top_idx)].copy()

    return adata


def infer_condition_pair(adata, cfg: Config) -> tuple[str, str]:
    values = sorted(map(str, adata.obs[cfg.condition_key].unique()))
    if len(values) < 2:
        raise RuntimeError(f"Expected at least two conditions in {cfg.condition_key!r}; found {values}.")
    condition_from = cfg.condition_from
    condition_to = cfg.condition_to
    if condition_from is None:
        condition_from = next((x for x in values if "ctrl" in x.lower() or "control" in x.lower()), values[0])
    if condition_to is None:
        condition_to = next(
            (x for x in values if x != condition_from and ("stim" in x.lower() or "ifn" in x.lower())),
            next(x for x in values if x != condition_from),
        )
    if condition_from not in values or condition_to not in values:
        raise RuntimeError(f"Requested conditions {condition_from!r}->{condition_to!r}; available={values}.")
    return condition_from, condition_to


def split_to_cell_type(adata, cfg: Config, split: str) -> str:
    raw = split.removeprefix("split_")
    alias = SPLIT_ALIASES.get(raw.lower(), raw)
    alias_norm = _normalize_label(alias)
    cell_types = sorted(map(str, adata.obs[cfg.label_key].unique()))

    exact = [value for value in cell_types if _normalize_label(value) == alias_norm]
    if exact:
        return exact[0]

    partial = [
        value for value in cell_types if alias_norm in _normalize_label(value) or _normalize_label(value) in alias_norm
    ]
    if partial:
        return sorted(partial, key=len)[0]
    raise RuntimeError(f"Could not map split={split!r} to a cell type. Available cell types: {cell_types}")


def prepare_model_input(adata, cfg: Config):
    groups = sorted(map(str, adata.obs[cfg.condition_key].unique()))
    adatas = {group: adata[adata.obs[cfg.condition_key].astype(str) == group].copy() for group in groups}
    prepared = sv.data.prepare_adatas(adatas)
    sv.model.spVIPESmulti.setup_anndata(
        prepared,
        groups_key="groups",
        label_key=cfg.label_key,
        condition_key=cfg.condition_key,
        donor_key=cfg.donor_key,
        sample_key=cfg.donor_key,
    )
    group_indices_list = [list(map(int, group)) for group in prepared.uns["groups_obs_indices"]]
    return prepared, group_indices_list, groups


def run_spvipesmulti_split(cfg: Config, adata, seed: int, split: str) -> list[dict[str, Any]]:
    try:
        condition_from, condition_to = infer_condition_pair(adata, cfg)
        target_cell_type = split_to_cell_type(adata, cfg, split)
        labels = adata.obs[cfg.label_key].astype(str).to_numpy()
        conditions = adata.obs[cfg.condition_key].astype(str).to_numpy()
        ctrl_mask = (labels == target_cell_type) & (conditions == condition_from)
        true_mask = (labels == target_cell_type) & (conditions == condition_to)
        if not np.any(ctrl_mask) or not np.any(true_mask):
            return empty_metric_rows(
                cfg,
                seed,
                "spVIPESmulti",
                split,
                "failed",
                f"missing held-out cells for {target_cell_type!r} {condition_from!r}->{condition_to!r}",
            )

        train_mask = ~true_mask
        prepared, group_indices_list, groups = prepare_model_input(adata[train_mask].copy(), cfg)
        if condition_from not in groups or condition_to not in groups:
            return empty_metric_rows(
                cfg,
                seed,
                "spVIPESmulti",
                split,
                "failed",
                f"training data missing required groups {condition_from!r}->{condition_to!r}",
            )
        group_src = groups.index(condition_from)
        group_dst = groups.index(condition_to)

        source_mask = (prepared.obs[cfg.label_key].astype(str).to_numpy() == target_cell_type) & (
            prepared.obs[cfg.condition_key].astype(str).to_numpy() == condition_from
        )
        source_obs_names = prepared.obs_names[source_mask].to_numpy()
        if source_obs_names.size == 0:
            return empty_metric_rows(
                cfg,
                seed,
                "spVIPESmulti",
                split,
                "failed",
                f"no source cells found for {target_cell_type!r} under {condition_from!r}",
            )

        model = sv.model.spVIPESmulti(
            prepared,
            n_hidden=cfg.n_hidden,
            n_dimensions_shared=cfg.n_shared,
            n_dimensions_private=cfg.n_private,
            dropout_rate=0.1,
            use_nf_prior=False,
            disentangle_preset=cfg.disentangle_preset,
        )
        start = perf_counter()
        model.train(
            group_indices_list,
            max_epochs=cfg.max_epochs,
            batch_size=cfg.batch_size,
            train_size=0.9,
            early_stopping=False,
            n_epochs_kl_warmup=min(5, cfg.max_epochs),
            accelerator="cpu",
            devices=1,
        )
        wall_time = perf_counter() - start

        result = transfer_condition(
            model,
            prepared,
            cells=source_obs_names,
            condition_from=condition_from,
            condition_to=condition_to,
            group_src=group_src,
            group_dst=group_dst,
            latent_type="shared",
        )
        target_var_names = list(map(str, result.info["var_names"]))
        raw_gene_names = strip_group_prefixes(target_var_names, groups[group_dst])
        missing_genes = [gene for gene in raw_gene_names if gene not in adata.var_names]
        if missing_genes:
            return empty_metric_rows(
                cfg,
                seed,
                "spVIPESmulti",
                split,
                "failed",
                f"target decoder genes are missing from source AnnData: {missing_genes[:5]}",
            )
        x_ctrl = _matrix_to_dense(adata[ctrl_mask, raw_gene_names].X)
        x_true = _matrix_to_dense(adata[true_mask, raw_gene_names].X)
        note = (
            f"ok; target_cell_type={target_cell_type}; condition={condition_from}->{condition_to}; "
            f"train_wall_time_sec={wall_time:.4f}; n_ctrl={x_ctrl.shape[0]}; n_true={x_true.shape[0]}"
        )
        return metric_rows_for_prediction(cfg, seed, "spVIPESmulti", split, x_ctrl, x_true, result.X, notes=note)
    except Exception as exc:  # pragma: no cover - benchmark guardrail
        return empty_metric_rows(cfg, seed, "spVIPESmulti", split, "failed", f"{type(exc).__name__}: {exc}")


def _import_external_celldisect() -> tuple[Any | None, str]:
    errors = []
    for name in ("celldisect", "CellDISECT"):
        try:
            return importlib.import_module(name), f"imported {name}"
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return None, "; ".join(errors)


def external_celldisect_rows(cfg: Config, seed: int, split: str) -> list[dict[str, Any]]:
    module, note = _import_external_celldisect()
    if module is None:
        return empty_metric_rows(
            cfg,
            seed,
            "CellDISECT",
            split,
            "skipped",
            f"external install unavailable ({note})",
        )
    return empty_metric_rows(
        cfg,
        seed,
        "CellDISECT",
        split,
        "skipped",
        f"external package import succeeded, but no reproducible adapter was detected ({note})",
    )


def build_recommendation(rows: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    statuses = [str(row.get("status", "")) for row in rows]
    sp_rows = [row for row in rows if row.get("model") == "spVIPESmulti"]
    external_rows = [row for row in rows if row.get("model") == "CellDISECT"]
    sp_failed = [row for row in sp_rows if row.get("status") == "failed"]
    external_ok = [row for row in external_rows if row.get("status") == "ok"]
    if sp_failed:
        verdict = "reject"
        reason = "One or more spVIPESmulti F10b metric rows failed."
    elif external_ok:
        verdict = "pass"
        reason = "spVIPESmulti and external CellDISECT rows are available for parity review."
    else:
        verdict = "informational"
        reason = "spVIPESmulti rows are available; external CellDISECT rows were skipped."
    return {
        "run_id": cfg.run_id,
        "feature_id": "F10b",
        "verdict": verdict,
        "reason": reason,
        "promotion": "audit harness only; no causal claims, F4 preset promotion, or F3 implementation",
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "config": asdict(cfg),
    }


def _summary_markdown(rows: list[dict[str, Any]], cfg: Config, recommendation: dict[str, Any]) -> str:
    lines = [
        "# F10b CellDISECT Parity Audit",
        "",
        f"- run_id: `{cfg.run_id}`",
        f"- verdict: `{recommendation['verdict']}`",
        f"- reason: {recommendation['reason']}",
        "- outputs are associative predictions and audit metrics only.",
        "",
        "| model | split | metric | value | status | notes |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        value = row.get("value", "")
        value_text = "" if value == "" else f"{float(value):.6g}"
        notes = str(row.get("notes", "")).replace("|", "/")
        lines.append(
            f"| {row.get('model', '')} | {row.get('split', '')} | {row.get('metric', '')} | "
            f"{value_text} | {row.get('status', '')} | {notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_artifacts(
    rows: list[dict[str, Any]],
    cfg: Config,
    audit_dir: str | Path | None = None,
    mirror_dir: str | Path | None = None,
) -> dict[str, Path]:
    audit_path = Path(audit_dir if audit_dir is not None else cfg.audit_dir)
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path
    mirror_path = Path(mirror_dir if mirror_dir is not None else DEFAULT_KANG_NOTE_DIR)
    if not mirror_path.is_absolute():
        mirror_path = ROOT / mirror_path
    audit_path.mkdir(parents=True, exist_ok=True)
    mirror_path.mkdir(parents=True, exist_ok=True)

    recommendation = build_recommendation(rows, cfg)
    metrics_path = f10_metrics.write_artifact_schema(audit_path / "metrics.csv", rows, fieldnames=ARTIFACT_FIELDS)
    summary_path = audit_path / "summary.md"
    recommendation_path = audit_path / "recommendation.json"
    summary = _summary_markdown(rows, cfg, recommendation)
    summary_path.write_text(summary, encoding="utf-8")
    recommendation_path.write_text(json.dumps(recommendation, indent=2) + "\n", encoding="utf-8")

    note_path = mirror_path / f"{cfg.run_id}_f10b.md"
    note_path.write_text(summary, encoding="utf-8")
    return {
        "metrics": metrics_path,
        "summary": summary_path,
        "recommendation": recommendation_path,
        "kang_note": note_path,
    }


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    all_rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        print(f"\n=== seed={seed} ===")
        np.random.seed(seed)
        adata = load_kang_subset(cfg, seed)
        for split in cfg.splits:
            print(f"  -> split={split} model=spVIPESmulti")
            sp_rows = run_spvipesmulti_split(cfg, adata, seed, split)
            all_rows.extend(sp_rows)
            print(f"     status={sorted({row['status'] for row in sp_rows})}")
            print(f"  -> split={split} model=CellDISECT")
            ext_rows = external_celldisect_rows(cfg, seed, split)
            all_rows.extend(ext_rows)
            print(f"     status={sorted({row['status'] for row in ext_rows})}")

    paths = write_artifacts(all_rows, cfg)
    print("\nWrote F10b artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
