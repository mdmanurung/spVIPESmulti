"""Run F4-lite covariate probe diagnostics on Kang IFNB."""

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
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

import spVIPESmulti as sv


ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "audits" / "F4"
METRICS_CSV = AUDIT_DIR / "metrics.csv"
SUMMARY_MD = AUDIT_DIR / "summary.md"
RECOMMENDATION_JSON = AUDIT_DIR / "recommendation.json"
VARIANTS = ["baseline", "donor_private", "donor_shared", "batch_shared", "full_bio"]
PRIMARY_METRIC = "balanced_accuracy"
MIN_PROMOTION_SEEDS = 3


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
    donor_key: str
    label_key: str
    batch_key: str | None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"f4_probes_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
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
    parser.add_argument("--donor-key", default="replicate")
    parser.add_argument("--label-key", default="cell_type")
    parser.add_argument("--batch-key", default=None)
    args = parser.parse_args()
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
        donor_key=args.donor_key,
        label_key=args.label_key,
        batch_key=args.batch_key,
    )


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

    required = {cfg.condition_key, cfg.donor_key, cfg.label_key}
    if cfg.batch_key is not None:
        required.add(cfg.batch_key)
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


def prepare_model_input(adata, cfg: Config):
    groups = sorted(map(str, adata.obs[cfg.condition_key].unique()))
    adatas = {g: adata[adata.obs[cfg.condition_key].astype(str) == g].copy() for g in groups}
    prepared = sv.data.prepare_adatas(adatas)
    setup_kwargs: dict[str, Any] = {
        "groups_key": "groups",
        "label_key": cfg.label_key,
        "condition_key": cfg.condition_key,
        "donor_key": cfg.donor_key,
        "sample_key": cfg.donor_key,
    }
    if cfg.batch_key is not None:
        setup_kwargs["batch_key"] = cfg.batch_key
    sv.model.spVIPESmulti.setup_anndata(prepared, **setup_kwargs)
    group_indices_list = [list(map(int, g)) for g in prepared.uns["groups_obs_indices"]]
    return prepared, group_indices_list


def _stitch(latents_by_group: dict[int, np.ndarray], group_indices_list: list[list[int]], n_obs: int) -> np.ndarray:
    sample = next(iter(latents_by_group.values()))
    out = np.zeros((n_obs, sample.shape[1]), dtype=np.float32)
    for group_idx, obs_idx in enumerate(group_indices_list):
        out[np.asarray(obs_idx)] = latents_by_group[group_idx]
    return out


def _variant_kwargs(cfg: Config, variant: str) -> tuple[dict[str, Any], str | None]:
    kwargs: dict[str, Any] = {}
    if variant == "baseline":
        pass
    elif variant == "donor_private":
        kwargs["disentangle_donor_private_weight"] = 0.5
    elif variant == "donor_shared":
        kwargs["disentangle_donor_shared_weight"] = 0.5
    elif variant == "batch_shared":
        if cfg.batch_key is None:
            return {}, "skipped: Kang batch_key not provided"
        kwargs["disentangle_batch_shared_weight"] = 0.5
    elif variant == "full_bio":
        kwargs["disentangle_donor_shared_weight"] = 0.5
        kwargs["disentangle_donor_private_weight"] = 0.5
        if cfg.batch_key is not None:
            kwargs["disentangle_batch_shared_weight"] = 0.5
    else:
        raise ValueError(f"Unknown variant: {variant}")

    kwargs.setdefault("disentangle_preset", "off")
    return kwargs, None


def train_variant(cfg: Config, prepared, group_indices_list, variant: str) -> tuple[dict[str, np.ndarray], float, str]:
    kwargs, skip_note = _variant_kwargs(cfg, variant)
    if skip_note is not None:
        return {}, 0.0, skip_note

    model = sv.model.spVIPESmulti(
        prepared,
        n_hidden=cfg.n_hidden,
        n_dimensions_shared=cfg.n_shared,
        n_dimensions_private=cfg.n_private,
        dropout_rate=0.1,
        use_nf_prior=False,
        **kwargs,
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
    wall = perf_counter() - start

    latents = model.get_latent_representation(group_indices_list=group_indices_list, batch_size=cfg.batch_size)
    return {
        "shared": _stitch(latents["shared"], group_indices_list, prepared.n_obs),
        "private": _stitch(latents["private"], group_indices_list, prepared.n_obs),
    }, wall, "ok"


def _probe_one(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float | None, float | None, str]:
    y = np.asarray(y).astype(str)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        return None, None, "skipped: <2 classes"
    stratify = y if np.min(counts) >= 2 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.3,
            random_state=seed,
            stratify=stratify,
        )
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        return float(accuracy_score(y_test, pred)), float(balanced_accuracy_score(y_test, pred)), "ok"
    except Exception as exc:
        return None, None, f"failed: {type(exc).__name__}: {exc}"


def probe_rows(cfg: Config, prepared, latents: dict[str, np.ndarray], variant: str, seed: int, wall_time: float, note: str):
    base = {
        "run_id": cfg.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_id": "F4",
        "dataset": "kang_ifnb",
        "seed": seed,
        "variant": variant,
        "n_cells": prepared.n_obs,
        "n_genes": prepared.n_vars,
        "train_wall_time_sec": round(wall_time, 4),
    }
    targets = {
        "donor": prepared.obs[cfg.donor_key].values,
        "condition": prepared.obs[cfg.condition_key].values,
        "cell_type": prepared.obs[cfg.label_key].values,
    }
    if cfg.batch_key is not None:
        targets["batch"] = prepared.obs[cfg.batch_key].values
    else:
        targets["batch"] = np.array(["missing"] * prepared.n_obs)

    rows = []
    if not latents:
        for target_name in targets:
            for latent_name in ("shared", "private"):
                rows.append({
                    **base,
                    "target": target_name,
                    "latent": latent_name,
                    "accuracy": None,
                    "balanced_accuracy": None,
                    "notes": note,
                })
        return rows

    for target_name, target_values in targets.items():
        for latent_name, x in latents.items():
            acc, bacc, probe_note = _probe_one(x, target_values, seed)
            rows.append({
                **base,
                "target": target_name,
                "latent": latent_name,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "notes": f"{note}; probe={probe_note}",
            })
    return rows


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _aggregate_probe_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["variant"]), str(row["target"]), str(row["latent"]))
        grouped.setdefault(key, []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for (variant, target, latent), group_rows in sorted(grouped.items()):
        values = [_as_float(r.get(PRIMARY_METRIC)) for r in group_rows]
        values = [v for v in values if v is not None]
        notes = sorted({str(r.get("notes", "")) for r in group_rows if r.get("notes")})
        key = f"{variant}:{target}:{latent}"
        if values:
            arr = np.asarray(values, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            cv = float(std / abs(mean)) if arr.size > 1 and abs(mean) > 1e-12 else 0.0
            out[key] = {
                "variant": variant,
                "target": target,
                "latent": latent,
                "metric": PRIMARY_METRIC,
                "mean": round(mean, 6),
                "std": round(std, 6),
                "cv": round(cv, 6),
                "n": int(arr.size),
                "notes": notes,
            }
        else:
            out[key] = {
                "variant": variant,
                "target": target,
                "latent": latent,
                "metric": PRIMARY_METRIC,
                "mean": None,
                "std": None,
                "cv": None,
                "n": 0,
                "notes": notes,
            }
    return out


def _metric_key(variant: str, target: str, latent: str) -> str:
    return f"{variant}:{target}:{latent}"


def _metric(aggregates: dict[str, dict[str, Any]], variant: str, target: str, latent: str) -> dict[str, Any] | None:
    stat = aggregates.get(_metric_key(variant, target, latent))
    if stat is None or stat.get("mean") is None:
        return None
    return stat


def _compare_gate(
    aggregates: dict[str, dict[str, Any]],
    *,
    gate_id: str,
    label: str,
    variant: str,
    target: str,
    latent: str,
    direction: str,
    pass_delta: float,
    reject_delta: float,
) -> dict[str, Any]:
    baseline = _metric(aggregates, "baseline", target, latent)
    observed = _metric(aggregates, variant, target, latent)
    gate = {
        "id": gate_id,
        "label": label,
        "variant": variant,
        "target": target,
        "latent": latent,
        "metric": PRIMARY_METRIC,
        "status": "missing",
        "baseline_mean": None if baseline is None else baseline["mean"],
        "variant_mean": None if observed is None else observed["mean"],
        "delta": None,
        "pass_rule": None,
        "reject_rule": None,
    }
    if baseline is None or observed is None:
        gate["reason"] = "baseline or variant probe metric is missing"
        return gate

    delta = float(observed["mean"]) - float(baseline["mean"])
    gate["delta"] = round(delta, 6)
    if direction == "increase":
        gate["pass_rule"] = f"delta >= {pass_delta:+.2f}"
        gate["reject_rule"] = f"delta < {reject_delta:+.2f}"
        if delta >= pass_delta:
            gate["status"] = "pass"
        elif delta < reject_delta:
            gate["status"] = "reject"
        else:
            gate["status"] = "review"
    elif direction == "decrease":
        gate["pass_rule"] = f"delta <= {pass_delta:+.2f}"
        gate["reject_rule"] = f"delta >= {reject_delta:+.2f}"
        if delta <= pass_delta:
            gate["status"] = "pass"
        elif delta >= reject_delta:
            gate["status"] = "reject"
        else:
            gate["status"] = "review"
    elif direction == "preserve":
        gate["pass_rule"] = f"delta >= {pass_delta:+.2f}"
        gate["reject_rule"] = f"delta < {reject_delta:+.2f}"
        if delta >= pass_delta:
            gate["status"] = "pass"
        elif delta < reject_delta:
            gate["status"] = "reject"
        else:
            gate["status"] = "review"
    else:
        raise ValueError(f"Unknown gate direction: {direction}")
    return gate


def _reported_gate(
    aggregates: dict[str, dict[str, Any]],
    *,
    gate_id: str,
    label: str,
    variant: str,
    target: str,
    latent: str,
) -> dict[str, Any]:
    baseline = _metric(aggregates, "baseline", target, latent)
    observed = _metric(aggregates, variant, target, latent)
    status = "pass" if baseline is not None and observed is not None else "reject"
    return {
        "id": gate_id,
        "label": label,
        "variant": variant,
        "target": target,
        "latent": latent,
        "metric": PRIMARY_METRIC,
        "status": status,
        "baseline_mean": None if baseline is None else baseline["mean"],
        "variant_mean": None if observed is None else observed["mean"],
        "delta": None if baseline is None or observed is None else round(observed["mean"] - baseline["mean"], 6),
        "pass_rule": "baseline and variant metric reported",
        "reject_rule": "metric missing",
    }


def _skipped_gate(gate_id: str, label: str, reason: str) -> dict[str, Any]:
    return {
        "id": gate_id,
        "label": label,
        "status": "skipped",
        "reason": reason,
    }


def _cv_gate(aggregates: dict[str, dict[str, Any]], n_seeds: int) -> dict[str, Any]:
    if n_seeds < 2:
        return {
            "id": "cross_seed_cv",
            "label": "Cross-seed CV on probe metrics",
            "status": "informational",
            "max_cv": None,
            "pass_rule": "max CV <= 0.20",
            "reject_rule": "max CV > 0.30",
            "reason": "requires at least two seeds",
        }
    cv_values = [
        stat["cv"]
        for stat in aggregates.values()
        if stat.get("n", 0) >= 2 and stat.get("cv") is not None and stat["variant"] != "batch_shared"
    ]
    if not cv_values:
        return {
            "id": "cross_seed_cv",
            "label": "Cross-seed CV on probe metrics",
            "status": "missing",
            "max_cv": None,
            "pass_rule": "max CV <= 0.20",
            "reject_rule": "max CV > 0.30",
            "reason": "no replicated probe metrics available",
        }

    max_cv = max(cv_values)
    if max_cv <= 0.20:
        status = "pass"
    elif max_cv > 0.30:
        status = "reject"
    else:
        status = "review"
    return {
        "id": "cross_seed_cv",
        "label": "Cross-seed CV on probe metrics",
        "status": status,
        "max_cv": round(float(max_cv), 6),
        "pass_rule": "max CV <= 0.20",
        "reject_rule": "max CV > 0.30",
    }


def build_recommendation(rows: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    aggregates = _aggregate_probe_metrics(rows)
    seeds_observed = sorted({int(r["seed"]) for r in rows})
    gates: list[dict[str, Any]] = [
        {
            "id": "seed_count",
            "label": "Minimum seed count",
            "status": "pass" if len(seeds_observed) >= MIN_PROMOTION_SEEDS else "informational",
            "observed": len(seeds_observed),
            "required": MIN_PROMOTION_SEEDS,
        },
        _compare_gate(
            aggregates,
            gate_id="donor_private_retention",
            label="Donor accuracy on z_private improves",
            variant="donor_private",
            target="donor",
            latent="private",
            direction="increase",
            pass_delta=0.10,
            reject_delta=0.02,
        ),
        _compare_gate(
            aggregates,
            gate_id="donor_shared_erasure",
            label="Donor accuracy on z_shared decreases",
            variant="donor_shared",
            target="donor",
            latent="shared",
            direction="decrease",
            pass_delta=-0.10,
            reject_delta=0.0,
        ),
        _compare_gate(
            aggregates,
            gate_id="full_bio_donor_private_retention",
            label="full_bio preserves donor signal in z_private",
            variant="full_bio",
            target="donor",
            latent="private",
            direction="increase",
            pass_delta=0.10,
            reject_delta=0.02,
        ),
        _compare_gate(
            aggregates,
            gate_id="full_bio_donor_shared_erasure",
            label="full_bio removes donor signal from z_shared",
            variant="full_bio",
            target="donor",
            latent="shared",
            direction="decrease",
            pass_delta=-0.10,
            reject_delta=0.0,
        ),
        _reported_gate(
            aggregates,
            gate_id="condition_shared_reported",
            label="Condition probe on z_shared is reported",
            variant="full_bio",
            target="condition",
            latent="shared",
        ),
        _compare_gate(
            aggregates,
            gate_id="full_bio_cell_type_preservation",
            label="full_bio preserves cell-type signal in z_shared",
            variant="full_bio",
            target="cell_type",
            latent="shared",
            direction="preserve",
            pass_delta=-0.03,
            reject_delta=-0.05,
        ),
    ]

    if cfg.batch_key is None:
        gates.append(
            _skipped_gate(
                "batch_shared_erasure",
                "Batch accuracy on z_shared decreases",
                "no real batch_key was provided",
            )
        )
    else:
        gates.extend(
            [
                _compare_gate(
                    aggregates,
                    gate_id="batch_shared_erasure",
                    label="Batch accuracy on z_shared decreases",
                    variant="batch_shared",
                    target="batch",
                    latent="shared",
                    direction="decrease",
                    pass_delta=-0.10,
                    reject_delta=0.0,
                ),
                _compare_gate(
                    aggregates,
                    gate_id="full_bio_batch_shared_erasure",
                    label="full_bio removes batch signal from z_shared",
                    variant="full_bio",
                    target="batch",
                    latent="shared",
                    direction="decrease",
                    pass_delta=-0.10,
                    reject_delta=0.0,
                ),
            ]
        )
    gates.append(_cv_gate(aggregates, len(seeds_observed)))

    hard_rejects = [g for g in gates if g["status"] == "reject"]
    undecided = [g for g in gates if g["status"] in {"missing", "review", "informational"}]
    if hard_rejects:
        verdict = "reject"
        reason = "One or more F4 probe gates failed."
    elif undecided:
        verdict = "informational"
        reason = "Probe gates ran, but promotion requires more seeds or manual review."
    else:
        verdict = "pass"
        reason = "All available F4 probe gates passed."

    passing_heads = []
    if next(g for g in gates if g["id"] == "donor_private_retention")["status"] == "pass":
        passing_heads.append("donor_private")
    if next(g for g in gates if g["id"] == "donor_shared_erasure")["status"] == "pass":
        passing_heads.append("donor_shared")
    batch_gate = next(g for g in gates if g["id"] == "batch_shared_erasure")
    if batch_gate["status"] == "pass":
        passing_heads.append("batch_shared")

    full_bio_gate_ids = {
        "full_bio_donor_private_retention",
        "full_bio_donor_shared_erasure",
        "condition_shared_reported",
        "full_bio_cell_type_preservation",
    }
    if cfg.batch_key is not None:
        full_bio_gate_ids.add("full_bio_batch_shared_erasure")
    full_bio_pass = all(g["status"] == "pass" for g in gates if g["id"] in full_bio_gate_ids)

    if verdict == "pass" and full_bio_pass:
        promotion = "probe gates support full_bio; verify reconstruction and integration gates before preset promotion"
    elif verdict == "pass" and passing_heads:
        promotion = f"probe gates support passing opt-in heads: {', '.join(passing_heads)}"
    elif passing_heads and not hard_rejects:
        promotion = f"candidate passing heads pending full gates: {', '.join(passing_heads)}"
    else:
        promotion = "keep F4 heads opt-in; do not promote presets"

    return {
        "run_id": cfg.run_id,
        "feature_id": "F4",
        "verdict": verdict,
        "reason": reason,
        "promotion": promotion,
        "primary_metric": PRIMARY_METRIC,
        "seeds_observed": seeds_observed,
        "gates": gates,
        "aggregates": list(aggregates.values()),
        "limitations": [
            "This script evaluates held-out classifier probes only.",
            "Reconstruction loss and iLISI/kBET gates must be checked by the broader Kang audit lane before preset promotion.",
        ],
        "config": asdict(cfg),
    }


def _format_gate_line(gate: dict[str, Any]) -> str:
    status = gate["status"]
    label = gate["label"]
    delta = gate.get("delta")
    if delta is None:
        detail = gate.get("reason", "")
    else:
        detail = f"delta={delta:+.3f}, baseline={gate['baseline_mean']}, variant={gate['variant_mean']}"
    return f"- `{status}` {label}: {detail}".rstrip()


def write_artifacts(rows: list[dict[str, Any]], cfg: Config) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    df_rows = [r for r in rows if r["balanced_accuracy"] is not None]
    recommendation = build_recommendation(rows, cfg)
    status = recommendation["verdict"]
    RECOMMENDATION_JSON.write_text(json.dumps(recommendation, indent=2) + "\n", encoding="utf-8")

    gate_lines = [_format_gate_line(g) for g in recommendation["gates"]]
    SUMMARY_MD.write_text(
        "\n".join(
            [
                "# F4-lite Covariate Probe Audit",
                "",
                f"Run ID: `{cfg.run_id}`",
                f"Verdict: **{status}**",
                f"Recommendation: {recommendation['promotion']}",
                "",
                "## Scope",
                "",
                "- Trains baseline and F4-lite covariate-head variants on a fixed Kang IFNB subset.",
                "- Fits held-out logistic probes for donor, batch, condition, and cell type on `z_shared` and `z_private`.",
                "- Missing technical batch is recorded as skipped rows for the standalone batch-shared variant.",
                "- The combined full_bio probe uses donor heads and adds batch-shared only when a real batch key is provided.",
                "",
                "## Output",
                "",
                f"- Metrics rows: `{METRICS_CSV.relative_to(ROOT)}`",
                f"- Non-skipped probe rows: `{len(df_rows)}`",
                "",
                "## Probe Gates",
                "",
                *gate_lines,
                "",
                "## Limitations",
                "",
                "- Reconstruction loss and iLISI/kBET gates are not computed by this probe script.",
                "- Preset promotion still requires those broader Kang audit metrics in addition to these probes.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    cfg = parse_args()
    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        adata = load_kang_subset(cfg, seed)
        prepared, group_indices_list = prepare_model_input(adata, cfg)
        for variant in VARIANTS:
            print(f"seed={seed} variant={variant}")
            latents, wall_time, note = train_variant(cfg, prepared, group_indices_list, variant)
            rows.extend(probe_rows(cfg, prepared, latents, variant, seed, wall_time, note))
    write_artifacts(rows, cfg)


if __name__ == "__main__":
    main()
