"""Append-only Kang IFNB benchmark runner.

This script benchmarks model variants on the Kang IFNB dataset and appends one
row per (model, seed) to audits/kang_ifnb/metrics.csv.

Primary use:
- compare a new spVIPESmulti enhancement against the current baseline,
- contrast against original spVIPES (if installed),
- contrast against scvi-tools contrastiveVAE (ContrastiveVI).

The script is robust to missing optional baselines. If a model is unavailable,
it appends a row with explanatory notes so the audit trail remains complete.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import spVIPESmulti
from spVIPESmulti.metrics import (
    clisi,
    ilisi,
    kbet,
    knn_purity,
    leiden_ari,
    per_group_silhouette,
)

try:
    import pertpy as pt
except Exception as exc:  # pragma: no cover - hard dependency for this script
    raise RuntimeError("pertpy is required for Kang IFNB benchmarking. Install pertpy first.") from exc


ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "audits" / "kang_ifnb"
METRICS_CSV = AUDIT_DIR / "metrics.csv"


@dataclass
class BenchmarkConfig:
    run_id: str
    feature_id: str
    seeds: list[int]
    models: list[str]
    max_epochs: int
    batch_size: int
    n_top_genes: int
    max_cells_per_condition: int
    n_shared: int
    n_private: int
    n_hidden: int
    train_size: float
    orthogonality_min_cells_per_stratum: int
    kang_h5ad_path: str | None


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark Kang IFNB and append metrics rows.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--models",
        default="spvipesmulti,spvipes_original,contrastivevae",
        help="Comma-separated: spvipesmulti,spvipes_original,contrastivevae",
    )
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-top-genes", type=int, default=5000)
    parser.add_argument("--max-cells-per-condition", type=int, default=2500)
    parser.add_argument("--n-shared", type=int, default=16)
    parser.add_argument("--n-private", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--train-size", type=float, default=0.9)
    parser.add_argument("--orthogonality-min-cells-per-stratum", type=int, default=16)
    parser.add_argument(
        "--kang-h5ad-path",
        default=None,
        help="Optional path to a local Kang IFNB .h5ad file. If set, pertpy download is skipped.",
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    return BenchmarkConfig(
        run_id=args.run_id,
        feature_id=args.feature_id,
        seeds=seeds,
        models=models,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        n_top_genes=args.n_top_genes,
        max_cells_per_condition=args.max_cells_per_condition,
        n_shared=args.n_shared,
        n_private=args.n_private,
        n_hidden=args.n_hidden,
        train_size=args.train_size,
        orthogonality_min_cells_per_stratum=args.orthogonality_min_cells_per_stratum,
        kang_h5ad_path=args.kang_h5ad_path,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_kang_raw(kang_h5ad_path: str | None):
    if kang_h5ad_path:
        local_path = Path(kang_h5ad_path).expanduser().resolve()
        if not local_path.exists():
            raise RuntimeError(f"Local Kang file not found: {local_path}")
        if local_path.stat().st_size == 0:
            raise RuntimeError(f"Local Kang file is empty: {local_path}")
        return sc.read_h5ad(local_path)

    cache_path = Path.home() / ".cache" / "pertpy" / "kang_2018.h5ad"
    if cache_path.exists() and cache_path.stat().st_size == 0:
        cache_path.unlink(missing_ok=True)

    adata = pt.data.kang_2018()

    if cache_path.exists() and cache_path.stat().st_size == 0:
        raise RuntimeError(
            "pertpy downloaded a zero-byte kang_2018.h5ad file. "
            "Use --kang-h5ad-path /path/to/kang_2018.h5ad to run against a local copy."
        )
    return adata


def load_kang_ifnb(seed: int, n_top_genes: int, max_cells_per_condition: int, kang_h5ad_path: str | None):
    """Load and preprocess Kang IFNB, including megakaryocyte removal."""
    adata = _load_kang_raw(kang_h5ad_path)
    adata.obs_names_make_unique()

    if "cell_type" not in adata.obs.columns or "label" not in adata.obs.columns:
        raise RuntimeError("Kang dataset missing required columns 'cell_type' and 'label'.")

    adata = adata[adata.obs["cell_type"].astype(str) != "Megakaryocytes"].copy()

    # Subsample per condition for repeatable wall-clock and memory.
    rng = np.random.default_rng(seed)
    keep = []
    for condition in sorted(map(str, adata.obs["label"].unique())):
        pos = np.where(adata.obs["label"].astype(str).values == condition)[0]
        n_keep = min(max_cells_per_condition, len(pos))
        keep.extend(rng.choice(pos, size=n_keep, replace=False))
    adata = adata[np.array(sorted(keep))].copy()

    # Batch-aware HVG like notebook flow.
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        batch_key="label",
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    return adata


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _extract_final(history: Any, key: str) -> float | None:
    if history is None or key not in history or len(history[key]) == 0:
        return None
    val = history[key].iloc[-1]
    try:
        return float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
    except Exception:
        arr = np.asarray(val).ravel()
        return float(arr[0]) if arr.size else None


def _label_variance_ratio(z: np.ndarray, labels: np.ndarray) -> float:
    """Simple proxy for label retention; higher means more label-structured latent."""
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")

    overall_var = float(np.var(z, axis=0).sum()) + 1e-8
    between = 0.0
    global_mean = np.mean(z, axis=0)
    for u in unique:
        mask = labels == u
        if mask.sum() == 0:
            continue
        mean_u = np.mean(z[mask], axis=0)
        between += float(mask.sum()) * float(np.sum((mean_u - global_mean) ** 2))
    between /= float(len(labels))
    return float(between / overall_var)


def _compute_orthogonality(z_shared: np.ndarray, z_private: np.ndarray, strata: np.ndarray, min_cells: int):
    from spVIPESmulti.module.spVIPESmultimodule import _within_stratum_corr_norm

    m, w, excluded = _within_stratum_corr_norm(
        z_shared,
        z_private,
        strata,
        min_cells=min_cells,
    )
    return float(m), float(w), float(excluded)


def _common_metric_payload(
    *,
    z_shared: np.ndarray,
    z_private: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    min_cells_per_stratum: int,
) -> dict[str, float | None]:
    # kbet() returns rejection rate; convert to acceptance (higher is better)
    # to stay consistent with existing project reporting.
    kbet_rej = kbet(z_shared, groups, k=20)
    kbet_accept = 1.0 - kbet_rej if np.isfinite(kbet_rej) else float("nan")

    ortho_mean, ortho_worst, ortho_excluded = _compute_orthogonality(
        z_shared,
        z_private,
        strata,
        min_cells=min_cells_per_stratum,
    )

    cycle_l2 = (
        float(np.mean((z_shared - z_private[:, : z_shared.shape[1]]) ** 2))
        if z_private.shape[1] >= z_shared.shape[1]
        else float("nan")
    )

    return {
        "iLISI": float(ilisi(z_shared, groups, k=20)),
        "cLISI": float(clisi(z_shared, labels, k=20)),
        "kBET": float(kbet_accept),
        "knn_purity": float(knn_purity(z_shared, labels, k=20)),
        "leiden_ari": float(leiden_ari(z_shared, labels, resolution=0.8)),
        "silhouette_group": float(per_group_silhouette(z_private, groups)),
        "silhouette_label": float(per_group_silhouette(z_shared, labels)),
        "orthogonality_within_stratum": ortho_mean,
        "orthogonality_worst_stratum": ortho_worst,
        "orthogonality_excluded_strata": ortho_excluded,
        "cycle_consistency_l2": cycle_l2,
        "target_decoder_realism": None,
        "identity_preservation": _label_variance_ratio(z_shared, labels),
    }


def _build_base_row(cfg: BenchmarkConfig, seed: int, model_name: str, n_cells: int, n_genes: int) -> dict[str, Any]:
    return {
        "run_id": cfg.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_id": cfg.feature_id,
        "model_name": model_name,
        "baseline_name": "kang_ifnb_fixed_protocol",
        "seed": seed,
        "subset": "kang_ifnb",
        "n_cells": n_cells,
        "n_genes": n_genes,
        "train_wall_time_sec": None,
        "iLISI": None,
        "cLISI": None,
        "kBET": None,
        "knn_purity": None,
        "leiden_ari": None,
        "silhouette_group": None,
        "silhouette_label": None,
        "reconstruction_loss_per_cell": None,
        "kl_shared": None,
        "kl_private": None,
        "orthogonality_within_stratum": None,
        "orthogonality_worst_stratum": None,
        "orthogonality_excluded_strata": None,
        "cycle_consistency_l2": None,
        "target_decoder_realism": None,
        "identity_preservation": None,
        "notes": "",
    }


def _stitch_latents(latents_dict: dict[int, np.ndarray], group_indices_list: list[list[int]], n_obs: int) -> np.ndarray:
    sample = next(iter(latents_dict.values()))
    out = np.zeros((n_obs, sample.shape[1]), dtype=np.float32)
    for gi, idxs in enumerate(group_indices_list):
        out[np.asarray(idxs)] = latents_dict[gi]
    return out


def run_spvipesmulti(cfg: BenchmarkConfig, adata, seed: int) -> dict[str, Any]:
    row = _build_base_row(cfg, seed, "spvipesmulti", adata.n_obs, adata.n_vars)
    try:
        groups = sorted(map(str, adata.obs["label"].unique()))
        adatas_dict = {g: adata[adata.obs["label"].astype(str) == g].copy() for g in groups}

        prepared = spVIPESmulti.data.prepare_adatas(adatas_dict)
        setup_kwargs: dict[str, Any] = {
            "groups_key": "groups",
            "label_key": "cell_type",
        }
        if "replicate" in prepared.obs.columns:
            setup_kwargs["sample_key"] = "replicate"
        spVIPESmulti.model.spVIPESmulti.setup_anndata(prepared, **setup_kwargs)

        group_indices_list = [list(map(int, g)) for g in prepared.uns["groups_obs_indices"]]

        model = spVIPESmulti.model.spVIPESmulti(
            prepared,
            n_hidden=cfg.n_hidden,
            n_dimensions_shared=cfg.n_shared,
            n_dimensions_private=cfg.n_private,
            disentangle_preset="full",
            dropout_rate=0.1,
        )

        t0 = perf_counter()
        model.train(
            group_indices_list,
            max_epochs=cfg.max_epochs,
            batch_size=cfg.batch_size,
            train_size=cfg.train_size,
            early_stopping=False,
            n_epochs_kl_warmup=min(20, cfg.max_epochs),
            compute_orthogonality_metric=True,
            orthogonality_groupby_keys=("sample",) if "sample" in prepared.obs.columns else ("groups",),
            orthogonality_min_cells_per_stratum=cfg.orthogonality_min_cells_per_stratum,
        )
        row["train_wall_time_sec"] = round(perf_counter() - t0, 2)

        latents = model.get_latent_representation(group_indices_list=group_indices_list, batch_size=cfg.batch_size)
        z_shared = _stitch_latents(latents["shared"], group_indices_list, prepared.n_obs)
        z_private = _stitch_latents(latents["private"], group_indices_list, prepared.n_obs)

        strata = prepared.obs["sample"].values if "sample" in prepared.obs.columns else prepared.obs["groups"].values
        payload = _common_metric_payload(
            z_shared=z_shared,
            z_private=z_private,
            groups=prepared.obs["groups"].values,
            labels=prepared.obs["cell_type"].values,
            strata=strata,
            min_cells_per_stratum=cfg.orthogonality_min_cells_per_stratum,
        )
        row.update(payload)

        row["reconstruction_loss_per_cell"] = _extract_final(model.history, "reconstruction_loss_train")
        row["kl_shared"] = _extract_final(model.history, "kl_local_train")
        row["kl_private"] = _extract_final(model.history, "kl_local_train")
        row["notes"] = "ok"
        return row
    except Exception as exc:  # pragma: no cover - benchmark guardrail
        row["notes"] = f"failed: {type(exc).__name__}: {exc}"
        return row


def _import_original_spvipes_module():
    for name in ("spVIPES", "spvipes"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def run_spvipes_original(cfg: BenchmarkConfig, adata, seed: int) -> dict[str, Any]:
    row = _build_base_row(cfg, seed, "spvipes_original", adata.n_obs, adata.n_vars)
    mod = _import_original_spvipes_module()
    if mod is None:
        row["notes"] = "unavailable: could not import spVIPES/spvipes package"
        return row

    try:
        data_mod = getattr(mod, "data", None)
        model_mod = getattr(mod, "model", None)
        model_cls = getattr(model_mod, "spVIPES", None) if model_mod is not None else None
        prepare_fn = getattr(data_mod, "prepare_adatas", None) if data_mod is not None else None
        if model_cls is None or prepare_fn is None:
            row["notes"] = "unavailable: imported package has incompatible API"
            return row

        groups = sorted(map(str, adata.obs["label"].unique()))
        adatas_dict = {g: adata[adata.obs["label"].astype(str) == g].copy() for g in groups}
        prepared = prepare_fn(adatas_dict)

        setup = getattr(model_cls, "setup_anndata", None)
        if setup is None:
            row["notes"] = "unavailable: spVIPES.setup_anndata missing"
            return row

        setup_kwargs = {"groups_key": "groups", "label_key": "cell_type"}
        if "replicate" in prepared.obs.columns:
            setup_kwargs["sample_key"] = "replicate"
        setup(prepared, **setup_kwargs)

        group_indices_list = [list(map(int, g)) for g in prepared.uns["groups_obs_indices"]]

        model = model_cls(
            prepared,
            n_hidden=cfg.n_hidden,
            n_dimensions_shared=cfg.n_shared,
            n_dimensions_private=cfg.n_private,
            dropout_rate=0.1,
        )

        t0 = perf_counter()
        model.train(
            group_indices_list,
            max_epochs=cfg.max_epochs,
            batch_size=cfg.batch_size,
            train_size=cfg.train_size,
            early_stopping=False,
            n_epochs_kl_warmup=min(20, cfg.max_epochs),
        )
        row["train_wall_time_sec"] = round(perf_counter() - t0, 2)

        latents = model.get_latent_representation(group_indices_list=group_indices_list, batch_size=cfg.batch_size)
        z_shared = _stitch_latents(latents["shared"], group_indices_list, prepared.n_obs)
        z_private = _stitch_latents(latents["private"], group_indices_list, prepared.n_obs)

        strata = prepared.obs["sample"].values if "sample" in prepared.obs.columns else prepared.obs["groups"].values
        row.update(
            _common_metric_payload(
                z_shared=z_shared,
                z_private=z_private,
                groups=prepared.obs["groups"].values,
                labels=prepared.obs["cell_type"].values,
                strata=strata,
                min_cells_per_stratum=cfg.orthogonality_min_cells_per_stratum,
            )
        )
        row["reconstruction_loss_per_cell"] = _extract_final(
            getattr(model, "history", None), "reconstruction_loss_train"
        )
        row["kl_shared"] = _extract_final(getattr(model, "history", None), "kl_local_train")
        row["kl_private"] = _extract_final(getattr(model, "history", None), "kl_local_train")
        row["notes"] = "ok"
        return row
    except Exception as exc:  # pragma: no cover - benchmark guardrail
        row["notes"] = f"failed: {type(exc).__name__}: {exc}"
        return row


def _extract_contrastive_latents(model, adata, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    # Try explicit API first.
    shared = None
    private = None

    if hasattr(model, "get_latent_representation"):
        fn = model.get_latent_representation
        for shared_kw in (
            {"representation_kind": "salient", "batch_size": batch_size},
            {"latent": "salient", "batch_size": batch_size},
        ):
            try:
                shared = fn(adata=adata, **shared_kw)
                break
            except Exception:
                try:
                    shared = fn(**shared_kw)
                    break
                except Exception:
                    pass

        for private_kw in (
            {"representation_kind": "background", "batch_size": batch_size},
            {"latent": "background", "batch_size": batch_size},
        ):
            try:
                private = fn(adata=adata, **private_kw)
                break
            except Exception:
                try:
                    private = fn(**private_kw)
                    break
                except Exception:
                    pass

        if shared is None:
            # Last-resort default latent.
            try:
                shared = fn(adata=adata, batch_size=batch_size)
            except Exception:
                shared = fn(batch_size=batch_size)

    if private is None:
        private = np.zeros_like(shared)

    return np.asarray(shared, dtype=np.float32), np.asarray(private, dtype=np.float32)


def run_contrastivevae(cfg: BenchmarkConfig, adata, seed: int) -> dict[str, Any]:
    row = _build_base_row(cfg, seed, "contrastivevae", adata.n_obs, adata.n_vars)
    try:
        from scvi.external import ContrastiveVI
    except Exception as exc:
        row["notes"] = f"unavailable: {type(exc).__name__}: {exc}"
        return row

    try:
        a = adata.copy()
        batch_key = "replicate" if "replicate" in a.obs.columns else None
        ContrastiveVI.setup_anndata(a, batch_key=batch_key)

        labels = a.obs["label"].astype(str).values
        unique_labels = sorted(np.unique(labels))
        if len(unique_labels) != 2:
            row["notes"] = f"failed: expected 2 label conditions, got {unique_labels}"
            return row

        bg_label = next((x for x in unique_labels if "ctrl" in x.lower() or "unstim" in x.lower()), unique_labels[0])
        tg_label = [x for x in unique_labels if x != bg_label][0]

        background_indices = np.where(labels == bg_label)[0]
        target_indices = np.where(labels == tg_label)[0]

        model = ContrastiveVI(
            a,
            background_indices=background_indices,
            target_indices=target_indices,
            n_background_latent=cfg.n_private,
            n_salient_latent=cfg.n_shared,
        )

        t0 = perf_counter()
        model.train(
            max_epochs=cfg.max_epochs,
            batch_size=cfg.batch_size,
            train_size=cfg.train_size,
            early_stopping=False,
        )
        row["train_wall_time_sec"] = round(perf_counter() - t0, 2)

        z_shared, z_private = _extract_contrastive_latents(model, a, cfg.batch_size)

        strata = a.obs["replicate"].values if "replicate" in a.obs.columns else a.obs["label"].values
        row.update(
            _common_metric_payload(
                z_shared=z_shared,
                z_private=z_private,
                groups=a.obs["label"].values,
                labels=a.obs["cell_type"].values,
                strata=strata,
                min_cells_per_stratum=cfg.orthogonality_min_cells_per_stratum,
            )
        )

        row["notes"] = f"ok (background={bg_label}, target={tg_label})"
        return row
    except Exception as exc:  # pragma: no cover - benchmark guardrail
        row["notes"] = f"failed: {type(exc).__name__}: {exc}"
        return row


def append_rows(rows: list[dict[str, Any]]) -> None:
    if not METRICS_CSV.exists():
        raise RuntimeError(f"Missing metrics file: {METRICS_CSV}")

    existing_cols = pd.read_csv(METRICS_CSV, nrows=0).columns.tolist()

    with METRICS_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=existing_cols)
        for row in rows:
            normalized = {k: row.get(k, None) for k in existing_cols}
            writer.writerow(normalized)


def summarize(rows: list[dict[str, Any]]) -> str:
    compact = []
    for r in rows:
        compact.append(
            {
                "model_name": r["model_name"],
                "seed": r["seed"],
                "iLISI": _safe_float(r.get("iLISI")),
                "cLISI": _safe_float(r.get("cLISI")),
                "kBET": _safe_float(r.get("kBET")),
                "knn_purity": _safe_float(r.get("knn_purity")),
                "leiden_ari": _safe_float(r.get("leiden_ari")),
                "orthogonality_within_stratum": _safe_float(r.get("orthogonality_within_stratum")),
                "notes": r.get("notes", ""),
            }
        )
    return json.dumps(compact, indent=2)


def main() -> None:
    cfg = parse_args()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    for seed in cfg.seeds:
        print(f"\n=== seed={seed} ===")
        set_seed(seed)
        adata = load_kang_ifnb(
            seed=seed,
            n_top_genes=cfg.n_top_genes,
            max_cells_per_condition=cfg.max_cells_per_condition,
            kang_h5ad_path=cfg.kang_h5ad_path,
        )

        for model_name in cfg.models:
            print(f"  -> model={model_name}")
            if model_name == "spvipesmulti":
                row = run_spvipesmulti(cfg, adata, seed)
            elif model_name == "spvipes_original":
                row = run_spvipes_original(cfg, adata, seed)
            elif model_name == "contrastivevae":
                row = run_contrastivevae(cfg, adata, seed)
            else:
                row = _build_base_row(cfg, seed, model_name, adata.n_obs, adata.n_vars)
                row["notes"] = "skipped: unknown model name"

            all_rows.append(row)
            print(f"     notes={row['notes']}")

    append_rows(all_rows)
    print(f"\nAppended {len(all_rows)} rows to {METRICS_CSV}")
    print(summarize(all_rows))


if __name__ == "__main__":
    main()
