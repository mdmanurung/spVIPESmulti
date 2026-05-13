"""Pilot sweep: can stronger label pressure / relaxed Jeffreys / per-group HVG
improve cell-type separation in the shared latent space for the malaria B-cell
dataset?

Three single-lever variants are compared against a reduced-budget baseline run:

  baseline — v3 config (disentangle_label_shared_weight=2.0, jeffreys=0.5, global HVG)
  variant_A — disentangle_label_shared_weight=4.0  (stronger label pressure)
  variant_B — jeffreys_integ_weight=0.2             (relaxed Jeffreys / less mixing pressure)
  variant_C — per-group HVG union                  (richer feature set for rare subtypes)

All variants use 150 epochs with early stopping (patience=20 × 5-epoch cadence)
so the sweep completes in a reasonable wall-clock time.

Usage:
  # From the repo root with the scvi-test conda environment active:
  python scripts/pilot_celltype_separation.py

  # Speed through quickly for testing (10 epochs):
  python scripts/pilot_celltype_separation.py --epochs 10

Outputs:
  scripts/pilot_results_celltype.json   raw numeric results
  scripts/pilot_results_celltype.md     human-readable comparison table
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import spVIPESmulti
from spVIPESmulti.metrics import integration_report
from spVIPESmulti.utils import (
    highly_variable_genes_union,
    resolve_group_indices_list,
    store_latents,
)

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "docs" / "notebooks" / "data"
OUT_DIR = Path(__file__).resolve().parent

SEED = 0
PILOT_EPOCHS = 150
BATCH_SIZE = 1024
KL_WARMUP = 30  # scaled down from 75 proportionally to reduced budget
N_HIDDEN = 256
N_SHARED = 20
N_PRIVATE = 20
N_TOP_GENES = 3000
TRAIN_SIZE = 0.9
EARLY_STOP_PATIENCE = 20  # checks; effective patience = 20 * 5 = 100 epochs
CHECK_VAL_EVERY = 5


# ── Data loading ──────────────────────────────────────────────────────────────


def load_raw_adata() -> ad.AnnData:
    """Load malaria B-cell data (RNA + obs) without HVG selection."""
    adata = ad.read_csv(DATA_ROOT / "bcells_rna.csv", first_column_names=True)
    obs = pd.read_csv(DATA_ROOT / "bcells_obs.csv", index_col=0)
    adata.obs = obs
    adata.layers["counts"] = adata.X.copy()
    # Remove 'Negative' antigen group (no group signal)
    adata = adata[adata.obs["antigen_specific"] != "Negative"].copy()
    return adata


def select_hvg_global(adata: ad.AnnData) -> ad.AnnData:
    """Global seurat_v3 HVG (batch-aware) — baseline feature set."""
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor="seurat_v3", batch_key="batch")
    return adata[:, adata.var["highly_variable"]].copy()


def select_hvg_union(adata: ad.AnnData) -> ad.AnnData:
    """Per-antigen-group HVG union — variant C feature set."""
    return highly_variable_genes_union(adata, group_key="antigen_specific", n_top_genes=N_TOP_GENES)


def prepare_for_training(
    adata: ad.AnnData,
) -> tuple[ad.AnnData, list[list[int]]]:
    """Split into per-antigen AnnData objects, concatenate, setup_anndata."""
    antigens = sorted(adata.obs["antigen_specific"].unique())
    adatas_dict: dict[str, ad.AnnData] = {}
    for ag in antigens:
        sub = adata[adata.obs["antigen_specific"] == ag].copy()
        sub.uns = {}
        sub.obsm = {}
        sub.layers = {}
        adatas_dict[ag] = sub

    adata_spv = spVIPESmulti.data.prepare_adatas(adatas_dict)
    spVIPESmulti.model.spVIPESmulti.setup_anndata(
        adata_spv,
        groups_key="antigen_specific",
        label_key="cluster_label",
        batch_key="batch",
    )
    group_indices_list, _ = resolve_group_indices_list(adata_spv)
    return adata_spv, group_indices_list


# ── Training ──────────────────────────────────────────────────────────────────


def train_and_score(
    adata_spv: ad.AnnData,
    group_indices_list: list[list[int]],
    *,
    label: str,
    max_epochs: int = PILOT_EPOCHS,
    disentangle_label_shared_weight: float = 2.0,
    jeffreys_integ_weight: float = 0.5,
) -> dict[str, Any]:
    """Train one variant and return the integration report + timing."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    group_sizes = [len(g) for g in group_indices_list]
    group_loss_weights = [1 / n**0.5 for n in group_sizes]

    t0 = perf_counter()
    model = spVIPESmulti.model.spVIPESmulti(
        adata_spv,
        n_hidden=N_HIDDEN,
        n_dimensions_shared=N_SHARED,
        n_dimensions_private=N_PRIVATE,
        dropout_rate=0.1,
        disentangle_preset="full",
        disentangle_label_shared_weight=disentangle_label_shared_weight,
        disentangle_label_private_weight=0.5,
        use_jeffreys_integ=True,
        jeffreys_integ_weight=jeffreys_integ_weight,
        group_loss_weights=group_loss_weights,
    )

    model.train(
        group_indices_list,
        batch_size=BATCH_SIZE,
        max_epochs=max_epochs,
        train_size=TRAIN_SIZE,
        early_stopping=True,
        early_stopping_patience=EARLY_STOP_PATIENCE,
        early_stopping_monitor="reconstruction_loss_validation",
        n_epochs_kl_warmup=KL_WARMUP,
        check_val_every_n_epoch=CHECK_VAL_EVERY,
        plan_kwargs={
            "lr": 5e-4,
            "lr_scheduler_type": "cosine",
            "lr_min": 1e-5,
        },
    )

    elapsed = perf_counter() - t0

    # Extract latents and compute metrics
    latents = model.get_latent_representation(group_indices_list, batch_size=1024)
    store_latents(adata_spv, latents, group_indices_list)

    z_shared = adata_spv.obsm["X_spVIPESmulti_shared"]
    groups_map = adata_spv.uns["groups_mapping"]
    z_private_dict = {
        str(groups_map.get(gi, gi)): latents["private_reordered"][gi] for gi in range(len(group_indices_list))
    }

    report = integration_report(
        z_shared,
        adata_spv.obs["antigen_specific"].values,
        adata_spv.obs["cluster_label"].values,
        z_private_dict=z_private_dict,
        k=20,
    )
    shared_row = report[report["latent"] == "z_shared"].iloc[0]

    # Training history summary
    h = model.history
    history: dict[str, Any] = {"elapsed_s": round(elapsed, 1)}
    for key, short in [
        ("reconstruction_loss_train", "recon_train_final"),
        ("elbo_train", "elbo_train_final"),
        ("kl_local_train", "kl_train_final"),
    ]:
        if key in h and len(h[key]) > 0:
            arr = np.asarray(
                [float(np.asarray(v).ravel()[0]) for v in h[key].to_numpy().ravel()],
                dtype=float,
            )
            history[short] = round(float(arr[-1]), 2)
            history[f"{short.replace('_final', '_epochs')}"] = len(arr)
        else:
            history[short] = None

    return {
        "label": label,
        "ilisi": round(float(shared_row["ilisi"]), 4),
        "kbet": round(float(shared_row["kbet"]), 4),
        "clisi": round(float(shared_row["clisi"]), 4),
        "knn_purity": round(float(shared_row["knn_purity"]), 4),
        "leiden_ari": round(float(shared_row["leiden_ari"]), 4),
        **history,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main(max_epochs: int = PILOT_EPOCHS) -> None:
    print(f"\n{'=' * 70}")
    print(f"Malaria B-cell pilot sweep — {max_epochs} epochs per variant")
    print(f"{'=' * 70}\n")

    results: list[dict[str, Any]] = []

    # ── Baseline + Variant A + Variant B  (share the same global-HVG feature set) ──
    print("Loading data with global HVG selection …")
    adata_raw = load_raw_adata()
    adata_global = select_hvg_global(adata_raw.copy())
    adata_spv_global, gil_global = prepare_for_training(adata_global)
    print(f"  shape: {adata_spv_global.shape}  groups: {adata_spv_global.uns['groups_mapping']}\n")

    for label, kwargs in [
        ("baseline (label_shared=2.0, jeffreys=0.5, global HVG)", {}),
        ("variant_A (label_shared=4.0)", {"disentangle_label_shared_weight": 4.0}),
        ("variant_B (jeffreys=0.2)", {"jeffreys_integ_weight": 0.2}),
    ]:
        print(f"── {label} ──")
        res = train_and_score(adata_spv_global, gil_global, label=label, max_epochs=max_epochs, **kwargs)
        results.append(res)
        print(
            f"  knn_purity={res['knn_purity']:.3f}  leiden_ari={res['leiden_ari']:.3f}"
            f"  clisi={res['clisi']:.3f}  ilisi={res['ilisi']:.3f}  kbet={res['kbet']:.3f}"
            f"  elapsed={res['elapsed_s']:.0f}s\n"
        )

    # ── Variant C  (re-prepare data with per-group HVG union) ─────────────────
    print("Loading data with per-group HVG union …")
    adata_union = select_hvg_union(adata_raw.copy())
    adata_spv_union, gil_union = prepare_for_training(adata_union)
    print(
        f"  shape: {adata_spv_union.shape}  (extra genes vs global HVG: {adata_spv_union.shape[1] - adata_global.shape[1]})\n"
    )

    print("── variant_C (per-group HVG union) ──")
    res_c = train_and_score(
        adata_spv_union,
        gil_union,
        label="variant_C (per-group HVG union)",
        max_epochs=max_epochs,
    )
    results.append(res_c)
    print(
        f"  knn_purity={res_c['knn_purity']:.3f}  leiden_ari={res_c['leiden_ari']:.3f}"
        f"  clisi={res_c['clisi']:.3f}  ilisi={res_c['ilisi']:.3f}  kbet={res_c['kbet']:.3f}"
        f"  elapsed={res_c['elapsed_s']:.0f}s\n"
    )

    # ── Results table ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    cols_display = [
        "label",
        "knn_purity",
        "leiden_ari",
        "clisi",
        "ilisi",
        "kbet",
        "recon_train_final",
        "elbo_train_final",
    ]
    df_disp = df[[c for c in cols_display if c in df.columns]]

    print(f"\n{'=' * 70}")
    print("Pilot results (shared latent — cell-type structure vs group mixing):")
    print(f"{'=' * 70}")
    print(df_disp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print("Decision guide:")
    print("  knn_purity / leiden_ari ↑  →  better cell-type separation  (want: higher)")
    print("  clisi                   ↓  →  better label separation      (want: lower)")
    print("  ilisi / kbet            ↑  →  better antigen mixing        (want: stable)")
    print()
    print("Acceptance: improvement on knn_purity or leiden_ari + clisi drop, with")
    print("<10% loss on ilisi/kbet relative to baseline.")
    print()

    # ── Save outputs ───────────────────────────────────────────────────────────
    json_path = OUT_DIR / "pilot_results_celltype.json"
    md_path = OUT_DIR / "pilot_results_celltype.md"

    with json_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    with md_path.open("w") as f:
        f.write("# Malaria B-cell pilot sweep: cell-type separation\n\n")
        f.write(f"Epochs per variant: {max_epochs}\n\n")
        f.write(df_disp.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n")
        f.write(
            "**Decision guide**: knn_purity/leiden_ari ↑ = better separation; "
            "clisi ↓ = tighter clusters; ilisi/kbet stable = integration preserved.\n"
        )

    print(f"Results saved:\n  {json_path}\n  {md_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=PILOT_EPOCHS,
        help=f"Max epochs per variant (default: {PILOT_EPOCHS})",
    )
    args = parser.parse_args()
    main(max_epochs=args.epochs)
