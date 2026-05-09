"""[ARCHIVED EVIDENCE] Ablation: full mixer vs low-rank mixer in LinearDecoderSPVIPE (P-PERF-2).

This script is kept as historical evidence supporting the 2026-05-09 decision
to make ``use_low_rank_mixer`` opt-in (default=False). Key findings from
``scripts/ablate_low_rank_mixer.json``:
  - rank=4: 3.8% WORSE reconstruction than baseline, SLOWER (59.4s vs 57.0s)
  - rank=8: 4.1% WORSE reconstruction than baseline, SLOWER (59.9s)
Conclusion: low-rank mixer is a regularizer, NOT a performance optimization.
Do not run this script to justify changing the ``use_low_rank_mixer`` default.

Original script description:
Compares three variants on the malaria B-cell dataset at 100 epochs:

  baseline   — full FCLayers mixer (use_low_rank_mixer=False)
  rank4      — low-rank mixer, rank=4
  rank8      — low-rank mixer, rank=8  (higher-capacity alternative)

Reports reconstruction loss, knn_purity, leiden_ari, ilisi, kbet, and elapsed time.

Acceptance criterion:
  - recon_train_final within +5% of baseline
  - knn_purity within -2pp of baseline
  - leiden_ari within -3pp of baseline
  - ilisi/kbet within ±5% of baseline

Usage:
  python scripts/ablate_low_rank_mixer.py           # 100 epochs
  python scripts/ablate_low_rank_mixer.py --epochs 20  # quick smoke test
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
from spVIPESmulti.utils import resolve_group_indices_list, store_latents

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "docs" / "notebooks" / "data"
OUT_DIR = Path(__file__).resolve().parent

SEED = 0
ABLATION_EPOCHS = 100
BATCH_SIZE = 1024
KL_WARMUP = 25
N_HIDDEN = 256
N_SHARED = 20
N_PRIVATE = 20
N_TOP_GENES = 3000
TRAIN_SIZE = 0.9
EARLY_STOP_PATIENCE = 20
CHECK_VAL_EVERY = 5


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_and_prepare() -> tuple[ad.AnnData, list[list[int]]]:
    adata = ad.read_csv(DATA_ROOT / "bcells_rna.csv", first_column_names=True)
    obs = pd.read_csv(DATA_ROOT / "bcells_obs.csv", index_col=0)
    adata.obs = obs
    adata.layers["counts"] = adata.X.copy()
    adata = adata[adata.obs["antigen_specific"] != "Negative"].copy()

    sc.pp.highly_variable_genes(
        adata, n_top_genes=N_TOP_GENES, flavor="seurat_v3", batch_key="batch"
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

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
    max_epochs: int,
    use_low_rank_mixer: bool = False,
    low_rank_mixer_rank: int = 4,
) -> dict[str, Any]:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    group_sizes = [len(g) for g in group_indices_list]
    group_loss_weights = [1 / n ** 0.5 for n in group_sizes]

    t0 = perf_counter()
    model = spVIPESmulti.model.spVIPESmulti(
        adata_spv,
        n_hidden=N_HIDDEN,
        n_dimensions_shared=N_SHARED,
        n_dimensions_private=N_PRIVATE,
        dropout_rate=0.1,
        disentangle_preset="full",
        disentangle_label_shared_weight=4.0,   # winning config from pilot
        disentangle_label_private_weight=0.5,
        use_jeffreys_integ=True,
        jeffreys_integ_weight=0.5,
        group_loss_weights=group_loss_weights,
        use_low_rank_mixer=use_low_rank_mixer,
        low_rank_mixer_rank=low_rank_mixer_rank,
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

    latents = model.get_latent_representation(group_indices_list, batch_size=1024)
    store_latents(adata_spv, latents, group_indices_list)

    z_shared = adata_spv.obsm["X_spVIPESmulti_shared"]
    groups_map = adata_spv.uns["groups_mapping"]
    z_private_dict = {
        str(groups_map.get(gi, gi)): latents["private_reordered"][gi]
        for gi in range(len(group_indices_list))
    }

    report = integration_report(
        z_shared,
        adata_spv.obs["antigen_specific"].values,
        adata_spv.obs["cluster_label"].values,
        z_private_dict=z_private_dict,
        k=20,
    )
    shared_row = report[report["latent"] == "z_shared"].iloc[0]

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
            history[short] = round(float(arr[-1]), 4)
            history[f"{short.replace('_final', '_epochs')}"] = len(arr)
        else:
            history[short] = None

    # Count mixer parameters for reference
    mixer_params = sum(
        p.numel()
        for name, p in model.module.named_parameters()
        if "mix_down" in name or "mix_up" in name
        or "sigmoid_decoder" in name or "mixture" in name
    )

    return {
        "label": label,
        "use_low_rank_mixer": use_low_rank_mixer,
        "low_rank_mixer_rank": low_rank_mixer_rank if use_low_rank_mixer else None,
        "mixer_params": mixer_params,
        "ilisi": round(float(shared_row["ilisi"]), 4),
        "kbet": round(float(shared_row["kbet"]), 4),
        "clisi": round(float(shared_row["clisi"]), 4),
        "knn_purity": round(float(shared_row["knn_purity"]), 4),
        "leiden_ari": round(float(shared_row["leiden_ari"]), 4),
        **history,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(max_epochs: int = ABLATION_EPOCHS) -> None:
    print(f"\n{'=' * 70}")
    print(f"P-PERF-2 ablation: full mixer vs low-rank mixer — {max_epochs} epochs")
    print(f"{'=' * 70}\n")

    print("Loading and preparing data …")
    adata_spv, gil = load_and_prepare()
    print(f"  shape: {adata_spv.shape}  groups: {adata_spv.uns['groups_mapping']}\n")

    variants = [
        ("baseline (full mixer)",  False, 4),
        ("low-rank mixer rank=4",  True,  4),
        ("low-rank mixer rank=8",  True,  8),
    ]

    results: list[dict[str, Any]] = []
    for label, use_lr, rank in variants:
        print(f"── {label} ──")
        res = train_and_score(
            adata_spv, gil,
            label=label,
            max_epochs=max_epochs,
            use_low_rank_mixer=use_lr,
            low_rank_mixer_rank=rank,
        )
        results.append(res)
        print(
            f"  mixer_params={res['mixer_params']:,}"
            f"  recon={res['recon_train_final']}"
            f"  knn_purity={res['knn_purity']:.3f}"
            f"  leiden_ari={res['leiden_ari']:.3f}"
            f"  ilisi={res['ilisi']:.3f}"
            f"  kbet={res['kbet']:.3f}"
            f"  elapsed={res['elapsed_s']:.0f}s\n"
        )

    # ── Summary table ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    cols_display = ["label", "mixer_params", "recon_train_final",
                    "knn_purity", "leiden_ari", "clisi", "ilisi", "kbet", "elapsed_s"]
    df_disp = df[[c for c in cols_display if c in df.columns]]

    print(f"\n{'=' * 70}")
    print("Ablation results:")
    print(f"{'=' * 70}")
    print(df_disp.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
    print()

    # ── Acceptance verdict ─────────────────────────────────────────────────────
    baseline = next(r for r in results if not r["use_low_rank_mixer"])
    print("Acceptance check vs baseline (full mixer):")
    for res in results[1:]:
        recon_ok   = res["recon_train_final"] <= baseline["recon_train_final"] * 1.05
        knn_ok     = res["knn_purity"] >= baseline["knn_purity"] - 0.02
        leiden_ok  = res["leiden_ari"] >= baseline["leiden_ari"] - 0.03
        ilisi_ok   = abs(res["ilisi"] - baseline["ilisi"]) / baseline["ilisi"] <= 0.05
        kbet_ok    = abs(res["kbet"] - baseline["kbet"]) / baseline["kbet"] <= 0.05
        passed     = all([recon_ok, knn_ok, leiden_ok, ilisi_ok, kbet_ok])
        verdict    = "PASS ✓" if passed else "FAIL ✗"
        print(
            f"  {res['label']:30s}  {verdict}"
            f"  recon={'OK' if recon_ok else 'FAIL'}"
            f"  knn={'OK' if knn_ok else 'FAIL'}"
            f"  leiden={'OK' if leiden_ok else 'FAIL'}"
            f"  ilisi={'OK' if ilisi_ok else 'FAIL'}"
            f"  kbet={'OK' if kbet_ok else 'FAIL'}"
        )
    print()

    # ── Save outputs ───────────────────────────────────────────────────────────
    json_path = OUT_DIR / "ablate_low_rank_mixer.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {json_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int, default=ABLATION_EPOCHS,
        help=f"Max epochs per variant (default: {ABLATION_EPOCHS})",
    )
    args = parser.parse_args()
    main(max_epochs=args.epochs)
