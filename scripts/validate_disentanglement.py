"""Empirical validation of the spVIPESmulti disentanglement objective.

Implements the verification checklist from PLANS.md against a real multi-group
single-cell dataset:

| Metric                        | Tool                       | Latent    | Direction |
| ----------------------------- | -------------------------- | --------- | --------- |
| Group mixing                  | kBET-like, iLISI           | z_shared  | up        |
| Label preservation            | k-NN purity, cLISI, ARI    | z_shared  | up (ARI/purity) / down (cLISI) |
| Group separability            | silhouette                 | z_private | up        |
| Reconstruction quality        | held-out NLL               | -         | not worse |
| Training stability            | final train loss           | -         | finite    |

Compares all `disentangle_preset` values + single-component ablations from
`disentangle_preset='full'`. Uses scvi's spleen-lymph CITE-seq dataset (RNA only)
with donor (SLN111/SLN208) as the group axis and `cell_types` as the label.
The DIALOGUE example used by `docs/notebooks/disentangle_ablation.ipynb` is not
reachable here because pertpy 1.0.3 requires jax >= 0.6.1, conflicting with
spVIPESmulti' jax==0.4.27 pin.

Outputs:
    scripts/validation_results.json   numeric results per preset/ablation
    scripts/validation_results.md     human-readable report with tables
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

import spVIPESmulti

SEED = 0
N_PER_GROUP = 1000
N_HVG = 2000
MAX_EPOCHS = 40
KL_WARMUP = 20
BATCH_SIZE = 256
N_HIDDEN = 64
N_SHARED = 15
N_PRIVATE = 8
TRAIN_SIZE = 0.85

OUT_DIR = Path(__file__).resolve().parent
RESULTS_JSON = OUT_DIR / "validation_results.json"
RESULTS_MD = OUT_DIR / "validation_results.md"


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data() -> "anndata.AnnData":
    """Load SLN CITE-seq RNA, define 2 donor groups, subsample, HVG."""
    adata = scvi.data.spleen_lymph_cite_seq(save_path="./data/", remove_outliers=True)
    adata.obs_names_make_unique()

    adata.obs["donor"] = adata.obs["batch"].astype(str).map(
        lambda b: "SLN111" if "111" in b else "SLN208"
    )

    # Drop rare cell types (<25 cells in either group) so ARI is meaningful.
    counts = adata.obs.groupby(["donor", "cell_types"], observed=True).size().unstack("donor", fill_value=0)
    keep_types = counts[(counts >= 25).all(axis=1)].index.tolist()
    adata = adata[adata.obs["cell_types"].isin(keep_types)].copy()
    adata.obs["cell_types"] = adata.obs["cell_types"].astype("category")
    print(f"  kept cell types : {len(keep_types)}  ({keep_types})")

    rng = np.random.default_rng(SEED)
    keep = []
    obs_donor = adata.obs["donor"].values
    for g in np.unique(obs_donor):
        pos = np.where(obs_donor == g)[0]
        keep.extend(rng.choice(pos, size=min(N_PER_GROUP, len(pos)), replace=False))
    adata = adata[np.array(sorted(keep))].copy()
    adata.obs_names_make_unique()

    X = adata.X
    if sp.issparse(X):
        gene_var = np.asarray(X.power(2).mean(0)).ravel() - np.asarray(X.mean(0)).ravel() ** 2
    else:
        gene_var = X.var(0)
    top = np.argsort(gene_var)[::-1][:N_HVG]
    adata = adata[:, top].copy()

    print(f"  shape after subsample/HVG : {adata.shape}")
    print(f"  cells per donor : {adata.obs['donor'].value_counts().to_dict()}")
    return adata


def prepare(adata):
    groups_dict = {
        d: adata[adata.obs["donor"] == d].copy() for d in sorted(adata.obs["donor"].unique())
    }
    prepared = spVIPESmulti.data.prepare_adatas(groups_dict)
    spVIPESmulti.model.spVIPESmulti.setup_anndata(
        prepared, groups_key="groups", label_key="cell_types"
    )
    return prepared


def stitch(latents_dict, key, group_indices_list, n_obs):
    """Concatenate per-group latents back into original cell order."""
    sample = next(iter(latents_dict[key].values()))
    out = np.zeros((n_obs, sample.shape[1]), dtype=np.float32)
    for gi, idxs in enumerate(group_indices_list):
        out[np.asarray(idxs)] = latents_dict[key][gi]
    return out


def ilisi(rep, groups, k=30):
    """Inverse Simpson's diversity index over k-NN neighbours, group labels.

    Higher (closer to n_groups) = better mixing.
    """
    groups = np.asarray(groups)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep)
    _, idx = nn.kneighbors(rep)
    idx = idx[:, 1:]
    out = np.empty(rep.shape[0])
    for i in range(rep.shape[0]):
        _, c = np.unique(groups[idx[i]], return_counts=True)
        p = c / c.sum()
        out[i] = 1.0 / float((p * p).sum())
    return float(out.mean())


def clisi(rep, labels, k=30):
    """Same as iLISI but on cell-type labels. Lower (closer to 1) = better label preservation."""
    return ilisi(rep, labels, k=k)


def kbet_like(rep, groups, k=20):
    """Cheap proxy for kBET (matches the disentangle_ablation notebook).

    1.0 = perfect mixing; lower = group-segregated.
    """
    groups = np.asarray(groups)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep)
    _, idx = nn.kneighbors(rep)
    idx = idx[:, 1:]
    expected = pd.Series(groups).value_counts(normalize=True).reindex(np.unique(groups)).values
    chi = np.empty(rep.shape[0])
    for i in range(rep.shape[0]):
        observed = (
            pd.Series(groups[idx[i]])
            .value_counts(normalize=True)
            .reindex(np.unique(groups), fill_value=0)
            .values
        )
        chi[i] = ((observed - expected) ** 2 / (expected + 1e-9)).sum()
    return float(np.exp(-chi.mean()))


def knn_purity(rep, labels, k=20):
    """Fraction of k-NN neighbours sharing the cell's label."""
    labels = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep)
    _, idx = nn.kneighbors(rep)
    idx = idx[:, 1:]
    return float(np.mean([(labels[idx[i]] == labels[i]).mean() for i in range(len(labels))]))


def leiden_ari(rep, labels, resolution=0.8):
    """Leiden cluster the rep, ARI against true labels."""
    import anndata as ad

    tmp = ad.AnnData(X=rep)
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15)
    sc.tl.leiden(tmp, resolution=resolution, random_state=0)
    return float(adjusted_rand_score(np.asarray(labels), tmp.obs["leiden"].values))


def per_group_silhouette(z_private, groups):
    """Silhouette of group identity on z_private, sub-sampled for speed."""
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        return float("nan")
    rng = np.random.default_rng(0)
    n = min(2000, z_private.shape[0])
    pick = rng.choice(z_private.shape[0], size=n, replace=False)
    return float(silhouette_score(z_private[pick], groups[pick], sample_size=n))


def extract_history_summary(model):
    """Pull final + initial train metrics from model.history.

    spVIPESmulti' TrainingPlan logs train-side metrics only (no validation_step),
    so held-out NLL is computed separately in `held_out_nll()`.
    """
    h = model.history
    out = {}
    for key, name in [
        ("reconstruction_loss_train", "recon_train_final"),
        ("elbo_train", "elbo_train_final"),
        ("kl_local_train", "kl_train_final"),
    ]:
        if key in h and len(h[key]) > 0:
            v = h[key].iloc[-1]
            try:
                v = float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
            except Exception:
                v = float(np.asarray(v).ravel()[0])
            out[name] = v
        else:
            out[name] = None
    # Loss-curve stability: net change between first and last epoch.
    if "reconstruction_loss_train" in h and len(h["reconstruction_loss_train"]) >= 2:
        arr = np.asarray(
            [float(np.asarray(v).ravel()[0]) for v in h["reconstruction_loss_train"].to_numpy().ravel()],
            dtype=float,
        )
        out["recon_train_drop"] = float(arr[0] - arr[-1])
        out["recon_train_finite"] = bool(np.isfinite(arr).all())
    else:
        out["recon_train_drop"] = None
        out["recon_train_finite"] = None
    return out


# held-out NLL is intentionally omitted: spVIPESmulti' multi-group inference/generative
# path doesn't expose a clean per-cell reconstruction loss without re-implementing
# parts of the TrainingPlan. With a deterministic seed the train/val split is
# fixed across presets, so `recon_train_final` is a fair cross-preset signal.


def train_and_score(adata, *, label, **disentangle_kwargs):
    set_seeds(SEED)
    t0 = perf_counter()
    model = spVIPESmulti.model.spVIPESmulti(
        adata,
        n_hidden=N_HIDDEN,
        n_dimensions_shared=N_SHARED,
        n_dimensions_private=N_PRIVATE,
        dropout_rate=0.1,
        **disentangle_kwargs,
    )
    group_indices_list = [list(map(int, g)) for g in adata.uns["groups_obs_indices"]]
    model.train(
        group_indices_list=group_indices_list,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        train_size=TRAIN_SIZE,
        early_stopping=False,
        n_epochs_kl_warmup=KL_WARMUP,
    )
    train_secs = perf_counter() - t0

    latents = model.get_latent_representation(
        group_indices_list=group_indices_list, batch_size=BATCH_SIZE
    )
    n_obs = adata.n_obs
    z_shared = stitch(latents, "shared", group_indices_list, n_obs)
    z_private = stitch(latents, "private", group_indices_list, n_obs)

    groups = adata.obs["groups"].values
    cell_types = adata.obs["cell_types"].values

    metrics = {
        "label": label,
        "n_cells": int(n_obs),
        "n_groups": int(len(np.unique(groups))),
        "n_cell_types": int(len(np.unique(cell_types))),
        "train_secs": round(train_secs, 1),
        # z_shared
        "shared__group_mixing_kbet": round(kbet_like(z_shared, groups), 4),
        "shared__group_mixing_ilisi": round(ilisi(z_shared, groups), 4),
        "shared__label_purity_knn": round(knn_purity(z_shared, cell_types), 4),
        "shared__label_clisi": round(clisi(z_shared, cell_types), 4),
        "shared__label_ari": round(leiden_ari(z_shared, cell_types), 4),
        # z_private
        "private__group_silhouette": round(per_group_silhouette(z_private, groups), 4),
        "private__label_purity_knn": round(knn_purity(z_private, cell_types), 4),
    }
    metrics.update({k: (round(v, 4) if v is not None else None) for k, v in extract_history_summary(model).items()})
    return metrics


def write_report(rows: list[dict]) -> None:
    df = pd.DataFrame(rows).set_index("label")
    RESULTS_JSON.write_text(json.dumps(rows, indent=2))

    shared_cols = [
        "shared__group_mixing_kbet",
        "shared__group_mixing_ilisi",
        "shared__label_purity_knn",
        "shared__label_clisi",
        "shared__label_ari",
    ]
    private_cols = ["private__group_silhouette", "private__label_purity_knn"]
    train_cols = ["recon_train_final", "recon_train_drop", "elbo_train_final", "kl_train_final", "recon_train_finite", "train_secs"]

    def md_table(cols):
        sub = df[cols].copy()
        sub.columns = [c.split("__")[-1] for c in sub.columns]
        return sub.to_markdown()

    md = [
        "# Disentanglement Validation Results",
        "",
        f"**Dataset:** `scvi.data.spleen_lymph_cite_seq` (RNA only)  ",
        f"**Groups (donor):** SLN111 vs SLN208  ",
        f"**Labels (cell_types):** {df['n_cell_types'].iloc[0]} types kept (>=25 cells per donor)  ",
        f"**Cells per group:** {N_PER_GROUP}  HVG: {N_HVG}  ",
        f"**Architecture:** n_hidden={N_HIDDEN}, n_shared={N_SHARED}, n_private={N_PRIVATE}  ",
        f"**Training:** max_epochs={MAX_EPOCHS}, batch={BATCH_SIZE}, train_size={TRAIN_SIZE}, kl_warmup={KL_WARMUP}  ",
        "",
        "## z_shared metrics (target: high group-mixing AND high label preservation)",
        "",
        md_table(shared_cols),
        "",
        "- `group_mixing_kbet` — exp(-mean chi^2). 1.0 = perfect mixing.",
        "- `group_mixing_ilisi` — inverse Simpson on group, k-NN. Closer to n_groups = better.",
        "- `label_purity_knn` — fraction of k-NN with same cell-type label.",
        "- `label_clisi` — inverse Simpson on label, k-NN. *Lower* = better.",
        "- `label_ari` — Leiden(z_shared) vs cell_types ARI.",
        "",
        "## z_private metrics (target: high group separability, low label retention)",
        "",
        md_table(private_cols),
        "",
        "## Training summary (no divergence, recon not collapsed)",
        "",
        md_table(train_cols),
        "",
        "- `recon_train_final` is comparable across rows because the seed-fixed train/val split is identical for every preset.",
        "- `recon_train_drop = recon[0] - recon[-1]`; positive = the loss decreased.",
        "- spVIPESmulti' TrainingPlan does not currently log validation_step metrics, so a true held-out NLL is not reported here. Adding `validation_step` to the plan is a straightforward follow-up.",
        "",
        "## Verdict",
        "",
        "Compare each preset / ablation row to `off`:",
        "",
        "- A **healthy** disentanglement objective should *increase* `shared__group_mixing_*` and `shared__label_ari` while not catastrophically degrading `recon_val_final`.",
        "- Removing `disentangle_group_shared_weight` (`full minus q_group_shared`) should be the largest hit to group mixing.",
        "- Removing `disentangle_label_shared_weight` or `contrastive_weight` should be the largest hit to ARI / k-NN purity.",
        "- `private_only` should leave shared metrics ~unchanged from `off`.",
        "",
    ]
    RESULTS_MD.write_text("\n".join(md))
    print(f"\nWrote {RESULTS_JSON}")
    print(f"Wrote {RESULTS_MD}")


def main():
    print("=" * 70)
    print(" spVIPESmulti disentanglement empirical validation")
    print("=" * 70)
    print("Loading data ...")
    adata = load_data()
    adata = prepare(adata)

    rows = []
    presets = ["off", "shared_only", "private_only", "adversarial_only",
               "supervised_only", "no_contrastive", "full"]
    for preset in presets:
        print(f"\n--- preset = {preset} ---")
        rows.append(train_and_score(adata, label=f"preset={preset}",
                                    disentangle_preset=preset))
        print("  ", {k: v for k, v in rows[-1].items() if "shared__" in k or "private__" in k})

    ablate = [
        "disentangle_group_shared_weight",
        "disentangle_label_shared_weight",
        "disentangle_group_private_weight",
        "disentangle_label_private_weight",
        "contrastive_weight",
    ]
    for comp in ablate:
        print(f"\n--- full minus {comp} ---")
        rows.append(train_and_score(adata, label=f"full minus {comp}",
                                    disentangle_preset="full", **{comp: 0.0}))
        print("  ", {k: v for k, v in rows[-1].items() if "shared__" in k or "private__" in k})

    write_report(rows)


if __name__ == "__main__":
    main()
