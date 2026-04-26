"""Empirical validation of the multimodal disentanglement objective (P8).

Companion to `scripts/validate_disentanglement.py`. Same protocol, but uses
both RNA and protein modalities from `scvi.data.spleen_lymph_cite_seq` —
exercising the multimodal loss path that wires `_compute_disentangle_losses`
into `_loss_multimodal`.

| Metric                        | Tool                       | Latent           | Direction |
| ----------------------------- | -------------------------- | ---------------- | --------- |
| Group mixing                  | kBET-like, iLISI           | z_shared (PoE)   | up        |
| Label preservation            | k-NN purity, cLISI, ARI    | z_shared (PoE)   | up (ARI/purity) / down (cLISI) |
| Group separability per modality | silhouette               | z_private[mod]   | up        |
| Reconstruction quality        | final train recon          | -                | not worse |
| Training stability            | recon delta over epochs    | -                | finite    |

Compares all 7 `disentangle_preset` values. Per-component ablations are
omitted to keep wall-clock under ~15 min on CPU; the existing
single-modality script covers ablation behaviour. The shape of the
disentanglement objective is identical (same five components, same
classifiers); P8 only adds per-modality looping for components 3 & 4.

Outputs:
    scripts/validation_results_multimodal.json   numeric per-preset rows
    scripts/validation_results_multimodal.md     human-readable report
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from time import perf_counter

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

import spVIPES

SEED = 0
N_PER_GROUP = 1000
N_HVG_RNA = 2000
MAX_EPOCHS = 40
KL_WARMUP = 20
BATCH_SIZE = 256
N_HIDDEN = 64
N_SHARED = 15
N_PRIVATE = 8
TRAIN_SIZE = 0.85

OUT_DIR = Path(__file__).resolve().parent
RESULTS_JSON = OUT_DIR / "validation_results_multimodal.json"
RESULTS_MD = OUT_DIR / "validation_results_multimodal.md"


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    """Load SLN CITE-seq with RNA + protein, define 2 donor groups, subsample."""
    adata = scvi.data.spleen_lymph_cite_seq(save_path="./data/", remove_outliers=True)
    adata.obs_names_make_unique()
    adata.obs["donor"] = adata.obs["batch"].astype(str).map(
        lambda b: "SLN111" if "111" in b else "SLN208"
    )

    # Drop rare cell types so ARI is meaningful.
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

    # HVG selection on RNA only (protein has ~15 features, no HVG step).
    X = adata.X
    if sp.issparse(X):
        gene_var = np.asarray(X.power(2).mean(0)).ravel() - np.asarray(X.mean(0)).ravel() ** 2
    else:
        gene_var = X.var(0)
    top = np.argsort(gene_var)[::-1][:N_HVG_RNA]
    adata = adata[:, top].copy()

    n_prot = adata.obsm["protein_expression"].shape[1]
    print(f"  shape after subsample/HVG : RNA={adata.shape}, protein={n_prot}")
    print(f"  cells per donor : {adata.obs['donor'].value_counts().to_dict()}")
    return adata


def split_modalities(adata):
    """Split a CITE-seq AnnData into per-donor {rna, protein} dicts.

    Both modalities carry identical obs (required because
    `prepare_multimodal_adatas` uses `merge='same'` for intra-group concat).
    """
    out = {}
    for g in sorted(adata.obs["donor"].unique()):
        sub = adata[adata.obs["donor"] == g].copy()
        rna = sub.copy()
        rna.uns = {}
        rna.obsm = {}
        prot_X = sub.obsm["protein_expression"].values.astype(np.float32)
        prot_var = pd.DataFrame(index=sub.obsm["protein_expression"].columns)
        prot = ad.AnnData(X=prot_X, obs=sub.obs.copy(), var=prot_var)
        out[str(g)] = {"rna": rna, "protein": prot}
    return out


def prepare(adata):
    groups = split_modalities(adata)
    prepared = spVIPES.data.prepare_multimodal_adatas(
        groups, modality_likelihoods={"rna": "nb", "protein": "nb"}
    )
    spVIPES.model.spVIPES.setup_anndata(
        prepared, groups_key="groups", label_key="cell_types",
        modality_likelihoods={"rna": "nb", "protein": "nb"},
    )
    return prepared


def stitch(latents_dict, key, group_indices_list, n_obs):
    sample = next(iter(latents_dict[key].values()))
    out = np.zeros((n_obs, sample.shape[1]), dtype=np.float32)
    for gi, idxs in enumerate(group_indices_list):
        out[np.asarray(idxs)] = latents_dict[key][gi]
    return out


def stitch_modality(latents_dict, mod, group_indices_list, n_obs):
    """Stitch the per-(group, modality) private latent into cell order."""
    pm = latents_dict["private_multimodal"]
    sample = next(v for (g, m), v in pm.items() if m == mod)
    out = np.zeros((n_obs, sample.shape[1]), dtype=np.float32)
    for gi, idxs in enumerate(group_indices_list):
        out[np.asarray(idxs)] = pm[(gi, mod)]
    return out


def ilisi(rep, groups, k=30):
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
    return ilisi(rep, labels, k=k)


def kbet_like(rep, groups, k=20):
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
    labels = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep)
    _, idx = nn.kneighbors(rep)
    idx = idx[:, 1:]
    return float(np.mean([(labels[idx[i]] == labels[i]).mean() for i in range(len(labels))]))


def leiden_ari(rep, labels, resolution=0.8):
    tmp = ad.AnnData(X=rep)
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15)
    sc.tl.leiden(tmp, resolution=resolution, random_state=0)
    return float(adjusted_rand_score(np.asarray(labels), tmp.obs["leiden"].values))


def per_group_silhouette(z, groups):
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        return float("nan")
    rng = np.random.default_rng(0)
    n = min(2000, z.shape[0])
    pick = rng.choice(z.shape[0], size=n, replace=False)
    return float(silhouette_score(z[pick], groups[pick], sample_size=n))


def extract_history_summary(model):
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


def train_and_score(adata, *, label, **disentangle_kwargs):
    set_seeds(SEED)
    t0 = perf_counter()
    model = spVIPES.model.spVIPES(
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

    groups = adata.obs["groups"].values
    cell_types = adata.obs["cell_types"].values

    metrics = {
        "label": label,
        "n_cells": int(n_obs),
        "n_groups": int(len(np.unique(groups))),
        "n_cell_types": int(len(np.unique(cell_types))),
        "train_secs": round(train_secs, 1),
        "shared__group_mixing_kbet": round(kbet_like(z_shared, groups), 4),
        "shared__group_mixing_ilisi": round(ilisi(z_shared, groups), 4),
        "shared__label_purity_knn": round(knn_purity(z_shared, cell_types), 4),
        "shared__label_clisi": round(clisi(z_shared, cell_types), 4),
        "shared__label_ari": round(leiden_ari(z_shared, cell_types), 4),
    }

    # Per-modality private metrics — exercises Components 3 & 4 in P8.
    if "private_multimodal" in latents:
        for mod in ("rna", "protein"):
            try:
                z_priv_mod = stitch_modality(latents, mod, group_indices_list, n_obs)
            except StopIteration:
                continue
            metrics[f"private_{mod}__group_silhouette"] = round(per_group_silhouette(z_priv_mod, groups), 4)
            metrics[f"private_{mod}__label_purity_knn"] = round(knn_purity(z_priv_mod, cell_types), 4)

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
    private_cols = [c for c in df.columns if c.startswith("private_")]
    train_cols = ["recon_train_final", "recon_train_drop", "elbo_train_final", "kl_train_final", "recon_train_finite", "train_secs"]

    def md_table(cols):
        sub = df[[c for c in cols if c in df.columns]].copy()
        sub.columns = [c.split("__")[-1] if "__" in c else c for c in sub.columns]
        return sub.to_markdown()

    md = [
        "# Multimodal Disentanglement Validation Results (P8)",
        "",
        f"**Dataset:** `scvi.data.spleen_lymph_cite_seq` (RNA + protein)  ",
        f"**Groups (donor):** SLN111 vs SLN208  ",
        f"**Modalities:** rna ({N_HVG_RNA} HVG, NB), protein (~15 features, NB)  ",
        f"**Labels (cell_types):** {df['n_cell_types'].iloc[0]} types kept (>=25 cells per donor)  ",
        f"**Cells per group:** {N_PER_GROUP}  ",
        f"**Architecture:** n_hidden={N_HIDDEN}, n_shared={N_SHARED}, n_private={N_PRIVATE}  ",
        f"**Training:** max_epochs={MAX_EPOCHS}, batch={BATCH_SIZE}, train_size={TRAIN_SIZE}, kl_warmup={KL_WARMUP}  ",
        "",
        "## z_shared (post-PoE) metrics",
        "",
        "Target: high group mixing AND high label preservation. The PoE shared latent",
        "is modality-agnostic; Components 1 (q_group_shared), 2 (q_label_shared), and 5",
        "(contrastive) operate here unchanged from single-modality.",
        "",
        md_table(shared_cols),
        "",
        "- `group_mixing_kbet` — exp(-mean chi^2). 1.0 = perfect mixing.",
        "- `group_mixing_ilisi` — inverse Simpson on group, k-NN. Closer to n_groups = better.",
        "- `label_purity_knn` — fraction of k-NN with same cell-type label.",
        "- `label_clisi` — inverse Simpson on label, k-NN. *Lower* = better.",
        "- `label_ari` — Leiden(z_shared) vs cell_types ARI.",
        "",
        "## z_private per-modality metrics",
        "",
        "P8 routes Components 3 (q_group_private) and 4 (q_label_private) over each",
        "modality's private latent. Target: high group separability per modality.",
        "",
        md_table(private_cols),
        "",
        "## Training summary",
        "",
        md_table(train_cols),
        "",
        "## Verdict",
        "",
        "Compare each preset row to `off`:",
        "",
        "- A **healthy** multimodal disentanglement objective should *increase* `shared__group_mixing_*`",
        "  and `shared__label_ari` while not catastrophically degrading `recon_train_final`.",
        "- `private_only` should leave shared metrics ~unchanged from `off` and improve",
        "  `private_*__group_silhouette` (since Components 3 & 4 push per-modality private",
        "  latents to encode group identity).",
        "- `shared_only` should leave private metrics ~unchanged from `off`.",
        "- `full` should deliver the largest combined effect on shared metrics; per-modality",
        "  private metrics may move modestly because the adversarial label-erasure (Component 4)",
        "  is summed across modalities.",
        "",
    ]
    RESULTS_MD.write_text("\n".join(md))
    print(f"\nWrote {RESULTS_JSON}")
    print(f"Wrote {RESULTS_MD}")


def main():
    print("=" * 70)
    print(" spVIPES multimodal disentanglement empirical validation (P8)")
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
        report = {k: v for k, v in rows[-1].items() if "shared__" in k or "private_" in k}
        print("  ", report)

    write_report(rows)


if __name__ == "__main__":
    main()
