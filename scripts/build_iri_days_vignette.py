"""Generate iri_days_vignette.ipynb programmatically.

This vignette mirrors the IRI (immediate repair) experiment from the
original spVIPES paper (Novella-Rausell et al., *Cell Systems Biology
Journal* 2024 — https://spj.science.org/doi/10.34133/csbj.0015), but
splits the data by **post-injury day** (1d / 3d / 14d) instead of by
the paper's two-group "IRI short" vs "IRI long" stratification. With
three groups we exercise the label-based PoE path that supports N >= 2
groups.
"""
import json
from pathlib import Path

cells = []


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src,
        }
    )


# ------------------------------------------------------------------
# Section A — Title and motivation
# ------------------------------------------------------------------
md("""# spVIPES on Kidney IRI — split by post-injury day (1d / 3d / 14d)

This vignette reproduces the **immediate repair injury (IRI)** analysis
from the original spVIPES paper
([Novella-Rausell *et al.*, 2024](https://spj.science.org/doi/10.34133/csbj.0015)),
but with one key difference: instead of grouping cells into the paper's
two buckets (**IRI short** vs **IRI long**), we split the data into
**three groups** by **post-injury day**:

| Group | Post-injury day |
|---|---|
| `day_1d` | 1 day after IRI |
| `day_3d` | 3 days after IRI |
| `day_14d` | 14 days after IRI |

A finer-grained day split lets the **shared latent** capture cell-type
identity that is preserved across the entire repair trajectory, while
each per-day **private latent** isolates the transcriptional programme
that is specific to that phase of repair (acute injury → proliferative
repair → scar resolution).

**What this vignette showcases:**

| Feature | API |
|---|---|
| **3-group integration** by day | `spVIPES.data.prepare_adatas`, label-based PoE |
| **Shared latent** preserves cell-type identity across days | `get_latent_representation` → shared UMAP |
| **Per-day private latent** isolates day-specific programmes | per-group private UMAP |
| **Quantitative integration metrics** | ARI on cell types + batch-mixing entropy on day labels |

> **Runtime.** The vignette subsamples to 2,000 cells per day and 2,000
> highly variable genes so it runs on a laptop CPU in roughly 10
> minutes. Increase `N_PER_GROUP` and `N_HVG` and use a GPU for a
> full-scale run.
""")

# ------------------------------------------------------------------
# Section B — Environment
# ------------------------------------------------------------------
md("## 1. Environment")

code("""import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import scipy.sparse as sp

import spVIPES

np.random.seed(42)
torch.manual_seed(42)
sc.settings.set_figure_params(dpi=80, frameon=False)

print(f"spVIPES : {spVIPES.__version__}")
print(f"scanpy  : {sc.__version__}")
print(f"torch   : {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"anndata : {ad.__version__}")""")

# ------------------------------------------------------------------
# Section C — Load the IRI data
# ------------------------------------------------------------------
md("""## 2. Load the IRI dataset

The original spVIPES paper uses the kidney **IRI** scRNA-seq atlas
from [Kirita *et al.* (2020)](https://doi.org/10.1073/pnas.2005477117)
(GEO accession **GSE139107**), which profiles proximal-tubule cells at
several post-injury timepoints. We expect the AnnData to live at
`./data/kirita_iri.h5ad` with at minimum the following `obs` columns:

- a **day / timepoint** column (e.g. `"timepoint"`, `"day"`,
  `"orig.ident"`) whose values include the strings ``"1d"``, ``"3d"``
  and ``"14d"`` — adjust `DAY_COL` and `DAY_VALUES` below if your copy
  of the data uses different labels (e.g. ``"day_1"``, ``"D1"``,
  ``"PT_1d"``);
- a **cell-type annotation** column (e.g. `"celltype"`, `"cell_type"`).

If you do not yet have the file locally, download the processed
AnnData from the paper's repository or rebuild it from the raw GEO
matrices. The cell below tries a couple of common paths and raises a
helpful error if none are found.
""")

code("""DATA_DIR = Path("./data")

CANDIDATE_PATHS = [
    DATA_DIR / "kirita_iri.h5ad",
    DATA_DIR / "iri.h5ad",
    DATA_DIR / "GSE139107_iri.h5ad",
]

iri_path = next((p for p in CANDIDATE_PATHS if p.exists()), None)
if iri_path is None:
    raise FileNotFoundError(
        "Could not find the IRI AnnData. Place it at one of:\\n  "
        + "\\n  ".join(str(p) for p in CANDIDATE_PATHS)
        + "\\n(or edit CANDIDATE_PATHS above)."
    )

adata_full = sc.read_h5ad(iri_path)
adata_full.obs_names_make_unique()

print(f"Loaded {iri_path}")
print(f"Shape         : {adata_full.shape}")
print(f"Obs columns   : {list(adata_full.obs.columns[:15])}")""")

# ------------------------------------------------------------------
# Section D — Identify the day column and the cell-type column
# ------------------------------------------------------------------
md("""## 3. Identify the day and cell-type columns

We auto-detect which `obs` column encodes post-injury day and which
encodes cell type. If the heuristics pick the wrong column for your
copy of the data, override `DAY_COL` / `LABEL_COL` directly.
""")

code("""# Heuristic: the day column should contain at least one of "1d", "3d", "14d"
DAY_VALUES = ["1d", "3d", "14d"]

def _looks_like_day_col(series: pd.Series) -> bool:
    vals = series.astype(str).str.lower().unique()
    return any(any(d in v for v in vals) for d in DAY_VALUES)

DAY_COL = next(
    (c for c in ["timepoint", "day", "time", "orig.ident", "sample", "condition"]
     if c in adata_full.obs.columns and _looks_like_day_col(adata_full.obs[c])),
    None,
)
if DAY_COL is None:
    DAY_COL = next(
        (c for c in adata_full.obs.columns if _looks_like_day_col(adata_full.obs[c])),
        None,
    )
if DAY_COL is None:
    raise ValueError(
        "No obs column with values containing '1d' / '3d' / '14d' was found. "
        "Set DAY_COL manually."
    )

# Heuristic: the label column should look like a cell-type annotation.
LABEL_COL = next(
    (c for c in ["celltype", "cell_type", "celltype.l1", "celltype.l2",
                 "cellType", "annotation", "subclass"]
     if c in adata_full.obs.columns),
    None,
)
if LABEL_COL is None:
    LABEL_COL = next(
        (c for c in adata_full.obs.columns
         if "cell" in c.lower() and "type" in c.lower()),
        None,
    )

print(f"DAY_COL  = {DAY_COL!r}")
print(f"LABEL_COL = {LABEL_COL!r}")
print()
print("Raw values in the day column:")
print(adata_full.obs[DAY_COL].value_counts())""")

code("""# Normalise the day labels to "1d" / "3d" / "14d"
def _to_day_label(value: str):
    s = str(value).lower()
    # Match "14d" before "1d" so we don't truncate
    for d in ["14d", "3d", "1d"]:
        if d in s:
            return f"day_{d}"
    return None

adata_full.obs["day"] = adata_full.obs[DAY_COL].apply(_to_day_label)

mask = adata_full.obs["day"].isin([f"day_{d}" for d in DAY_VALUES])
adata_full = adata_full[mask].copy()
adata_full.obs["day"] = adata_full.obs["day"].astype("category")

print("Cells per day:")
print(adata_full.obs["day"].value_counts().sort_index())""")

# ------------------------------------------------------------------
# Section E — Subsample + HVG selection
# ------------------------------------------------------------------
md("""## 4. Subsample cells and pick highly variable genes

To keep the vignette runnable on a laptop CPU we keep at most 2,000
cells per day and the top 2,000 highly variable genes. Increase these
numbers for a real run.

We also make sure raw counts are in `adata.X` (spVIPES uses a
Negative-Binomial likelihood by default).
""")

code("""N_PER_GROUP = 2000   # cells per day
N_HVG       = 2000   # genes

# --- Subsample cells per day, balanced ---
rng = np.random.default_rng(42)
day_col = adata_full.obs["day"].values
pos_keep = []
for d in sorted(adata_full.obs["day"].unique()):
    pos = np.where(day_col == d)[0]
    pick = rng.choice(pos, size=min(N_PER_GROUP, len(pos)), replace=False)
    pos_keep.extend(pick)
pos_keep = np.array(sorted(pos_keep))
adata = adata_full[pos_keep].copy()
adata.obs_names_make_unique()

# --- Move raw counts into .X if a "counts" layer exists ---
if "counts" in adata.layers:
    adata.X = adata.layers["counts"].copy()
    print("Using raw counts from adata.layers['counts'].")
else:
    print("No 'counts' layer found — assuming adata.X already holds raw counts.")

# --- Top-N_HVG genes by variance (robust on small subsamples) ---
X = adata.X
if sp.issparse(X):
    gene_var = (
        np.asarray(X.power(2).mean(axis=0)).flatten()
        - np.asarray(X.mean(axis=0)).flatten() ** 2
    )
else:
    gene_var = np.var(X, axis=0)
top_gene_idx = np.argsort(gene_var)[::-1][:N_HVG]
adata = adata[:, top_gene_idx].copy()

print(f"After subsample + HVG : {adata.shape}")
print("Cells per day         :")
print(adata.obs["day"].value_counts().sort_index())""")

# ------------------------------------------------------------------
# Section F — Three-group split + prepare_adatas
# ------------------------------------------------------------------
md("""## 5. Build the per-day AnnData dict and call `prepare_adatas`

`spVIPES.data.prepare_adatas` expects a dict mapping group name →
AnnData. We split the subsampled AnnData by day, hand the dict to
`prepare_adatas`, and end up with a single concatenated AnnData
carrying the per-group bookkeeping spVIPES needs.
""")

code("""days = sorted(adata.obs["day"].unique())  # ["day_14d", "day_1d", "day_3d"] alphabetically
# Re-order chronologically for nicer plots / printouts
day_order = [f"day_{d}" for d in DAY_VALUES if f"day_{d}" in days]

adatas_dict = {}
for d in day_order:
    sub = adata[adata.obs["day"] == d].copy()
    sub.uns = {}
    sub.obsm = {}
    sub.layers = {}
    adatas_dict[d] = sub
    print(f"  {d}: {sub.shape}")

adata_spv = spVIPES.data.prepare_adatas(adatas_dict)

print()
print("Concatenated AnnData :", adata_spv.shape)
print("Groups               :", list(adata_spv.uns["groups_mapping"].values()))
print("Group sizes          :", [len(g) for g in adata_spv.uns["groups_obs_indices"]])""")

# ------------------------------------------------------------------
# Section G — setup_anndata (label-based PoE)
# ------------------------------------------------------------------
md("""## 6. `setup_anndata` — label-based PoE

With **three** groups we must use the **label-based** PoE variant: the
optimal-transport variants in spVIPES are 2-group only. The label
column is the cell-type annotation we identified in Section 3.
""")

code("""spVIPES.model.spVIPES.setup_anndata(
    adata_spv,
    groups_key="groups",
    label_key=LABEL_COL,
)""")

# ------------------------------------------------------------------
# Section H — Instantiate and train
# ------------------------------------------------------------------
md("""## 7. Instantiate and train spVIPES

We use a moderate-sized model: 15 shared latent dimensions (cell-type
identity), 8 private dimensions per day (day-specific repair
programmes), and KL warmup over the first 30 epochs.
""")

code("""N_SHARED   = 15
N_PRIVATE  = 8
N_HIDDEN   = 128
DROPOUT    = 0.1
MAX_EPOCHS = 50
BATCH_SIZE = 128
KL_WARMUP  = 30

model = spVIPES.model.spVIPES(
    adata_spv,
    n_hidden=N_HIDDEN,
    n_dimensions_shared=N_SHARED,
    n_dimensions_private=N_PRIVATE,
    dropout_rate=DROPOUT,
)
print(model)""")

code("""group_indices_list = [
    np.where(adata_spv.obs["groups"] == g)[0]
    for g in day_order
]
print("Group sizes:", [len(g) for g in group_indices_list])

model.train(
    group_indices_list,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    train_size=0.9,
    early_stopping=False,
    n_epochs_kl_warmup=KL_WARMUP,
)""")

# ------------------------------------------------------------------
# Section I — Training diagnostics
# ------------------------------------------------------------------
md("""## 8. Training diagnostics

A quick look at the ELBO curve to confirm convergence.
""")

code("""fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(model.history["elbo_train"]["elbo_train"], label="ELBO (train)")
if "elbo_validation" in model.history:
    ax.plot(model.history["elbo_validation"]["elbo_validation"], label="ELBO (val)")
ax.set_xlabel("Epoch")
ax.set_ylabel("ELBO")
ax.set_title("spVIPES IRI — training curve")
ax.legend()
sns.despine()
plt.tight_layout()
plt.show()""")

# ------------------------------------------------------------------
# Section J — Latents
# ------------------------------------------------------------------
md("""## 9. Extract latent representations

`get_latent_representation` returns:

- a **shared latent** per cell (`shared_reordered[g]`),
- a **per-day private latent** (`private_reordered[g]`).

The `_reordered` variants are aligned to the order of cells inside
each group's slice of `adata_spv`.
""")

code("""latents = model.get_latent_representation(group_indices_list, batch_size=BATCH_SIZE)

print("Latent keys:", list(latents.keys()))
for g, d in enumerate(day_order):
    print(f"  {d}  shared: {latents['shared_reordered'][g].shape}  "
          f"private: {latents['private_reordered'][g].shape}")""")

code("""# Stitch the per-group shared latent back into the concatenated AnnData
latent_shared = np.concatenate(
    [latents["shared_reordered"][g] for g in range(len(day_order))],
    axis=0,
)
adata_spv = adata_spv[:len(latent_shared)].copy()
adata_spv.obsm["X_spVIPES_shared"] = latent_shared
print("X_spVIPES_shared :", adata_spv.obsm["X_spVIPES_shared"].shape)""")

# ------------------------------------------------------------------
# Section K — Shared UMAP
# ------------------------------------------------------------------
md("""## 10. Shared latent — UMAP

A successful shared latent should:

- separate cells by **cell type** (left panel),
- mix cells well across **days** within each cell-type cluster
  (right panel) — i.e. day labels should *not* drive the structure of
  the shared space.
""")

code("""sc.pp.neighbors(adata_spv, use_rep="X_spVIPES_shared",
                key_added="spvipes_shared", n_neighbors=15)
sc.tl.umap(adata_spv, neighbors_key="spvipes_shared", min_dist=0.3)
adata_spv.obsm["X_umap_shared"] = adata_spv.obsm["X_umap"].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.embedding(
    adata_spv, basis="X_umap_shared", color=LABEL_COL,
    ax=axes[0], show=False, title="Shared latent — cell types",
    legend_loc="on data", legend_fontsize=6, size=8,
)
sc.pl.embedding(
    adata_spv, basis="X_umap_shared", color="day",
    ax=axes[1], show=False, title="Shared latent — post-injury day",
    palette="Set2", size=8,
)
sns.despine()
plt.tight_layout()
plt.show()""")

# ------------------------------------------------------------------
# Section L — Per-day private UMAP
# ------------------------------------------------------------------
md("""## 11. Per-day private latents — UMAP

Each day has its own private latent. We expect each private space to
capture day-specific repair biology rather than cell-type identity, so
colouring by cell type should yield **less structured** UMAPs than the
shared latent above.
""")

code("""group_adatas = {}
for g, d in enumerate(day_order):
    mask = adata_spv.obs["groups"] == d
    sub = adata_spv[mask].copy()
    sub.obsm["X_spVIPES_private"] = latents["private_reordered"][g]
    sc.pp.neighbors(sub, use_rep="X_spVIPES_private",
                    key_added="spvipes_private", n_neighbors=15)
    sc.tl.umap(sub, neighbors_key="spvipes_private", min_dist=0.3)
    sub.obsm["X_umap_private"] = sub.obsm["X_umap"].copy()
    group_adatas[d] = sub

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, d in zip(axes, day_order):
    sc.pl.embedding(
        group_adatas[d], basis="X_umap_private", color=LABEL_COL,
        ax=ax, show=False, title=f"Private latent — {d}",
        legend_loc=None, size=10,
    )
sns.despine()
plt.tight_layout()
plt.show()""")

# ------------------------------------------------------------------
# Section M — Quantitative metrics
# ------------------------------------------------------------------
md("""## 12. Quantitative integration metrics

Two complementary numbers on the shared latent:

- **ARI** between Leiden clusters on the shared embedding and the
  ground-truth cell-type labels — measures **biological conservation**
  (higher is better).
- **Batch-mixing entropy** of the day labels in k-NN balls on the
  shared embedding, normalised by ``log2(N_days)`` — measures
  **integration** (1.0 = perfect mixing).

A good shared latent has both numbers high simultaneously.
""")

code("""from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

def leiden_ari(rep, labels, resolution=0.8):
    tmp = ad.AnnData(X=rep)
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15)
    sc.tl.leiden(tmp, resolution=resolution, random_state=0)
    return adjusted_rand_score(labels, tmp.obs["leiden"].values)

def batch_entropy(rep, group_obs, k=30):
    group_obs = np.asarray(group_obs)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep)
    _, idx = nn.kneighbors(rep)
    idx = idx[:, 1:]
    n_groups = len(np.unique(group_obs))
    max_entropy = np.log2(n_groups)
    entropies = []
    for i in range(rep.shape[0]):
        neigh = group_obs[idx[i]]
        _, counts = np.unique(neigh, return_counts=True)
        p = counts / counts.sum()
        entropies.append(-np.sum(p * np.log2(p + 1e-12)) / max_entropy)
    return float(np.mean(entropies))

rep = adata_spv.obsm["X_spVIPES_shared"]
ari = leiden_ari(rep, adata_spv.obs[LABEL_COL].values)
mix = batch_entropy(rep, adata_spv.obs["day"].values, k=30)

print(f"Shared latent — ARI (cell types)        : {ari:.3f}")
print(f"Shared latent — batch mixing (day, ↑)   : {mix:.3f}")""")

# ------------------------------------------------------------------
# Section N — Wrap-up
# ------------------------------------------------------------------
md("""## 13. Summary

We re-ran the spVIPES IRI experiment from the original paper with a
**three-group day split** (1d / 3d / 14d) instead of the binary IRI
short / long stratification. The **shared latent** integrates cells
across the three repair days while preserving cell-type identity, and
each per-day **private latent** isolates the day-specific repair
programme.

**Try next:**

- Increase `N_PER_GROUP` and `N_HVG` and rerun on GPU.
- Inspect `model.get_loadings()` to see which genes drive each shared
  / private latent dimension at each day.
- Swap in the Neural Spline Flow prior (`use_nf_prior=True`,
  `nf_target="shared"`) and compare ARI / mixing to the Gaussian
  baseline above.
""")

# ------------------------------------------------------------------
# Build the notebook JSON
# ------------------------------------------------------------------
def _to_source_lines(src: str) -> list:
    """Split source into a list of strings, each ending in '\\n' except the last."""
    lines = src.split("\n")
    out = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


for c in cells:
    c["source"] = _to_source_lines(c["source"])

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("docs/notebooks/iri_days_vignette.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {out_path}  ({len(cells)} cells)")
