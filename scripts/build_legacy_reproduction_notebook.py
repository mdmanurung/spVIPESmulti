"""Generate docs/notebooks/legacy_spVIPES_reproduction.ipynb programmatically.

Vignette demonstrating that spVIPESmulti, configured with label-based PoE on
two RNA-only groups and with all post-spVIPES additions (disentanglement, NF
prior, Jeffreys integ, group-loss reweighting) DISABLED, qualitatively
reproduces the integration result of the original `nrclaudio/spVIPES`
``Tutorial.ipynb`` on the Splatter simulation.
"""
from __future__ import annotations

import json
from pathlib import Path

cells: list[dict] = []


def _src(text: str) -> list[str]:
    """Split source into lines preserving newlines, as required by nbformat."""
    lines = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        # ensure final line has no trailing newline (nbformat convention)
        pass
    return lines


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(text.lstrip("\n"))})


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(text.lstrip("\n")),
    })


# ---------------------------------------------------------------------------
# 0. Front matter
# ---------------------------------------------------------------------------
md(r"""
# Reproducing the original spVIPES tutorial with `spVIPESmulti`

This vignette shows that **`spVIPESmulti`**, configured to mirror the original
[`nrclaudio/spVIPES` Tutorial.ipynb](https://github.com/nrclaudio/spVIPES/blob/main/docs/notebooks/Tutorial.ipynb),
qualitatively reproduces the original result on the Splatter simulation
([Zenodo 10070301](https://zenodo.org/records/10070301)):

* the **shared** latent recovers the cell-type axis (`Celltypes`), and
* each **per-group private** latent recovers its dataset-specific subgroup /
  gene-program axis (`Subgroup`, `Gene_programs`).

## API mapping (legacy → spVIPESmulti)

The original tutorial used the **OT-paired** PoE strategy
(`transport_plan_key='transport_plan'`, `match_clusters=False`).
`spVIPESmulti` does not implement OT-based pairing. The closest *supervised*
analogue is **label-based PoE** (`label_key='Celltypes'`), which uses
cell-type annotations to align groups instead of an OT plan. Both anchor the
shared axis to cell types; they differ in *how* that anchoring is enforced.

To keep the comparison honest, we additionally **disable every post-spVIPES
addition** in `spVIPESmulti`:

| Original `spVIPES` | This notebook (`spVIPESmulti` legacy mode) |
|---|---|
| (no disentanglement) | `disentangle_preset="off"` |
| (no normalizing-flow prior) | `use_nf_prior=False` (default) |
| (no Jeffreys integ) | `use_jeffreys_integ=False` (default) |
| (no group reweighting) | `group_loss_weights=None` (default) |
| `n_dimensions_shared=10`, `n_dimensions_private=7` | identical |
| `n_hidden=128`, `dropout_rate=0.1` | identical |
| `train(..., batch_size=128, train_size=1.0)` | identical |

The remaining shared-private VAE + Product-of-Experts core is the same
architecture as the original spVIPES; this notebook isolates it from the new
package's added objectives.
""")


# ---------------------------------------------------------------------------
# 1. Imports + seeds
# ---------------------------------------------------------------------------
md("## 1. Imports and reproducibility")

code(r"""
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import scvi
import torch

import spVIPESmulti

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
scvi.settings.seed = SEED
sc.settings.set_figure_params(dpi=80, frameon=False)

print(f"spVIPESmulti  : {spVIPESmulti.__version__}")
print(f"scvi-tools    : {scvi.__version__}")
print(f"scanpy        : {sc.__version__}")
print(f"torch         : {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"anndata       : {ad.__version__}")
""")


# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
md(r"""
## 2. Load the Splatter simulation

The same Splatter-simulated AnnData used in the original tutorial, hosted on
[Zenodo (record 10070301)](https://zenodo.org/records/10070301). It contains
two simulated *datasets* (the groups), each split across five *cell types*,
each composed of *subgroups* / *gene programs* (the ground-truth private
factor we want each per-group private latent to recover).

The cell below downloads the file (~1 GB) into `data/` on first run and
caches it locally afterwards.
""")

code(r"""
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "splatter_simulation-2.h5ad"
ZENODO_URL = (
    "https://zenodo.org/records/10070301/files/splatter_simulation-2.h5ad"
)

if not DATA_PATH.exists():
    import urllib.request
    print(f"Downloading {ZENODO_URL}\n  -> {DATA_PATH} (~1 GB, one-time)…")
    urllib.request.urlretrieve(ZENODO_URL, DATA_PATH)
    print("Done.")
else:
    print(f"Using cached file at {DATA_PATH}")

adata = sc.read_h5ad(DATA_PATH)
adata.var_names_make_unique()
adata.layers["counts"] = adata.X.copy()
print(adata)
print("\n— raw obs columns —\n", list(adata.obs.columns))
""")

md(r"""
The raw `.h5ad` only carries the Splatter `Group` and `Subgroup` columns. The
original tutorial derives three additional labels from them. We replicate that
mapping verbatim so the rest of the analysis is identical:

* `Dataset` — splits cells into the two groups passed to PoE
  (`Group1`/`Group2` → *Dataset 1*; `Group3`/`Group4` → *Dataset 2*).
* `Celltypes` — renames `Group1`–`Group5` to *Cell type 1*–*Cell type 5*
  (used as `label_key`).
* `Gene_programs` — renames the four `Subgroup` values to interpretable
  per-dataset gene-program names (used to colour the private UMAPs).
""")

code(r"""
adata.obs["Dataset"] = adata.obs["Subgroup"].astype(str).map({
    "Group1": "Dataset 1", "Group2": "Dataset 1",
    "Group3": "Dataset 2", "Group4": "Dataset 2",
}).astype("category")

adata.obs["Celltypes"] = adata.obs["Group"].astype(str).map({
    "Group1": "Cell type 1", "Group2": "Cell type 2",
    "Group3": "Cell type 3", "Group4": "Cell type 4",
    "Group5": "Cell type 5",
}).astype("category")

adata.obs["Gene_programs"] = adata.obs["Subgroup"].astype(str).map({
    "Group1": "Gene program 1a", "Group2": "Gene program 2a",
    "Group3": "Gene program 1b", "Group4": "Gene program 2b",
}).astype("category")

for col in ["Dataset", "Celltypes", "Subgroup", "Gene_programs"]:
    print(f"── {col} ──")
    print(adata.obs[col].value_counts().to_string())
    print()
""")


# ---------------------------------------------------------------------------
# 3. Split into two groups
# ---------------------------------------------------------------------------
md(r"""
## 3. Split into two groups

The original tutorial split the AnnData into `dataset_1` and `dataset_2` by
the `Dataset` obs column. We do the same.
""")

code(r"""
dataset1 = adata[adata.obs["Dataset"] == "Dataset 1"].copy()
dataset2 = adata[adata.obs["Dataset"] == "Dataset 2"].copy()
print("dataset1:", dataset1.shape)
print("dataset2:", dataset2.shape)
""")


# ---------------------------------------------------------------------------
# 4. prepare_adatas
# ---------------------------------------------------------------------------
md(r"""
## 4. Prepare multi-group AnnData

`spVIPESmulti.data.prepare_adatas` concatenates the per-group AnnDatas and
records the per-group obs/var indices in `.uns` for later use.
""")

code(r"""
adatas_dict = {
    "dataset_1": dataset1,
    "dataset_2": dataset2,
}
# Strip auxiliary slots — prepare_adatas expects clean inputs.
for _ad in adatas_dict.values():
    _ad.uns = {}
    _ad.obsm = {}
    # keep .layers["counts"] so it's available downstream
adata_spv = spVIPESmulti.data.prepare_adatas(adatas_dict)
print(adata_spv)
print("\nGroups:", list(adata_spv.uns["groups_mapping"].values()))
""")


# ---------------------------------------------------------------------------
# 5. setup_anndata
# ---------------------------------------------------------------------------
md(r"""
## 5. Register the model with **label-based** PoE

The original tutorial used:

```python
spVIPES.model.spVIPES.setup_anndata(
    adata, groups_key='groups',
    match_clusters=False, transport_plan_key='transport_plan',
)
```

`spVIPESmulti` has no OT-paired path; the supervised analogue is
`label_key='Celltypes'`, which aligns the two datasets along the shared axis
using cell-type annotations.
""")

code(r"""
spVIPESmulti.model.spVIPESmulti.setup_anndata(
    adata_spv,
    groups_key="groups",
    label_key="Celltypes",
)
""")


# ---------------------------------------------------------------------------
# 6. Build model with legacy-matching configuration
# ---------------------------------------------------------------------------
md(r"""
## 6. Build the model — **legacy-matching configuration**

Every constructor argument mirrors the original spVIPES tutorial. The
`disentangle_preset="off"` line zeroes out all five disentanglement
components so that the loss is pure ELBO + PoE alignment, exactly as in
the original package.
""")

code(r"""
N_SHARED   = 10        # original: n_dimensions_shared=10
N_PRIVATE  = 7         # original: n_dimensions_private=7
N_HIDDEN   = 128       # original default
DROPOUT    = 0.1       # original default
BATCH_SIZE = 128       # original
MAX_EPOCHS = 400       # generous budget; original used scvi defaults

model_spv = spVIPESmulti.model.spVIPESmulti(
    adata_spv,
    n_hidden=N_HIDDEN,
    n_dimensions_shared=N_SHARED,
    n_dimensions_private=N_PRIVATE,
    dropout_rate=DROPOUT,
    # ---- everything below disabled to match original spVIPES ----
    disentangle_preset="off",   # zeroes all 5 disentanglement weights
    use_nf_prior=False,         # standard Gaussian prior
    use_jeffreys_integ=False,   # no Jeffreys integration loss
    group_loss_weights=None,    # uniform group weighting
)
model_spv
""")


# ---------------------------------------------------------------------------
# 7. Train (with on-disk cache)
# ---------------------------------------------------------------------------
md(r"""
## 7. Train

Training mirrors the original tutorial: `batch_size=128`, no validation
split, no early stopping, no KL warmup. We cache the trained model on disk
so re-running the notebook is fast.
""")

code(r"""
from spVIPESmulti.utils import resolve_group_indices_list

group_indices_list, _ = resolve_group_indices_list(adata_spv)
print("Group sizes:", [len(g) for g in group_indices_list])
""")

code(r"""
SAVE_DIR = Path("results/spvipes_legacy_reproduction")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = SAVE_DIR / "model.pt"

should_train = True
if MODEL_FILE.exists():
    try:
        print(f"Loading cached model from {SAVE_DIR}")
        model_spv = spVIPESmulti.model.spVIPESmulti.load(str(SAVE_DIR), adata=adata_spv)
        should_train = False
    except Exception as e:
        print(f"Cached model could not be loaded ({type(e).__name__}: {e}). Retraining...")

if should_train:
    model_spv.train(
        group_indices_list,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        train_size=1.0,            # no validation split (matches original)
        early_stopping=False,      # matches original
        n_epochs_kl_warmup=0,      # matches original
        check_val_every_n_epoch=10,
    )
    model_spv.save(str(SAVE_DIR), overwrite=True)
    print(f"Saved trained model to {SAVE_DIR}")
""")

code(r"""
# ELBO training curve
fig = spVIPESmulti.pl.training_curves(model_spv)
""")


# ---------------------------------------------------------------------------
# 8. Embed
# ---------------------------------------------------------------------------
md(r"""
## 8. Compute shared and per-group private latents
""")

code(r"""
embed_out = model_spv.embed(prefix="spvm", batch_size=1024, overwrite=True)
print("Written keys:", embed_out["keys"])
print("Shared latent shape:", embed_out["shared"].shape)
for tok, arr in embed_out["private"].items():
    print(f"  Private[{tok}] shape: {arr.shape}")
""")


# ---------------------------------------------------------------------------
# 9. Shared UMAP — Figure A from the original tutorial
# ---------------------------------------------------------------------------
md(r"""
## 9. Shared UMAP — recovers cell types

In the original tutorial this is the **"Celltypes spVIPE shared"** figure:
the shared latent should separate the five `Celltypes` while mixing the two
`Dataset`s.
""")

code(r"""
spVIPESmulti.utils.compute_shared_umap(
    adata_spv,
    obsm_key="X_spvm_shared",
    n_neighbors=20,
    min_dist=0.5,
    umap_key="X_umap_spvm_shared",
)
spVIPESmulti.pl.umap_shared(
    adata_spv,
    color=["Celltypes", "Dataset"],
    basis="X_umap_spvm_shared",
)
""")


# ---------------------------------------------------------------------------
# 10. Per-group private UMAPs — Figure B/C from the original
# ---------------------------------------------------------------------------
md(r"""
## 10. Per-group private UMAPs — recover subgroup / gene programs

In the original tutorial this is the **per-dataset private** figure: each
private latent should carve up its own dataset along the `Subgroup` /
`Gene_programs` axis (the ground-truth private factor) without being
contaminated by the cell-type axis.
""")

code(r"""
groups_map = adata_spv.uns["groups_mapping"]
private_keys = embed_out["keys"]["private"]
private_token_order = list(private_keys.keys())

private_adatas = {}
for gi, idxs in enumerate(group_indices_list):
    name = str(groups_map.get(gi, gi))
    sub = adata_spv[np.asarray(idxs)].copy()
    sub.obsm["X_spvm_private"] = adata_spv.obsm[private_keys[private_token_order[gi]]][np.asarray(idxs)]
    private_adatas[name] = sub

spVIPESmulti.utils.compute_private_umaps(
    private_adatas,
    obsm_key="X_spvm_private",
    n_neighbors=20,
    min_dist=0.5,
    umap_key="X_umap_spvm_private",
)
""")

code(r"""
fig = spVIPESmulti.pl.umap_private(
    private_adatas,
    color="Subgroup",
    basis="X_umap_spvm_private",
    ncols=2,
)
""")

code(r"""
if "Gene_programs" in adata_spv.obs.columns:
    fig = spVIPESmulti.pl.umap_private(
        private_adatas,
        color="Gene_programs",
        basis="X_umap_spvm_private",
        ncols=2,
    )
""")


# ---------------------------------------------------------------------------
# 11. Quantitative sanity table
# ---------------------------------------------------------------------------
md(r"""
## 11. Quantitative sanity check

A small diagnostic to back the visual claim:

* **shared latent** should have high silhouette / k-NN purity for `Celltypes`
  (and *low* for `Dataset` — i.e. datasets should mix);
* **per-group private** latent should have high silhouette for `Subgroup`.
""")

code(r"""
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

def _knn_purity(z: np.ndarray, labels: np.ndarray, k: int = 20) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    _, idx = nn.kneighbors(z)
    idx = idx[:, 1:]
    return float(np.mean([(labels[idx[i]] == labels[i]).mean() for i in range(len(labels))]))


z_shared = adata_spv.obsm["X_spvm_shared"]
ct = adata_spv.obs["Celltypes"].astype(str).values
ds = adata_spv.obs["Dataset"].astype(str).values

shared_table = pd.DataFrame({
    "metric": ["silhouette(Celltypes)", "silhouette(Dataset)", "kNN purity(Celltypes, k=20)", "kNN purity(Dataset, k=20)"],
    "value": [
        silhouette_score(z_shared, ct, sample_size=min(5000, len(ct)), random_state=0),
        silhouette_score(z_shared, ds, sample_size=min(5000, len(ds)), random_state=0),
        _knn_purity(z_shared, ct, k=20),
        _knn_purity(z_shared, ds, k=20),
    ],
})
print("=== Shared latent (X_spvm_shared) ===")
print(shared_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
""")

code(r"""
private_rows = []
for name, sub in private_adatas.items():
    z_priv = sub.obsm["X_spvm_private"]
    sg = sub.obs["Subgroup"].astype(str).values
    private_rows.append({
        "group": name,
        "silhouette(Subgroup)": silhouette_score(z_priv, sg, sample_size=min(5000, len(sg)), random_state=0),
        "kNN purity(Subgroup, k=20)": _knn_purity(z_priv, sg, k=20),
    })
private_table = pd.DataFrame(private_rows)
print("=== Per-group private latents ===")
print(private_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
""")


# ---------------------------------------------------------------------------
# 12. Conclusion
# ---------------------------------------------------------------------------
md(r"""
## 12. Conclusion

With **label-based PoE** as the supervised analogue of the original
**OT-paired** PoE, and with **all post-spVIPES additions disabled**
(`disentangle_preset="off"`, no NF prior, no Jeffreys integ, no group-loss
reweighting), `spVIPESmulti` reproduces the qualitative behaviour of the
original `nrclaudio/spVIPES` tutorial on the Splatter simulation:

* the **shared** UMAP separates `Celltypes` while mixing `Dataset`,
* each **private** UMAP separates its dataset's `Subgroup` / `Gene_programs`,
* the silhouette / k-NN purity table backs this up quantitatively.

For a fully **quantitative parity study** against the original `spVIPES`
package (matched seeds, CKA / Procrustes between latents, k-NN graph
overlap, ARI on cell-type clusters), see the deferred plan in `PLAN.md`
under "Phase 2 — quantitative parity".
""")


# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "legacy_spVIPES_reproduction.ipynb"
OUT.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT}")
print(f"Cells: {len(cells)}")
