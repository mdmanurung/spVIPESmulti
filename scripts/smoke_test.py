"""Smoke test: run full spVIPESmulti multimodal + NF-prior path on a tiny subsample."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import anndata as ad
import spVIPESmulti

np.random.seed(0)
torch.manual_seed(0)

# Load
adata_all = scvi.data.spleen_lymph_cite_seq(save_path="./data/", remove_outliers=True)
print("Loaded:", adata_all.shape, "proteins:", adata_all.obsm["protein_expression"].shape[1])

# Build a stable 3-group label
def g_label(row):
    b = row["batch"]
    t = row["tissue"]
    donor = "SLN111" if "111" in b else "SLN208"
    # Three groups: SLN111_Spleen, SLN111_LymphNode, SLN208_Spleen
    # (drop SLN208_LymphNode so one label is missing from one group — exercises _label_based_poe branch)
    if donor == "SLN111" and t == "Spleen":
        return "SLN111_Spleen"
    if donor == "SLN111" and t == "Lymph_Node":
        return "SLN111_LymphNode"
    if donor == "SLN208" and t == "Spleen":
        return "SLN208_Spleen"
    return None

adata_all.obs["group"] = adata_all.obs.apply(g_label, axis=1)
adata_all = adata_all[~adata_all.obs["group"].isna()].copy()
print("After group filter:", adata_all.shape)
print(adata_all.obs["group"].value_counts().to_dict())

# Tiny subsample: 300 cells per group (positional indices to avoid non-unique obs_names)
adata_all.obs_names_make_unique()
rng = np.random.default_rng(0)
pos_keep = []
obs_group = adata_all.obs["group"].values
for g in np.unique(obs_group):
    pos = np.where(obs_group == g)[0]
    pick = rng.choice(pos, size=min(300, len(pos)), replace=False)
    pos_keep.extend(pick)
pos_keep = np.array(sorted(pos_keep))
adata_small = adata_all[pos_keep].copy()
adata_small.obs_names_make_unique()
print("After subsample:", adata_small.shape)

# HVG: select top-500 genes by variance of raw counts (simple + robust on tiny subsamples)
import scipy.sparse as sp
X = adata_small.X
if sp.issparse(X):
    gene_var = np.asarray(X.power(2).mean(axis=0)).flatten() - np.asarray(X.mean(axis=0)).flatten()**2
else:
    gene_var = X.var(axis=0)
top = np.argsort(gene_var)[::-1][:500]
adata_small = adata_small[:, top].copy()
print("After HVG:", adata_small.shape)

# Build per-group, per-modality dict
groups = sorted(adata_small.obs["group"].unique())
adatas_dict = {}
for g in groups:
    sub = adata_small[adata_small.obs["group"] == g].copy()
    rna_raw = sub.copy()
    rna_raw.X = sub.X.copy()
    prot_X = sub.obsm["protein_expression"].values.astype(np.float32)
    prot_var = pd.DataFrame(index=sub.obsm["protein_expression"].columns)
    prot = ad.AnnData(X=prot_X, obs=sub.obs.copy(), var=prot_var)
    adatas_dict[g] = {"rna": rna_raw, "protein": prot}
    print(f"  {g}: rna={rna_raw.shape}, prot={prot.shape}")

# prepare_multimodal_adatas
adata = spVIPESmulti.data.prepare_multimodal_adatas(
    adatas_dict,
    modality_likelihoods={"rna": "nb", "protein": "nb"},
)
print("Prepared:", adata.shape)
print("uns keys:", [k for k in adata.uns if "group" in k or "modality" in k or "is_" in k])
print("groups_modality_lengths:", adata.uns["groups_modality_lengths"])
print("modality_names:", adata.uns["modality_names"])

# setup_anndata  (label-based PoE path)
spVIPESmulti.model.spVIPESmulti.setup_anndata(
    adata,
    groups_key="groups",
    label_key="cell_types",
    modality_likelihoods={"rna": "nb", "protein": "nb"},
)

# Build model — baseline Gaussian prior
model = spVIPESmulti.model.spVIPESmulti(
    adata,
    n_hidden=64,
    n_dimensions_shared=12,
    n_dimensions_private=6,
    dropout_rate=0.1,
    use_nf_prior=False,
)
print(model)

# Build group_indices_list
group_idx_list = [list(g) for g in adata.uns["groups_obs_indices"]]
print("group sizes:", [len(x) for x in group_idx_list])

# Train for a handful of epochs
model.train(
    group_indices_list=group_idx_list,
    max_epochs=5,
    batch_size=128,
    train_size=0.9,
    early_stopping=False,
    n_epochs_kl_warmup=5,
)
print("Baseline train OK")

# Get latents
latents = model.get_latent_representation(group_indices_list=group_idx_list, batch_size=128)
print("latent keys:", list(latents.keys()))
for k in ("shared", "private"):
    for g, arr in latents[k].items():
        print(f"  {k}[{g}]: shape={arr.shape}")
if "private_multimodal" in latents:
    for (g, mod), arr in latents["private_multimodal"].items():
        print(f"  pm[{g},{mod}]: shape={arr.shape}")

# Now NF prior
model_nf = spVIPESmulti.model.spVIPESmulti(
    adata,
    n_hidden=64,
    n_dimensions_shared=12,
    n_dimensions_private=6,
    dropout_rate=0.1,
    use_nf_prior=True,
    nf_type="NSF",
    nf_transforms=3,
    nf_target="shared",
)
model_nf.train(
    group_indices_list=group_idx_list,
    max_epochs=5,
    batch_size=128,
    train_size=0.9,
    early_stopping=False,
    n_epochs_kl_warmup=5,
)
print("NF train OK")

print("\n=== SMOKE TEST PASSED ===")
