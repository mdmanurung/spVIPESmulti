"""Smoke-test the distinct spVIPES API combinations exercised by the vignettes.

Each vignette in `docs/notebooks/` exercises a particular combination of:

  - groups (2 vs N>=2)
  - PoE strategy (label / OT-cluster / OT-paired)
  - prior (Gaussian / NSF on shared / NSF on private)
  - disentanglement preset (off / full)
  - modality (single / multimodal)

Re-running every notebook here is impractical because:
  - 3 vignettes (cinemaot_nf, dialogue_multigroup, disentangle_ablation) require
    pertpy 1.x, which pulls in jax >= 0.6.1 and conflicts with spVIPES' jax==0.4.27 pin.
  - 3 vignettes need local h5ad files not bundled in the repo (Tutorial splatter
    simulation, IRI time-course, Plasmodium liver-stage).

Instead we run each *distinct combination* on a tiny SLN CITE-seq subsample and
report PASS/FAIL. Vignettes mapped to each combination are listed in `MAPPING`.

Usage:
    python scripts/smoke_vignettes.py [--epochs N] [--cells_per_group N]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from time import perf_counter

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scvi
import torch

import spVIPES

# Vignette --> combination mapping for the report.
MAPPING = {
    "single-modality / 2-group / label PoE / off": [
        "Tutorial.ipynb (without OT plan)",
    ],
    "single-modality / 2-group / OT cluster-based PoE / off": [
        "Tutorial.ipynb",
    ],
    "single-modality / 2-group / NSF prior on shared / off": [
        "cinemaot_nf_vignette.ipynb (Gaussian vs NSF)",
    ],
    "single-modality / 2-group / NSF prior on shared / full": [
        "cinemaot_nf_vignette.ipynb (NSF + disentangle)",
        "biolord_comparison_plasmodium_tutorial.ipynb (with disentangle)",
    ],
    "single-modality / 3-group / label PoE / off": [
        "dialogue_multigroup_vignette.ipynb",
        "iri_days_vignette.ipynb",
        "pbmc_citeseq_tutorial.ipynb",
    ],
    "single-modality / 3-group / label PoE / full": [
        "dialogue_multigroup_vignette.ipynb (disentangle)",
        "disentangle_ablation.ipynb",
        "iri_days_vignette.ipynb (disentangle)",
        "pbmc_citeseq_tutorial.ipynb (disentangle)",
    ],
    "multimodal / 3-group / NSF prior on shared / off": [
        "multimodal_nf_tutorial.ipynb",
    ],
    "multimodal / 3-group / disentangle=full": [
        "P8: multimodal + disentanglement objective",
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--cells_per_group", type=int, default=300)
    p.add_argument("--n_hvg", type=int, default=600)
    return p.parse_args()


def set_seeds(s=0):
    np.random.seed(s)
    torch.manual_seed(s)


def load_sln(remove_outliers=True):
    a = scvi.data.spleen_lymph_cite_seq(save_path="./data/", remove_outliers=remove_outliers)
    a.obs_names_make_unique()
    a.obs["donor"] = a.obs["batch"].astype(str).map(lambda b: "SLN111" if "111" in b else "SLN208")
    a.obs["donor_tissue"] = a.obs["donor"].astype(str) + "_" + a.obs["tissue"].astype(str)
    return a


def subsample_two_group(adata, n_per_group, n_hvg):
    """Donor as 2 groups, drop rare cell types, top-N_HVG by variance."""
    counts = adata.obs.groupby(["donor", "cell_types"], observed=True).size().unstack("donor", fill_value=0)
    keep_types = counts[(counts >= 25).all(axis=1)].index.tolist()
    sub = adata[adata.obs["cell_types"].isin(keep_types)].copy()
    sub.obs["cell_types"] = sub.obs["cell_types"].astype("category")

    rng = np.random.default_rng(0)
    keep = []
    for g in sub.obs["donor"].unique():
        pos = np.where(sub.obs["donor"].values == g)[0]
        keep.extend(rng.choice(pos, size=min(n_per_group, len(pos)), replace=False))
    sub = sub[np.array(sorted(keep))].copy()
    sub.obs_names_make_unique()

    X = sub.X
    if sp.issparse(X):
        gene_var = np.asarray(X.power(2).mean(0)).ravel() - np.asarray(X.mean(0)).ravel() ** 2
    else:
        gene_var = X.var(0)
    sub = sub[:, np.argsort(gene_var)[::-1][:n_hvg]].copy()
    return sub


def subsample_three_group(adata, n_per_group, n_hvg):
    """3 groups by donor x tissue, dropping the SLN208 x Lymph_Node combination."""
    valid = ["SLN111_Spleen", "SLN111_Lymph_Node", "SLN208_Spleen"]
    sub = adata[adata.obs["donor_tissue"].isin(valid)].copy()
    sub.obs["donor_tissue"] = sub.obs["donor_tissue"].astype("category")

    counts = sub.obs.groupby(["donor_tissue", "cell_types"], observed=True).size().unstack("donor_tissue", fill_value=0)
    keep_types = counts[(counts >= 25).any(axis=1)].index.tolist()
    sub = sub[sub.obs["cell_types"].isin(keep_types)].copy()
    sub.obs["cell_types"] = sub.obs["cell_types"].astype("category")

    rng = np.random.default_rng(0)
    keep = []
    for g in sub.obs["donor_tissue"].unique():
        pos = np.where(sub.obs["donor_tissue"].values == g)[0]
        keep.extend(rng.choice(pos, size=min(n_per_group, len(pos)), replace=False))
    sub = sub[np.array(sorted(keep))].copy()
    sub.obs_names_make_unique()

    X = sub.X
    if sp.issparse(X):
        gene_var = np.asarray(X.power(2).mean(0)).ravel() - np.asarray(X.mean(0)).ravel() ** 2
    else:
        gene_var = X.var(0)
    sub = sub[:, np.argsort(gene_var)[::-1][:n_hvg]].copy()
    return sub


def make_groups_dict(adata, group_col):
    return {str(g): adata[adata.obs[group_col] == g].copy() for g in sorted(adata.obs[group_col].unique())}


def split_modalities(adata):
    """Split SLN AnnData into RNA + protein per (group_value)."""
    out = {}
    for g in sorted(adata.obs["donor_tissue"].unique()):
        sub = adata[adata.obs["donor_tissue"] == g].copy()
        rna = sub.copy()
        rna.uns = {}
        rna.obsm = {}
        prot_X = sub.obsm["protein_expression"].values.astype(np.float32)
        prot_var = pd.DataFrame(index=sub.obsm["protein_expression"].columns)
        prot = ad.AnnData(X=prot_X, obs=sub.obs.copy(), var=prot_var)
        out[str(g)] = {"rna": rna, "protein": prot}
    return out


def build_and_train(prepared, *, epochs, batch_size, **model_kwargs):
    set_seeds(0)
    model = spVIPES.model.spVIPES(prepared, n_hidden=64, n_dimensions_shared=12,
                                  n_dimensions_private=6, dropout_rate=0.1,
                                  **model_kwargs)
    gi = [list(map(int, g)) for g in prepared.uns["groups_obs_indices"]]
    model.train(
        group_indices_list=gi, max_epochs=epochs, batch_size=batch_size,
        train_size=0.85, n_epochs_kl_warmup=min(3, epochs),
    )
    latents = model.get_latent_representation(group_indices_list=gi, batch_size=batch_size)
    n_per_group = [len(g) for g in gi]
    assert sum(latents["shared"][i].shape[0] for i in range(len(gi))) == sum(n_per_group), \
        "Shared latent shape doesn't match group cell counts."
    return model


# ----------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------

def case_single_2g_label_off(adata2, args):
    prepared = spVIPES.data.prepare_adatas(make_groups_dict(adata2, "donor"))
    spVIPES.model.spVIPES.setup_anndata(prepared, groups_key="groups", label_key="cell_types")
    build_and_train(prepared, epochs=args.epochs, batch_size=128, disentangle_preset="off")


def case_single_2g_otcluster_off(adata2, args):
    """OT cluster-based PoE on 2 groups. Build a tiny transport plan from cluster centroids."""
    sc_kwargs = dict(use_rep="X_pca")
    sc_adata = adata2.copy()
    import scanpy as sc
    sc.pp.normalize_total(sc_adata, target_sum=1e4)
    sc.pp.log1p(sc_adata)
    sc.tl.pca(sc_adata, n_comps=20)
    # Trivial uniform transport plan between donor-A and donor-B cells.
    a_idx = np.where(sc_adata.obs["donor"].values == "SLN111")[0]
    b_idx = np.where(sc_adata.obs["donor"].values == "SLN208")[0]
    transport = np.full((len(a_idx), len(b_idx)), 1.0 / (len(a_idx) * len(b_idx)), dtype=np.float32)

    groups = make_groups_dict(adata2, "donor")
    prepared = spVIPES.data.prepare_adatas(groups)
    prepared.uns["transport_plan"] = transport
    spVIPES.model.spVIPES.setup_anndata(
        prepared, groups_key="groups", transport_plan_key="transport_plan",
        match_clusters=True,
    )
    build_and_train(prepared, epochs=args.epochs, batch_size=128, disentangle_preset="off")


def case_single_2g_nsf_off(adata2, args):
    prepared = spVIPES.data.prepare_adatas(make_groups_dict(adata2, "donor"))
    spVIPES.model.spVIPES.setup_anndata(prepared, groups_key="groups", label_key="cell_types")
    build_and_train(prepared, epochs=args.epochs, batch_size=128,
                    use_nf_prior=True, nf_type="NSF", nf_transforms=2, nf_target="shared",
                    disentangle_preset="off")


def case_single_2g_nsf_full(adata2, args):
    prepared = spVIPES.data.prepare_adatas(make_groups_dict(adata2, "donor"))
    spVIPES.model.spVIPES.setup_anndata(prepared, groups_key="groups", label_key="cell_types")
    build_and_train(prepared, epochs=args.epochs, batch_size=128,
                    use_nf_prior=True, nf_type="NSF", nf_transforms=2, nf_target="shared",
                    disentangle_preset="full")


def case_single_3g_label_off(adata3, args):
    prepared = spVIPES.data.prepare_adatas(make_groups_dict(adata3, "donor_tissue"))
    spVIPES.model.spVIPES.setup_anndata(prepared, groups_key="groups", label_key="cell_types")
    build_and_train(prepared, epochs=args.epochs, batch_size=128, disentangle_preset="off")


def case_single_3g_label_full(adata3, args):
    prepared = spVIPES.data.prepare_adatas(make_groups_dict(adata3, "donor_tissue"))
    spVIPES.model.spVIPES.setup_anndata(prepared, groups_key="groups", label_key="cell_types")
    build_and_train(prepared, epochs=args.epochs, batch_size=128, disentangle_preset="full")


def case_multimodal_3g_nsf_off(adata3, args):
    adatas_dict = split_modalities(adata3)
    prepared = spVIPES.data.prepare_multimodal_adatas(
        adatas_dict, modality_likelihoods={"rna": "nb", "protein": "nb"}
    )
    spVIPES.model.spVIPES.setup_anndata(
        prepared, groups_key="groups", label_key="cell_types",
        modality_likelihoods={"rna": "nb", "protein": "nb"},
    )
    build_and_train(prepared, epochs=args.epochs, batch_size=128,
                    use_nf_prior=True, nf_type="NSF", nf_transforms=2, nf_target="shared",
                    disentangle_preset="off")


def case_multimodal_3g_disentangle_full(adata3, args):
    """P8: multimodal + disentangle_preset='full'. Trains end-to-end."""
    adatas_dict = split_modalities(adata3)
    prepared = spVIPES.data.prepare_multimodal_adatas(
        adatas_dict, modality_likelihoods={"rna": "nb", "protein": "nb"}
    )
    spVIPES.model.spVIPES.setup_anndata(
        prepared, groups_key="groups", label_key="cell_types",
        modality_likelihoods={"rna": "nb", "protein": "nb"},
    )
    build_and_train(prepared, epochs=args.epochs, batch_size=128,
                    disentangle_preset="full")


CASES = [
    ("single-modality / 2-group / label PoE / off", case_single_2g_label_off, "two_group"),
    ("single-modality / 2-group / OT cluster-based PoE / off", case_single_2g_otcluster_off, "two_group"),
    ("single-modality / 2-group / NSF prior on shared / off", case_single_2g_nsf_off, "two_group"),
    ("single-modality / 2-group / NSF prior on shared / full", case_single_2g_nsf_full, "two_group"),
    ("single-modality / 3-group / label PoE / off", case_single_3g_label_off, "three_group"),
    ("single-modality / 3-group / label PoE / full", case_single_3g_label_full, "three_group"),
    ("multimodal / 3-group / NSF prior on shared / off", case_multimodal_3g_nsf_off, "three_group"),
    ("multimodal / 3-group / disentangle=full", case_multimodal_3g_disentangle_full, "three_group"),
]


def main():
    args = parse_args()
    print("=" * 78)
    print(" spVIPES vignette smoke tests")
    print("=" * 78)
    print(f"  epochs={args.epochs}  cells/group={args.cells_per_group}  HVGs={args.n_hvg}")

    adata = load_sln()
    print(f"\nFull SLN:  {adata.shape}")
    adata2 = subsample_two_group(adata, args.cells_per_group, args.n_hvg)
    adata3 = subsample_three_group(adata, args.cells_per_group, args.n_hvg)
    print(f"  2-group subset:  {adata2.shape}")
    print(f"  3-group subset:  {adata3.shape}")

    summary = []
    for name, fn, kind in CASES:
        adata_use = adata2 if kind == "two_group" else adata3
        t0 = perf_counter()
        try:
            fn(adata_use, args)
            status = "PASS"
            err = ""
        except Exception as e:
            status = "FAIL"
            err = f"{type(e).__name__}: {e}"
            import traceback; traceback.print_exc()
        secs = round(perf_counter() - t0, 1)
        summary.append({"case": name, "status": status, "secs": secs, "error": err})
        print(f"  [{status}]  {secs:6.1f}s   {name}")
        if err:
            print(f"          {err}")

    print("\n" + "=" * 78)
    print(" Vignette mapping (which notebooks each case represents)")
    print("=" * 78)
    for case_name, vignettes in MAPPING.items():
        print(f"  - {case_name}")
        for v in vignettes:
            print(f"      <- {v}")

    df = pd.DataFrame(summary)
    df.to_csv(Path(__file__).resolve().parent / "smoke_vignettes_results.csv", index=False)
    n_pass = (df["status"] == "PASS").sum()
    print(f"\n {n_pass}/{len(df)} cases passed.")
    sys.exit(0 if n_pass == len(df) else 1)


if __name__ == "__main__":
    main()
