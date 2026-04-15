# Vignette Plan — Multi-group, Multimodal, and NF-Prior Showcase

This document is a step-by-step plan for a new tutorial notebook that
demonstrates the features added to spVIPES on the
`claude/multigroup-multimodal-support-P9rZB` branch (now merged into `main`).

The notebook will live at `docs/notebooks/multimodal_nf_tutorial.ipynb`
alongside the existing `Tutorial.ipynb`, and will be linked from `docs/index.md`.

---

## 1. What we are showcasing

Three feature groups, all opt-in and backward-compatible:

| Feature | Where it lives | How the vignette exercises it |
|---|---|---|
| **N ≥ 2 groups** (previously hard-capped at 2) | `data/prepare_adatas.py`, `module._poe_n`, `module._label_based_poe`, `model.spvipes.get_latent_representation` | Run spVIPES on **3 groups** simultaneously (label-based PoE across donors/tissues). |
| **Multimodal (per-modality encoders/decoders + two-level PoE)** | `data/prepare_multimodal_adatas`, `module._inference_multimodal`, `module._generative_multimodal`, `module._loss_multimodal`, `module.utils.build_likelihood` | RNA (NB) + protein (NB or Gaussian) from CITE-seq, per-(group, modality) encoders, intra-group PoE across modalities, inter-group PoE across groups. |
| **Zuko Normalizing-Flow prior** (NSF / MAF) | `module.spVIPESmodule.__init__`, `_nf_kl`, `loss` | Toggle `use_nf_prior=True, nf_type="NSF", nf_target="shared"` and compare latent quality to the default N(0, I) prior. |

Non-goals: the vignette will **not** re-derive the optimal-transport PoE
variants, since they remain 2-group-only and are covered in the existing
Tutorial.

---

## 2. Dataset selection — scvi-tools built-ins

**Primary choice: `scvi.data.spleen_lymph_cite_seq()`**

Rationale:
- Ships **RNA + protein** in a single AnnData (`X` and `obsm["protein_expression"]`).
- Provides two natural groupings: `obs["batch"]` (`sln_111`, `sln_208`) **and**
  `obs["tissue"]` (`Spleen`, `Lymph Node`). We can combine them into **3+ groups**
  (e.g. `{sln_111_Spleen, sln_111_LymphNode, sln_208_Spleen, sln_208_LymphNode}`).
- Carries `obs["cell_types"]` — needed for the label-based PoE path (which is
  the only PoE path generalized to N ≥ 2 groups).
- Published as the totalVI benchmark (Gayoso et al., *Nat. Methods* 2021), so
  readers can sanity-check our embeddings against a known reference.
- Size (~30–40k cells, ~13k genes, ~40 proteins) is tractable on a laptop GPU
  and fits in CI time budgets.

**Fallback / extension dataset: `scvi.data.pbmcs_10x_cite_seq()`**
- Same `protein_expression` convention; only 2 batches (`PBMC10k` / `PBMC5k`).
- Used in an optional appendix cell as a smaller-scale reproducer (~15k cells).

**Not chosen:** `pbmc_seurat_v4_cite_seq()` — excellent for multi-group
integration (24 batches) but the dataset is large (~160k cells) and protein
counts live under `obsm["protein_counts"]`, not `"protein_expression"`, which
would add friction for the first tutorial. We will mention it at the end as
"scaling up".

---

## 3. Notebook outline

The notebook will have **14 sections** (roughly 35–45 cells). Every code cell
has a target output the author should verify before publishing.

### Section A — Front matter and environment
1. Markdown cell: title, one-paragraph abstract, list of new features (table
   from §1), link to the paper.
2. Markdown cell: runtime note — GPU recommended, expected wall-clock on T4
   (~15 min for 50 epochs on the full dataset; ~3 min if we subsample to 8k
   cells per group, which is what we will do for CI).
3. Code cell: imports — `scanpy`, `scvi`, `numpy`, `pandas`, `torch`,
   `matplotlib`, `seaborn`, `spVIPES`. Print the spVIPES version and whether
   CUDA is available.
4. Code cell: `sc.settings.set_figure_params(dpi=80, frameon=False)` and a
   fixed `torch.manual_seed(0); np.random.seed(0)`.

### Section B — Load and inspect CITE-seq data
5. Code cell: `adata_all = scvi.data.spleen_lymph_cite_seq(save_path="./data/")`.
   Verify: `adata_all.X` is counts, `adata_all.obsm["protein_expression"]` is a
   DataFrame, print `.shape`, `.obs["batch"].value_counts()`,
   `.obs["tissue"].value_counts()`, `.obs["cell_types"].value_counts()`.
6. Markdown cell: explain the two-level grouping (`batch` × `tissue`).

### Section C — Construct the three groups
7. Code cell: create the `group` label by combining tissue and batch into
   **three** groups, e.g.:
    - `SpleenA = sln_111 ∩ Spleen`
    - `LymphA  = sln_111 ∩ LymphNode`
    - `SpleenB = sln_208 ∩ Spleen`
   (Drop `sln_208 ∩ LymphNode` so we end up with exactly three groups, and
   document why: we want to exercise the "one label present in only one
   group" branch of `_label_based_poe`.)
8. Code cell: subsample each group to min(8000, n) cells for reproducible
   runtime. Print the resulting `(n_cells, n_genes, n_proteins)` per group.

### Section D — Per-group, per-modality AnnData construction
9. Markdown cell: explain that `prepare_multimodal_adatas` expects
   `dict[group → dict[modality → AnnData]]`. RNA uses raw counts (NB
   likelihood). Protein counts go into a separate AnnData built from
   `obsm["protein_expression"]`.
10. Code cell: split the loaded AnnData into three per-group AnnData objects.
11. Code cell: for each group, build `{"rna": adata_rna, "protein": adata_prot}`.
    `adata_prot = ad.AnnData(X=adata_group.obsm["protein_expression"].values,
    obs=adata_group.obs.copy(), var=pd.DataFrame(index=<protein names>))`.
    Assert that `adata_rna.n_obs == adata_prot.n_obs` for each group.
12. Code cell: light HVG selection on the RNA side only
    (`sc.pp.highly_variable_genes(flavor="seurat_v3", n_top_genes=2000,
    subset=True)` per group, or — cleaner — on the concatenated RNA). Proteins
    are kept whole.

### Section E — `prepare_multimodal_adatas`
13. Code cell:
    ```python
    adatas_dict = {
        "SpleenA": {"rna": rna_SpleenA, "protein": prot_SpleenA},
        "LymphA":  {"rna": rna_LymphA,  "protein": prot_LymphA},
        "SpleenB": {"rna": rna_SpleenB, "protein": prot_SpleenB},
    }
    adata = spVIPES.data.prepare_multimodal_adatas(
        adatas_dict,
        modality_likelihoods={"rna": "nb", "protein": "nb"},
    )
    ```
14. Code cell: inspect `adata.uns["is_multimodal"]`, `modality_names`,
    `groups_modality_lengths`, `groups_mapping`. This cell exists specifically
    to make the new `.uns` schema visible to readers.

### Section F — `setup_anndata` and model instantiation (baseline)
15. Code cell: register the AnnData with `groups_key="groups"`, `label_key="cell_types"`,
    `modality_likelihoods={"rna": "nb", "protein": "nb"}`. Watch the printed
    banner explain that **Label-based PoE** is selected (it's the only PoE
    variant compatible with N ≥ 2 groups + multimodal — §1 in `README`).
16. Code cell: instantiate the baseline model with the standard Gaussian prior:
    ```python
    model_base = spVIPES.model.spVIPES(
        adata,
        n_hidden=128,
        n_dimensions_shared=20,
        n_dimensions_private=10,
        dropout_rate=0.1,
        use_nf_prior=False,
    )
    print(model_base)
    ```
17. Code cell: build `group_indices_list` from `adata.uns["groups_obs_indices"]`
    (this is the exact API `train` and `get_latent_representation` require).
18. Code cell: train `model_base.train(group_indices_list, max_epochs=50,
    batch_size=256, early_stopping=True)`. Plot training losses from the
    history (`reconst_loss_group_0_rna`, `reconst_loss_group_0_protein`, ...,
    `kl_divergence_poe_group_*`). This is where the reader sees per-modality
    reconstruction losses appearing in the loss output — evidence that the
    multimodal path is live.

### Section G — Extract latents and downstream embedding (baseline)
19. Code cell: `latents_base = model_base.get_latent_representation(group_indices_list)`.
    Show the returned keys: `shared`, `private`, `shared_reordered`,
    `private_reordered`, **plus** the new `private_multimodal` and
    `private_multimodal_reordered` (keyed by `(group_idx, modality)`).
20. Code cell: build one combined shared embedding by stacking
    `latents_base["shared_reordered"]` across groups in the original cell
    order, and write it into `adata.obsm["X_spVIPES_shared"]`.
21. Code cell: UMAP + plot on the shared latent, coloured by
    `cell_types`, `groups`, and `tissue`. Expected: clean cell-type
    structure, groups mixed.
22. Code cell: UMAP on each group's **private** latent (loop over groups and
    plot small subplots coloured by cell type). Expected: group-specific
    variation is captured and distinct from the shared space.
23. Code cell (new-feature spotlight): UMAP on the **per-modality** private
    latent — loop over `latents_base["private_multimodal_reordered"].items()`
    and embed each `(group, modality)` array separately. Expected: protein
    private latents should recapture lineage structure even when RNA private
    latents alone are weaker, and vice versa. This is the cell that clearly
    demonstrates the "per-(group, modality) private latent" capability.

### Section H — Turn on the NF prior
24. Markdown cell: one-paragraph explanation of why: the Gaussian prior is a
    common capacity bottleneck for learned latent geometry; a normalizing-flow
    prior lets the aggregate posterior match a flexible learnable
    distribution. Cite Bishop/Papamakarios. Note that in our implementation
    the KL is computed by Monte-Carlo (§`_nf_kl`) rather than in closed form.
25. Code cell: fresh model with NSF prior on the shared latent:
    ```python
    model_nf = spVIPES.model.spVIPES(
        adata,
        n_hidden=128,
        n_dimensions_shared=20,
        n_dimensions_private=10,
        dropout_rate=0.1,
        use_nf_prior=True,
        nf_type="NSF",
        nf_transforms=3,
        nf_target="shared",
    )
    ```
26. Code cell: train with the same hyper-parameters and seed. Plot loss curves
    side-by-side with `model_base`.
27. Code cell: extract latents (`latents_nf`), UMAP, plot next to the baseline
    UMAP.

### Section I — Quantitative comparison
28. Code cell: compute three scores on the shared embedding for both models:
    - **kBET** or **graph-connectivity** (batch mixing across groups, lower is
      better for batch effect),
    - **ARI** of Leiden clusters vs. `cell_types` (biological conservation,
      higher is better),
    - **Silhouette** by `cell_types` on the shared latent.
    Use `scib-metrics` if available; otherwise `sklearn.metrics.adjusted_rand_score`
    + a hand-rolled group-mixing entropy. Present as a small DataFrame / bar
    plot labelled `Gaussian prior` vs `NSF prior`.
29. Markdown cell: interpret. Expectation: NSF prior should **not** hurt and
    often modestly improves bio-conservation while keeping groups mixed. If it
    hurts, say so — this vignette is not a marketing piece.

### Section J — Per-group loadings
30. Code cell: `loadings = model_nf.get_loadings()` → DataFrame for each
    `(group, "shared"|"private")`. Show the top-10 genes per shared dim for
    one group.
31. Code cell: **new feature reminder** — with multimodal data, loadings still
    come from the group-level decoder. Point out the limitation (there is no
    separate protein loadings accessor yet) and file it as follow-up.

### Section K — Sanity checks and footguns
32. Markdown cell: small callout block listing pitfalls encountered during
    development that the reader should know:
    - Label-based PoE is the **only** PoE path that supports N > 2 groups;
      transport-plan PoE variants raise on > 2 groups.
    - `protein_join="inner"` is the default and means proteins not present in
      all batches are dropped. Mention alternative `"outer"`.
    - `use_nf_prior=True` adds flow parameters to the single optimizer; no
      warmup changes needed.
    - `nf_target="both"` uses two independent flows (one for shared, one for
      private) — more parameters, longer training.

### Section L — Appendix: 2-group PBMC quick reproducer
33. Code cell: `adata_pbmc = scvi.data.pbmcs_10x_cite_seq(save_path="./data/")`
    and re-run the minimal `prepare_multimodal_adatas` → `setup_anndata` →
    `train` → `get_latent_representation` flow in ~8 cells. This section
    doubles as a smoke test that the vignette's code path works on a smaller
    dataset.

### Section M — What next
34. Markdown cell: pointers to follow-up work — scaling to
    `pbmc_seurat_v4_cite_seq` (24 batches), mixing `modality_likelihoods=
    {"rna": "nb", "protein": "gaussian"}` on CLR-normalized proteins,
    trying `nf_type="MAF"` and `nf_target="both"`, wiring per-modality
    loadings.

### Section N — References
35. Markdown cell: bib entries for spVIPES, totalVI (Gayoso et al.),
    Zuko / Neural Spline Flows (Durkan et al.).

---

## 4. Files touched

| Path | Change |
|---|---|
| `docs/notebooks/multimodal_nf_tutorial.ipynb` | **New** notebook implementing §3. |
| `docs/notebooks/vignette_plan.md` | **This file** — planning scratchpad; keep it in-tree until the notebook lands, then delete. |
| `docs/index.md` | Add a tutorial entry pointing to the new notebook (below the existing `Tutorial.ipynb` link). |
| `docs/api.md` | Add `spVIPES.data.prepare_multimodal_adatas` to the API autodoc list if it isn't there yet (verify first). |
| `CHANGELOG.md` | Under `[Unreleased] → Added`, note "Multimodal + NF-prior tutorial notebook". |

No code changes in `src/` are required by the vignette. If the author runs the
notebook end-to-end and finds bugs, fix them in a separate commit — don't mix
tutorial work with library fixes.

---

## 5. Validation plan before commit

1. **Run the notebook top-to-bottom** on a single GPU with `max_epochs=50`.
   Every cell must execute without error.
2. `jupyter nbconvert --to notebook --execute multimodal_nf_tutorial.ipynb`
   must succeed in < 20 min.
3. `pytest tests/test_multigroup_multimodal.py tests/test_nf_prior.py -q` must
   still pass (the notebook does not touch library code, but re-run as a
   safety net).
4. Check that the saved notebook has stripped outputs except for the plots
   we explicitly want ship (`nbstripout`-friendly).
5. Eyeball the final UMAPs: expect cell-type clusters, good cross-group
   mixing, and sensible per-modality private embeddings.
6. Link-check `docs/index.md` renders the new entry.

---

## 6. Runtime and resource budget

| Config | Epochs | Wall-clock (T4) | Wall-clock (CPU) |
|---|---|---|---|
| Full dataset (~30k cells, 3 groups, 2 modalities) | 50 | ~15 min | skip |
| Subsampled 8k/group (~24k total) | 50 | ~5 min | ~40 min |
| PBMC appendix (~15k cells, 2 groups) | 30 | ~2 min | ~15 min |

Default the notebook to the **subsampled 8k/group** configuration so it runs
on modest hardware, and leave the full-dataset command commented out for
users who want the "real" numbers.

---

## 7. Open questions (resolve before writing the notebook)

1. **Protein likelihood:** NB on raw protein counts vs. Gaussian on CLR-
   normalized? Decision: default to **NB on raw counts** (one-line change to
   switch) to avoid teaching a second preprocessing step. Call out the
   alternative in Section K.
2. **Third group choice:** stick with the `{SpleenA, LymphA, SpleenB}` layout
   above, or use all four `batch×tissue` buckets? Decision: **three** —
   simpler narrative, and exercises the `_label_based_poe` branch where some
   cell types are missing from a group (documented in the label-based PoE
   implementation).
3. **Metric library:** `scib-metrics` is an extra dependency. Decision: soft
   import; if absent, use sklearn-only fallbacks and note that the richer
   metrics are available via `pip install scib-metrics`.
4. **Notebook inclusion in tests:** do we run the notebook in CI? Decision:
   **no** (compute budget) — but add a pytest marker to run it optionally
   (`pytest -m notebook`).

---

## 8. Done criteria

- [ ] `multimodal_nf_tutorial.ipynb` executes top-to-bottom.
- [ ] All three new features (N ≥ 2 groups, multimodal PoE, NF prior) are
      each exercised by at least one code cell and explained in at least one
      markdown cell.
- [ ] Both quantitative (metric table) and qualitative (UMAPs) comparisons
      between baseline and NF prior are shown.
- [ ] `docs/index.md` links the new notebook.
- [ ] `CHANGELOG.md` updated.
- [ ] This plan file (`vignette_plan.md`) is removed in the same PR that
      lands the notebook.
