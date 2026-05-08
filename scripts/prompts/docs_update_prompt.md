# One-off Prompt: API Reference, README, and Vignette Update for spVIPESmulti

## Context

You are working on **spVIPESmulti v1.0.0** — a scvi-tools extension for multi-group
single-cell integration via a shared-private VAE with Product of Experts (PoE).
The package lives at `src/spVIPESmulti/` and uses a Sphinx + MyST-NB doc stack.

### What already exists

| Artifact | Location | Status |
|---|---|---|
| API reference (partial) | `docs/api.md` | Has model, data-prep, disentanglement, NF prior, module, utils — **missing plotting module** |
| README | `README.md` | Has install, quickstart, hyperparameter table, tutorial links — **needs version bump, badge fixes, cleaner minimal example** |
| Vignettes (notebooks) | `docs/notebooks/` | 12 notebooks — **some have stale version strings, one has a broken `pertpy` import** |
| Sphinx config | `docs/conf.py` | Configured with `autodoc_mock_imports`, `autosummary_generate=True` |

### Public API surface

```
spVIPESmulti.model.spVIPESmulti          # main model class
  .setup_anndata(adata, groups_key, label_key, sample_key, batch_key, layer, modality_likelihoods)
  .train(max_epochs, batch_size, train_size, n_epochs_kl_warmup, **trainer_kwargs)
  .embed(group_indices_list, batch_size, prefix, overwrite, normalized, give_mean, mc_samples)
  .get_latent_representation(group_indices_list, ...)
  .get_shared_posterior(batch_size)
  .get_aggregated_posterior(sample_subset)
  .differential_abundance(group_a, group_b)
  .evaluate(label_key, z_shared_key, include_private)
  .get_loadings()
  .get_enrichment_scores(network, methods, obsm_key, uns_key)
  .summarize_enrichment(scores_df, groupby)
  .interpretation_report(scores_df, groupby, label_key)

spVIPESmulti.data.prepare_adatas(adatas, layers)
spVIPESmulti.data.prepare_multimodal_adatas(adatas, modality_likelihoods, layers)

spVIPESmulti.utils.store_latents(adata, latents, group_indices_list, obsm_prefix)
spVIPESmulti.utils.add_latent_dims_to_obs(adata, obsm_key, prefix, max_dims)
spVIPESmulti.utils.compute_shared_umap(adata, obsm_key, n_neighbors, min_dist, umap_key)
spVIPESmulti.utils.compute_private_umaps(adatas_per_group, obsm_key, n_neighbors, min_dist, umap_key)
spVIPESmulti.utils.get_top_genes(loadings_df, model, group_idx, latent_type, n_top, signed)
spVIPESmulti.utils.score_cells_on_factor(adata, obsm_key, dim, groupby)

spVIPESmulti.pl.heatmap_loadings(model, group_idx, latent_type, n_top, figsize)
spVIPESmulti.pl.umap_shared(adata, color, obsm_key, **kwargs)
spVIPESmulti.pl.umap_private(adatas_per_group, color, obsm_key, **kwargs)
spVIPESmulti.pl.factor_violin(adata, dim, groupby, obsm_key, **kwargs)
spVIPESmulti.pl.training_curves(model, figsize)
spVIPESmulti.pl.loadings_dotplot(model, group_idx, n_top, **kwargs)
spVIPESmulti.pl.enrichment_heatmap(scores_df, groupby, **kwargs)
spVIPESmulti.pl.interpretation_dashboard(model, scores_df, groupby, label_key)

spVIPESmulti.module.spVIPESmultimodule   # internal PyTorch module
spVIPESmulti.nn.Encoder
spVIPESmulti.nn.LinearDecoderSPVIPE
spVIPESmulti.dataloaders.ConcatDataLoader
```

### Key architectural facts

- `prepare_adatas` outer-joins groups, prefixes var names `"{group}_GENE"`, writes metadata to `.uns`
- `prepare_multimodal_adatas` handles `{group: {modality: AnnData}}` input; sets `is_multimodal=True`
- `setup_anndata` selects PoE strategy: label-based (when `label_key` given) or unsupervised
- `train()` accepts `accelerator="gpu", devices=1` in `**trainer_kwargs` (replaces removed `use_gpu=True`)
- Disentanglement: 5-weight system via `disentangle_preset` + individual overrides
- NF prior: Zuko-based, global (one flow for all groups), `nf_type` ∈ `{"NSF","MAF"}`
- `group_loss_weights`: normalized internally; `None` = equal weights

---

## Task 1 — Complete `docs/api.md`

**File:** `docs/api.md`

The file already covers: Model constructor, `setup_anndata`, `evaluate`, `train`, `embed`,
`get_latent_representation`, `get_shared_posterior`, `get_aggregated_posterior`,
`differential_abundance`, `get_loadings`, `get_enrichment_scores`, `summarize_enrichment`,
`interpretation_report`, `prepare_adatas`, `prepare_multimodal_adatas`, disentanglement presets,
NF prior, `spVIPESmultimodule`, `Encoder`, `LinearDecoderSPVIPE`, and all `utils.*` functions.

### What is missing — add these sections

#### 1. Plotting (`spVIPESmulti.pl`)

Add a new `## Plotting` section after the Utilities section. For each function, follow the
existing parameter-table + code-snippet format used in the rest of the file.

```python
# heatmap_loadings
spVIPESmulti.pl.heatmap_loadings(model, group_idx=0, latent_type="shared", n_top=10, figsize=None)
# Returns matplotlib Figure. Plots gene loadings as a heatmap.

# umap_shared
spVIPESmulti.pl.umap_shared(adata, color="cell_type", obsm_key="X_spVIPESmulti_shared", **sc_kwargs)
# Thin wrapper around sc.pl.embedding that defaults to the shared UMAP key.

# umap_private
spVIPESmulti.pl.umap_private(adatas_per_group, color="cell_type", obsm_key="X_spVIPESmulti_private", **sc_kwargs)
# Plots one UMAP panel per group side-by-side.

# factor_violin
spVIPESmulti.pl.factor_violin(adata, dim=0, groupby="cell_type", obsm_key="X_spVIPESmulti_shared", **sc_kwargs)
# Violin plot of a single latent dimension split by groupby.

# training_curves
spVIPESmulti.pl.training_curves(model, figsize=(10,4))
# Plots train (and optionally validation) ELBO curves from model history.

# loadings_dotplot
spVIPESmulti.pl.loadings_dotplot(model, group_idx=0, n_top=5, **sc_kwargs)
# Scanpy-style dotplot of top genes per latent factor.

# enrichment_heatmap
spVIPESmulti.pl.enrichment_heatmap(scores_df, groupby="groups", **sns_kwargs)
# Seaborn clustermap of per-group mean enrichment scores.

# interpretation_dashboard
spVIPESmulti.pl.interpretation_dashboard(model, scores_df, groupby="groups", label_key="cell_type")
# Multi-panel figure: shared UMAP + loadings heatmap + enrichment heatmap.
```

#### 2. DataLoader

Add a short `## DataLoader` section covering `ConcatDataLoader`:

```python
from spVIPESmulti.dataloaders import ConcatDataLoader
```

- Purpose: cycles shorter groups so every group produces the same number of batches per epoch
- Constructor: `ConcatDataLoader(adata_manager, indices_list, shuffle, drop_last, batch_size)`
- Key attribute: `n_batches_per_epoch` (derived from the largest group)

#### 3. `score_cells_on_factor` (currently truncated in api.md)

Ensure the `score_cells_on_factor` entry is complete with a parameter table and example.

```python
spVIPESmulti.utils.score_cells_on_factor(
    adata,
    obsm_key="X_spVIPESmulti_shared",
    dim=0,
    groupby="cell_type",
)
```

Returns a DataFrame with mean factor score per group and a score normalised to [0, 1].

#### 4. Autosummary stubs for `pl` module

Add after the `## Plotting` prose section:

```rst
.. currentmodule:: spVIPESmulti

.. autosummary::
    :toctree: generated

    pl.heatmap_loadings
    pl.umap_shared
    pl.umap_private
    pl.factor_violin
    pl.training_curves
    pl.loadings_dotplot
    pl.enrichment_heatmap
    pl.interpretation_dashboard
```

---

## Task 2 — Update `README.md`

### 2a. Version badge and header

- Update the version badge placeholder to reflect `v1.0.0`
- Ensure the GitHub Actions badge URL pattern is:
  `https://img.shields.io/github/actions/workflow/status/mdmanurung/spVIPESmulti/test.yaml?branch=main`
- Add a **PyPI badge** if not present:
  `[![PyPI](https://img.shields.io/pypi/v/spVIPESmulti)](https://pypi.org/project/spVIPESmulti)`

### 2b. Minimal quickstart (replace or augment existing)

The README's current quickstart is multi-step. Add a **"5-line quickstart"** block
immediately after the installation section that demonstrates the absolute minimum:

```python
import spVIPESmulti

adata = spVIPESmulti.data.prepare_adatas({"ctrl": adata_ctrl, "treat": adata_treat})
spVIPESmulti.model.spVIPESmulti.setup_anndata(adata, groups_key="groups", label_key="cell_type")
model = spVIPESmulti.model.spVIPESmulti(adata)
model.train(max_epochs=200)
model.embed()  # writes X_spvm_shared and X_spvm_private_* into adata.obsm
```

### 2c. GPU note

After the quickstart, add a callout box (using a markdown blockquote):

> **GPU acceleration:** Pass `accelerator="gpu", devices=1` (or `"auto"`) to
> `model.train()` via `**trainer_kwargs`. The deprecated `use_gpu=True` is removed.

### 2d. Multimodal quickstart stub

Below the single-modality quickstart, add a collapsed section (GitHub-flavoured markdown
`<details>`) for multimodal usage:

```python
# Multimodal (RNA + protein)
mdata = spVIPESmulti.data.prepare_multimodal_adatas(
    {"spleen": {"rna": rna_sp, "protein": prot_sp},
     "lymph":  {"rna": rna_ln, "protein": prot_ln}},
    modality_likelihoods={"rna": "nb", "protein": "nb"},
)
spVIPESmulti.model.spVIPESmulti.setup_anndata(
    mdata, groups_key="groups", label_key="cell_types",
    modality_likelihoods={"rna": "nb", "protein": "nb"},
)
model = spVIPESmulti.model.spVIPESmulti(mdata, use_nf_prior=True)
model.train(max_epochs=100, batch_size=512)
```

### 2e. Disentanglement section

Add a **"Disentanglement"** subsection (after the quickstart, before the hyperparameter table):

- One-sentence description: enables adversarial + supervised auxiliary losses to
  push group signal out of `z_shared` and biological signal out of `z_private`.
- Show preset usage: `disentangle_preset="full"`, and single override example.
- Reference `docs/api.md#disentanglement-presets` for the full preset table.

### 2f. Citation block

Ensure the BibTeX entry is present and the DOI link is live:
`https://doi.org/10.1101/2023.11.07.565957`

---

## Task 3 — Vignette hygiene

### 3a. Version string audit

In every notebook under `docs/notebooks/`, find cells that print version strings.
Any cell that prints `spVIPESmulti  : 0.3.0` (or similar pre-1.0 versions) is stale.
Update the **Markdown introduction cell** of each stale notebook to include:

```
> **Version note:** This vignette was last validated with spVIPESmulti 1.0.0,
> scvi-tools 1.4.2, Python 3.10+.
```

Do **not** re-execute cells — only update Markdown metadata cells.

Notebooks to check:
- `multimodal_nf_tutorial.ipynb` — shows `0.3.0` → add version note
- `cinemaot_nf_vignette.ipynb` — has broken `pertpy` import (jax/numpyro compat) → add warning cell
- All others — verify introduction Markdown is present and says v1.0.0

### 3b. cinemaot_nf_vignette.ipynb — broken import workaround

Insert a new Markdown cell **before** the first code cell with the broken `pertpy` import:

```markdown
> **Dependency note:** This vignette uses `pertpy` which requires `numpyro` and `jax`.
> If you see `ImportError: cannot import name 'xla_pmap_p' from 'jax.extend.core.primitives'`,
> install a compatible combination:
> ```bash
> pip install "jax[cuda12]>=0.4.30" "numpyro>=0.15" pertpy
> ```
> The spVIPESmulti sections of this notebook do **not** depend on pertpy and can be
> run independently by skipping the CINEMA-OT cells.
```

### 3c. Tutorial.ipynb — structural check

The main `Tutorial.ipynb` should have these top-level sections in order:
1. Environment (version print)
2. Data loading and preparation (`prepare_adatas`)
3. Model setup (`setup_anndata`)
4. Training (`model.train`)
5. Embedding (`model.embed` or `get_latent_representation`)
6. Visualization (shared UMAP, private UMAPs)
7. Gene loadings (`get_loadings`, `get_top_genes`)
8. Evaluation (`model.evaluate`)

If any section is missing or out of order, add a stub Markdown cell with the section
heading and a one-line description of what code belongs there.

### 3d. `docs/index.md` — vignette TOC

Verify `docs/index.md` references all 12 notebooks listed in `README.md`.
Missing entries to add if not present:
- `notebooks/malaria_bcells_recommended_time`
- `notebooks/malaria_bcells_nodisentangle`
- `notebooks/malaria_bcells_hparam_explore`
- `notebooks/malaria_bcells_gallery`

---

## Constraints and quality gates

1. **Do not break the existing api.md structure.** Append new sections; do not reorder existing ones.
2. **Use the established table + code-snippet format** from the existing `docs/api.md`.
3. **Do not re-run notebooks** — only edit Markdown cells and metadata.
4. **Keep the README under 700 lines.** The existing content is ~660 lines; new additions
   should be concise.
5. **All code snippets must be valid Python** for the current API (v1.0.0).
6. **RST autosummary directives** must use the import paths exactly as listed in the
   "Public API surface" section above.
7. **Do not remove the `sphinx_copybutton` or `myst_nb` Sphinx extensions** from `docs/conf.py`.

---

## Deliverables checklist

- [ ] `docs/api.md` — added Plotting section (8 functions), DataLoader section, completed `score_cells_on_factor`
- [ ] `README.md` — 5-line quickstart, GPU note, multimodal stub, disentanglement subsection, badge update
- [ ] `docs/notebooks/multimodal_nf_tutorial.ipynb` — version note Markdown cell added
- [ ] `docs/notebooks/cinemaot_nf_vignette.ipynb` — dependency warning Markdown cell added
- [ ] `docs/notebooks/Tutorial.ipynb` — structural sections verified / stubs added
- [ ] `docs/index.md` — all 12 vignettes listed in TOC
