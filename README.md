<div align="center">

# spVIPESmulti

**Shared-private Variational Inference with Product of Experts and Supervision**

[![PyPI][badge-pypi]][link-pypi]
[![Documentation][badge-docs]][link-docs]

</div>

---

## About

spVIPESmulti (v1.0.0) enables robust integration of multi-group single-cell datasets through a principled shared-private latent space decomposition. The model learns both **shared** representations (biological signals common across groups) and **private** representations (group-specific variation) using a Product of Experts (PoE) framework.

An optional **disentanglement objective** (inspired by CellDISECT and Multi-ContrastiveVAE) can additionally enforce that `z_shared` encodes biology — and only biology — while `z_private` encodes group-specific variation — and only that. This objective is fully supported in both single-modal and multimodal modes. See [Disentanglement Objective](#disentanglement-objective) below.

### Integration Strategies

spVIPESmulti aligns groups via a label-supervised Product of Experts (PoE) framework. Cell-type labels guide the PoE integration and support N ≥ 2 groups:

| Method | How it's selected | Best use case |
| --- | --- | --- |
| **Label-based PoE** | `label_key` provided | High-quality cell type labels; supports N ≥ 2 groups |
| **Unsupervised PoE** | `label_key` omitted | No label annotations available; integration quality depends on data overlap |

## Installation

### Requirements

-   Python ≥ 3.10
-   scvi-tools ≥ 1.0, < 2 (built on `lightning.pytorch`)
-   PyTorch ≥ 2.0 (GPU strongly recommended)
-   zuko ≥ 1.0.0 (normalizing flows prior)

> **scvi-tools 1.x note.** The deprecated `use_gpu=True` kwarg on `model.train(...)` has been removed upstream. Pass GPU settings via `trainer_kwargs`: `model.train(accelerator="gpu", devices=1)`. Several private scvi-tools modules removed in 1.x are now vendored under `spVIPESmulti.data`.

### Quick Install

```bash
pip install spVIPESmulti
```

Development version:

```bash
pip install git+https://github.com/mdmanurung/spVIPESmulti.git@main
```

With test/dev extras:

```bash
pip install -e ".[dev,test]"
```

## Quick Start

### Data Preparation

```python
import spVIPESmulti
import scanpy as sc

# Single-modal: dict of {group_name: AnnData}
adata1 = sc.read_h5ad("dataset1.h5ad")
adata2 = sc.read_h5ad("dataset2.h5ad")

combined = spVIPESmulti.data.prepare_adatas({"control": adata1, "treatment": adata2})

# Multimodal: dict of {group_name: {modality_name: AnnData}}
combined = spVIPESmulti.data.prepare_multimodal_adatas({
    "control":   {"rna": rna1,   "protein": prot1},
    "treatment": {"rna": rna2,   "protein": prot2},
})
```

`prepare_adatas` and `prepare_multimodal_adatas` write integration metadata into `adata.uns` (`groups_lengths`, `groups_var_indices`, `groups_obs_indices`, `groups_mapping`, and for multimodal: `groups_modality_lengths`, `groups_modality_masks`, `modality_names`, `modality_likelihoods`).

### Basic Workflow

```python
# 1. Register the AnnData
spVIPESmulti.model.spVIPESmulti.setup_anndata(
    combined,
    groups_key="groups",
    label_key="cell_type",   # optional; enables label-supervised PoE
    batch_key="batch",       # optional; enables batch correction
)

# 2. Build and train
group_indices_list = [list(map(int, g)) for g in combined.uns["groups_obs_indices"]]
model = spVIPESmulti.model.spVIPESmulti(combined)
model.train(group_indices_list, max_epochs=200)

# 3. Extract representations
latents = model.get_latent_representation(group_indices_list, batch_size=512)
spVIPESmulti.utils.store_latents(combined, latents, group_indices_list)
# writes: combined.obsm["X_spVIPESmulti_shared"], combined.obsm["X_spVIPESmulti_private_g0"], ...
```

### Integration Strategies

<details>
<summary><b>Label-based Integration (recommended, N ≥ 2 groups)</b></summary>

```python
spVIPESmulti.model.spVIPESmulti.setup_anndata(
    combined,
    groups_key="groups",
    label_key="cell_type",
    batch_key="batch",   # optional
)
```

</details>

<details>
<summary><b>Unsupervised Integration (no labels)</b></summary>

```python
spVIPESmulti.model.spVIPESmulti.setup_anndata(
    combined,
    groups_key="groups",
    # omit label_key for unsupervised PoE
)
```

</details>

### Model Parameters

```python
model = spVIPESmulti.model.spVIPESmulti(
    combined,
    n_dimensions_shared=25,      # shared latent dimensionality
    n_dimensions_private=10,     # private latent dimensionality per group
    n_hidden=128,                # hidden layer width
    dropout_rate=0.1,
    # Normalizing flow prior (optional):
    use_nf_prior=True,
    nf_type="NSF",               # "NSF" or "MAF"
    nf_transforms=3,
    nf_target="shared",          # "shared", "private", or "both"
    # Disentanglement (optional):
    disentangle_preset="full",   # see Disentanglement section below
)
```

### Training

```python
model.train(
    group_indices_list,
    max_epochs=300,
    batch_size=512,
    early_stopping=True,
    check_val_every_n_epoch=10,
    accelerator="gpu",   # replaces the removed use_gpu=True
    devices=1,
)
```

## Disentanglement Objective

spVIPESmulti exposes an optional disentanglement objective inspired by **CellDISECT** and **Multi-ContrastiveVAE**. It is implemented as a mix of:

-   **Adversarial losses** via gradient reversal (GRL / DANN-style) — to *erase* a covariate from a latent space
-   **Supervised classification losses** — acting as variational MI lower bounds to *preserve* a covariate
-   **Prototype InfoNCE** on `z_shared` — pulls same-label cells together across groups

The five loss components and what they enforce:

| # | Component | Input | Goal | Mechanism |
|---|---|---|---|---|
| 1 | `q_group_shared` | `z_shared` | erase group identity | adversarial CE via GRL |
| 2 | `q_label_shared` | `z_shared` | preserve cell-type info | supervised CE (MI lower bound) |
| 3 | `q_group_private` | `z_private` | preserve group identity | supervised CE |
| 4 | `q_label_private` | `z_private` | erase cell-type info | adversarial CE via GRL |
| 5 | contrastive | `z_shared` | pull same-label cells together across groups | prototype InfoNCE (EMA prototypes) |

Together they enforce: **`z_shared` ↔ biology only; `z_private` ↔ group only**.

### Presets

Select a preset via `disentangle_preset=` on the model constructor. Individual weights can always override a preset — `None` means "use the preset's value"; any numeric value (including `0.0`) overrides.

| Preset | `group_shared` | `label_shared` | `group_private` | `label_private` | `contrastive` | Description |
|---|---|---|---|---|---|---|
| `"off"` **(default)** | 0 | 0 | 0 | 0 | 0 | No disentanglement; fully backward-compatible |
| `"full"` | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | All five components active at sensible defaults |
| `"shared_only"` | 1.0 | 1.0 | 0 | 0 | 0.5 | Only `z_shared` decoupling losses |
| `"private_only"` | 0 | 0 | 1.0 | 1.0 | 0 | Only `z_private` decoupling losses |
| `"adversarial_only"` | 1.0 | 0 | 0 | 1.0 | 0 | Only the GRL (gradient reversal) components |
| `"supervised_only"` | 0 | 1.0 | 1.0 | 0 | 0.5 | Only non-GRL (supervised) components |
| `"no_contrastive"` | 1.0 | 1.0 | 1.0 | 1.0 | 0 | `"full"` without the InfoNCE term |

```python
# No disentanglement (default):
model = spVIPESmulti.model.spVIPESmulti(combined)

# Full disentanglement:
model = spVIPESmulti.model.spVIPESmulti(combined, disentangle_preset="full")

# Preset with per-component override (e.g. ablation study):
model = spVIPESmulti.model.spVIPESmulti(combined, disentangle_preset="full", contrastive_weight=0.0)

# Fine-grained manual control:
model = spVIPESmulti.model.spVIPESmulti(
    combined,
    disentangle_group_shared_weight=1.0,
    disentangle_label_shared_weight=1.0,
    disentangle_group_private_weight=0.5,
    disentangle_label_private_weight=0.5,
    contrastive_weight=0.2,
    contrastive_temperature=0.1,
)
```

### Constraints

-   **Labels required for label-using components.** Components 2 (`label_shared`), 4 (`label_private`), and 5 (contrastive) require `label_key` in `setup_anndata`. Components 1 and 3 (the `group_*` classifiers) work without labels — group identity is always known.
-   **Multimodal fully supported.** Components 1, 2, and 5 act on the post-PoE shared latent (modality-agnostic). Components 3 and 4 loop over each modality's private latent, summing per-modality CE terms.

See [`docs/notebooks/disentangle_ablation.ipynb`](docs/notebooks/disentangle_ablation.ipynb) for a per-component ablation walkthrough, and `scripts/validate_disentanglement_multimodal.py` for a systematic multimodal preset benchmark.

## Multimodal Integration

`prepare_multimodal_adatas` accepts `{group: {modality: AnnData}}` and builds a single combined AnnData. The model then learns per-(group, modality) encoders/decoders with a two-level PoE: intra-group across modalities, then inter-group across groups.

```python
combined = spVIPESmulti.data.prepare_multimodal_adatas({
    "control":   {"rna": rna1,   "protein": prot1},
    "treatment": {"rna": rna2,   "protein": prot2},
})

spVIPESmulti.model.spVIPESmulti.setup_anndata(
    combined,
    groups_key="groups",
    label_key="cell_type",
    modality_likelihoods={"rna": "nb", "protein": "nb"},
)

model = spVIPESmulti.model.spVIPESmulti(
    combined,
    # Re-balance per-modality reconstruction loss (~1000 HVGs vs. ~110 proteins):
    modality_loss_weights={"rna": 1.0, "protein": 5.0},
    # Symmetric-KL alignment between group PoE posteriors (complements disentanglement):
    use_jeffreys_integ=True,
    jeffreys_integ_weight=0.5,
    # Disentanglement works in multimodal mode:
    disentangle_preset="full",
)
```

Inspect which (group, modality) pairs hold real data:

```python
mask = combined.uns["groups_modality_masks"]  # {group_idx: {modality: bool}}
```

See [`docs/notebooks/multimodal_nf_tutorial.ipynb`](docs/notebooks/multimodal_nf_tutorial.ipynb) for an end-to-end CITE-seq example.

## Normalizing Flow Prior

Replace the standard Gaussian prior with a learned normalizing flow over `z_shared`, `z_private`, or both:

```python
model = spVIPESmulti.model.spVIPESmulti(
    combined,
    use_nf_prior=True,
    nf_type="NSF",       # Neural Spline Flow (default) or "MAF"
    nf_transforms=3,     # number of coupling transforms
    nf_target="shared",  # "shared", "private", or "both"
)
```

See [`docs/notebooks/cinemaot_nf_vignette.ipynb`](docs/notebooks/cinemaot_nf_vignette.ipynb) for a comparison of Gaussian vs. NSF prior vs. disentanglement.

## Post-training Utilities

The `spVIPESmulti.utils` and `spVIPESmulti.pl` modules provide ready-to-use helpers that
eliminate the boilerplate repeated in every analysis notebook.

### Storing latent representations

After calling `model.get_latent_representation(...)`, use `store_latents` to
stitch per-group arrays back into `adata.obsm` in original cell order:

```python
latents = model.get_latent_representation(group_indices_list, batch_size=512)
spVIPESmulti.utils.store_latents(adata, latents, group_indices_list)
# writes: adata.obsm["X_spVIPESmulti_shared"], adata.obsm["X_spVIPESmulti_private_g0"], ...
```

### UMAP embeddings

```python
# Shared latent UMAP (all groups integrated):
spVIPESmulti.utils.compute_shared_umap(adata)
spVIPESmulti.pl.umap_shared(adata, color=["cell_type", "groups"])

# Per-group private latent UMAPs:
adatas = {"control": adata_g0, "treatment": adata_g1}
spVIPESmulti.utils.compute_private_umaps(adatas)
fig = spVIPESmulti.pl.umap_private(adatas, color="cell_type")
```

### Gene loadings

Rank genes by loading magnitude per latent dimension and visualise them:

```python
# Top genes per shared latent dimension:
top = spVIPESmulti.utils.get_top_genes(model=model, n_top=10)
print(top[["dim", "pos_genes"]].to_string(index=False))

# Heatmap of top-5 genes per dimension (requires seaborn):
ax = spVIPESmulti.pl.heatmap_loadings(model=model, n_top=5)

# Scanpy dotplot of selected dimensions:
spVIPESmulti.pl.loadings_dotplot(adata, dims=[0, 2, 4], groupby="cell_type", model=model)
```

### Per-factor coloring and violin plots

```python
# Copy a single dimension into adata.obs for use as a color key:
spVIPESmulti.utils.score_cells_on_factor(adata_g0, dim_idx=2, obsm_key="X_spVIPESmulti_private_g0")

# Or copy all dimensions at once (optionally capped):
spVIPESmulti.utils.add_latent_dims_to_obs(adata_g0, "X_spVIPESmulti_private_g0", max_dims=5)

# Violin plot of a specific latent factor:
spVIPESmulti.pl.factor_violin(adata_g0, dim_idx=1, groupby="cell_type",
                          obsm_key="X_spVIPESmulti_private_g0")
```

### Training diagnostics

```python
fig = spVIPESmulti.pl.training_curves(model)
fig.savefig("training.pdf")
```

| Function | Module | Description |
|---|---|---|
| `store_latents` | `spVIPESmulti.utils` | Stitch per-group latents into `adata.obsm` in original cell order |
| `add_latent_dims_to_obs` | `spVIPESmulti.utils` | Copy latent dims into `adata.obs` for use as scanpy `color=` keys |
| `compute_shared_umap` | `spVIPESmulti.utils` | Run neighbours + UMAP on the shared latent |
| `compute_private_umaps` | `spVIPESmulti.utils` | Run neighbours + UMAP on each group's private latent |
| `get_top_genes` | `spVIPESmulti.utils` | Rank genes by loading magnitude per latent dimension |
| `score_cells_on_factor` | `spVIPESmulti.utils` | Write one latent dimension into `adata.obs` |
| `heatmap_loadings` | `spVIPESmulti.pl` | Seaborn heatmap of top-N gene loadings per dimension |
| `umap_shared` | `spVIPESmulti.pl` | Plot the shared latent UMAP (wraps `sc.pl.embedding`) |
| `umap_private` | `spVIPESmulti.pl` | Grid of per-group private UMAP panels |
| `factor_violin` | `spVIPESmulti.pl` | Violin plot of a single latent factor by cell metadata |
| `training_curves` | `spVIPESmulti.pl` | Multi-panel plot of training history |
| `loadings_dotplot` | `spVIPESmulti.pl` | Scanpy dotplot of top genes for selected latent dimensions |

## Documentation & Tutorials

-   [Basic Tutorial](docs/notebooks/Tutorial.ipynb) — Complete walkthrough of spVIPESmulti functionality
-   [Disentanglement ablation](docs/notebooks/disentangle_ablation.ipynb) — Per-component ablation of the disentanglement objective
-   [PBMC CITE-seq vaccination](docs/notebooks/pbmc_citeseq_tutorial.ipynb) — Three time-point integration + multimodal appendix
-   [CINEMA-OT + NF prior](docs/notebooks/cinemaot_nf_vignette.ipynb) — Gaussian vs. NSF prior vs. disentanglement
-   [Plasmodium liver-stage](docs/notebooks/biolord_comparison_plasmodium_tutorial.ipynb) — Comparison with biolord
-   [Multimodal + NF prior](docs/notebooks/multimodal_nf_tutorial.ipynb) — RNA + protein integration with `prepare_multimodal_adatas`
-   [API Documentation][link-api] — Comprehensive API reference

## Support

-   [Issue Tracker][issue-tracker] — Report bugs and request features

## Citation

If you use spVIPESmulti in your research, please cite:

```bibtex
@article{spVIPESmulti2023,
  title={Integrative learning of disentangled representations},
  author={C. Novella-Rausell, D.J.M Peters and A. Mahfouz},
  journal={bioRxiv},
  year={2023},
  doi={10.1101/2023.11.07.565957},
  url={https://www.biorxiv.org/content/10.1101/2023.11.07.565957v1}
}
```

**Paper**: [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.11.07.565957v1)

---

<!-- Badge references -->

[badge-tests]: https://img.shields.io/github/actions/workflow/status/mdmanurung/spVIPESmulti/test.yaml?branch=main
[badge-python]: https://img.shields.io/pypi/pyversions/spVIPESmulti
[badge-pypi]: https://img.shields.io/pypi/v/spVIPESmulti
[badge-docs]: https://readthedocs.org/projects/spvipesmulti/badge/?version=latest
[link-tests]: https://github.com/mdmanurung/spVIPESmulti/actions/workflows/test.yml
[link-python]: https://pypi.org/project/spVIPESmulti
[link-pypi]: https://pypi.org/project/spVIPESmulti
[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/mdmanurung/spVIPESmulti/issues
[changelog]: https://spVIPESmulti.readthedocs.io/latest/changelog.html
[link-docs]: https://spvipesmulti.readthedocs.io/en/latest/
[link-api]: https://spvipesmulti.readthedocs.io/en/latest/api.html
