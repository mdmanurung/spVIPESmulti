<div align="center">

# spVIPESmulti

**Shared-private Variational Inference with Product of Experts and Supervision**

[![PyPI][badge-pypi]][link-pypi]
[![Documentation][badge-docs]][link-docs]

</div>

______________________________________________________________________

## About

spVIPESmulti (v1.0.1) enables robust integration of multi-group single-cell datasets through a principled shared-private latent space decomposition. The model learns both **shared** representations (biological signals common across groups) and **private** representations (group-specific variation) using a Product of Experts (PoE) framework.

An optional **disentanglement objective** (inspired by CellDISECT and Multi-ContrastiveVAE) can additionally enforce that `z_shared` encodes biology — and only biology — while `z_private` encodes group-specific variation — and only that. This objective is fully supported in both single-modal and multimodal modes. See [Disentanglement Objective](#disentanglement-objective) below.

### Integration Strategies

spVIPESmulti aligns groups via a label-supervised Product of Experts (PoE) framework. Cell-type labels guide the PoE integration and support N ≥ 2 groups:

| Method | How it's selected | Best use case |
| --- | --- | --- |
| **Label-based PoE** | `label_key` provided | High-quality cell type labels; supports N ≥ 2 groups |
| **Unsupervised PoE** | `label_key` omitted | No label annotations available; integration quality depends on data overlap |

Current package capabilities include single-modal and multimodal AnnData preparation, label-supervised or unsupervised PoE integration, optional normalizing-flow priors, optional shared/private disentanglement losses, donor/batch covariate controls, sample-aware posterior aggregation, differential-abundance scoring, enrichment analysis, integration diagnostics, latent traversal, and plotting utilities.

## Installation

### Requirements

- Python ≥ 3.10
- scvi-tools ≥ 1.2.2.post2, < 2 (built on `lightning.pytorch`)
- PyTorch ≥ 2.0 (GPU strongly recommended)
- zuko ≥ 1.0.0 (normalizing flows prior)

> **scvi-tools 1.x note.** The deprecated `use_gpu=True` kwarg on `model.train(...)` has been removed upstream. Pass GPU settings via `trainer_kwargs`: `model.train(accelerator="gpu", devices=1)`. Several private scvi-tools modules removed in 1.x are now vendored under `spVIPESmulti.data`.

### Notebook and Script Environment Guard

In shared HPC environments, Python can silently mix conda packages with packages from
`~/.local/lib/python*/site-packages`. For the `torch`/`torchvision`/`scvi-tools` stack,
that can surface as import-time binary-extension errors such as
`RuntimeError: operator torchvision::nms does not exist`.

For this workspace, select the Jupyter kernel named `Python (spvm)`. It points at the
`spvm` conda interpreter and sets `PYTHONNOUSERSITE=1`.

To recreate the kernel:

```bash
conda run -n spvm python -m ipykernel install --user --name spvm --display-name "Python (spvm)"
```

For scripts, prefer:

```bash
PYTHONNOUSERSITE=1 python scripts/<script>.py
```

or:

```bash
conda run -n spvm env PYTHONNOUSERSITE=1 python scripts/<script>.py
```

The editable repo also includes two safeguards: `src/sitecustomize.py` and the
`import spVIPESmulti` guard. Both remove user-site paths before importing heavy
scientific dependencies. Set `SPVIPESMULTI_ALLOW_USER_SITE=1` only when
intentionally debugging user-site packages.

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

With enrichment extras (decoupler integration):

```bash
pip install -e ".[enrichment]"
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
import pandas as pd

# 1. Register the AnnData
spVIPESmulti.model.spVIPESmulti.setup_anndata(
    combined,
    groups_key="groups",
    label_key="cell_type",   # optional; enables label-supervised PoE
    sample_key="sample_id",  # optional; enables sample-aware posterior/DA helpers
    condition_key="stim",    # optional; registers perturbation/treatment state
    donor_key="donor",       # optional; enables donor-aware disentanglement
    batch_key="batch",       # optional; enables batch correction
)

# 2. Build and train
model = spVIPESmulti.model.spVIPESmulti(combined)
model.train(max_epochs=200)

# 3. One-call embedding (compute + store)
payload = model.embed(batch_size=512)
# writes: combined.obsm["X_spvm_shared"], combined.obsm["X_spvm_private_<group>"], ...
# payload["keys"] returns the exact keys written

# 4. Interpretation-first enrichment (includes ULM)
network = pd.DataFrame(
    {
        "source": ["TF1", "TF1", "TF1", "TF1", "TF1"],
        "target": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"],
    }
)
res = model.get_enrichment_scores(network, methods=["ora", "gsea", "ulm"])
report = model.interpretation_report(
    res["scores_df"],
    groupby="groups",
    label_key="cell_type",  # optional
)

# 5. Public evaluation API (diagnostics-first)
evaluation = model.evaluate(
    label_key="cell_type",
    z_shared_key=payload["keys"]["shared"],
    include_private=True,
)
# evaluation["metrics"] is a DataFrame with shared/private latent diagnostics
# evaluation["held_out_metrics"] includes validation ELBO / reconstruction NLL when available

# 6. Sample-aware posterior aggregation and differential abundance (optional)
posterior = model.get_aggregated_posterior(sample_subset=["S1", "S2"])  # requires sample_key in setup_anndata
da = model.differential_abundance(group_a=0, group_b=1)
# da["scores"] is a per-cell DataFrame with da_score and group labels
```

See the dedicated quick tutorial: [`docs/enrichment_quickstart.md`](docs/enrichment_quickstart.md).

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
    disentangle_warmup=True,      # warm up covariate GRL strength with KL warmup
    # Optional strict likelihood checks (default False):
    strict_likelihood_support=False,
    # Optional per-group ELBO weights (normalized internally):
    group_loss_weights=None,
)
```

`strict_likelihood_support` enables additional input validation before
likelihood `log_prob` evaluation:

- always validates finite inputs and non-negative targets for NB likelihood;
- in strict mode, also enforces integer-like counts for NB when
  `log_variational_generative=False`.

This is useful when you want early, explicit failures on mismatched training
targets instead of downstream warning-only behavior.

### Training

```python
model.train(
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

- **Adversarial losses** via gradient reversal (GRL / DANN-style) — to *erase* a covariate from a latent space
- **Supervised classification losses** — acting as variational MI lower bounds to *preserve* a covariate
- **Prototype InfoNCE** on `z_shared` — pulls same-label cells together across groups
- **Optional F3 orthogonality loss** — penalizes aligned-dimension correlation between `z_shared` and `z_private` within configured strata

The core loss components and what they enforce:

| # | Component | Input | Goal | Mechanism |
|---|---|---|---|---|
| 1 | `q_group_shared` | `z_shared` | erase group identity | adversarial CE via GRL |
| 2 | `q_label_shared` | `z_shared` | preserve cell-type info | supervised CE (MI lower bound) |
| 3 | `q_group_private` | `z_private` | preserve group identity | supervised CE |
| 4 | `q_label_private` | `z_private` | erase cell-type info | adversarial CE via GRL |
| 5 | contrastive | `z_shared` | pull same-label cells together across groups | prototype InfoNCE (EMA prototypes) |
| 6 | `q_batch_shared` | `z_shared` | erase technical batch | adversarial CE via GRL |
| 7 | `q_donor_shared` | `z_shared` | erase donor identity | adversarial CE via GRL |
| 8 | `q_donor_private` | `z_private` | preserve donor identity | supervised CE |
| 9 | `orthogonality_weight` | `z_shared`, `z_private` | reduce shared/private leakage | differentiable within-stratum correlation penalty |

Together they support the intended split: **`z_shared` captures cross-group biology; `z_private` captures group-specific or donor/private variation**.

### Presets

Select a preset via `disentangle_preset=` on the model constructor. Individual weights can always override a preset — `None` means "use the preset's value"; any numeric value (including `0.0`) overrides.

| Preset | Group/label disentanglement | Batch/donor disentanglement | Contrastive | Description |
|---|---|---|---|---|
| `"off"` **(default)** | off | off | off | No disentanglement; fully backward-compatible |
| `"full"` | all four group/label components at 1.0 | off | 0.5 | Original full preset |
| `"shared_only"` | shared components only | off | 0.5 | Only `z_shared` decoupling losses |
| `"private_only"` | private components only | off | off | Only `z_private` decoupling losses |
| `"adversarial_only"` | GRL components only | off | off | Only adversarial group/label losses |
| `"supervised_only"` | supervised components only | off | 0.5 | Only non-GRL group/label losses |
| `"no_contrastive"` | all four group/label components at 1.0 | off | off | `"full"` without InfoNCE |
| `"minimal_safe_bio"` | off | donor-private at 0.5 | off | Preserve donor/private structure without forcing shared biology |
| `"full_bio"` | all four group/label components at 1.0 | batch-shared, donor-shared, donor-private at 0.5 | 0.5 | Full biological/covariate disentanglement preset |

> **Current F4 audit note.** The `"minimal_safe_bio"` and `"full_bio"` presets are retained for reproducibility and manual experiments, but the 3-seed F4 probe audit rejected preset promotion. In the current evidence they are not very useful as recommended defaults; prefer `"off"` plus explicit per-weight overrides when testing covariate heads.

> **F3 orthogonality note.** `orthogonality_weight` is default-off in every preset. Enable it explicitly for experiments, then audit with `scripts/benchmark_f3_orthogonality.py` before treating a weight as recommended.

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
    disentangle_batch_shared_weight=0.2,
    disentangle_donor_shared_weight=0.2,
    disentangle_donor_private_weight=0.5,
    orthogonality_weight=0.05,
    contrastive_weight=0.2,
    contrastive_temperature=0.1,
)
```

### Constraints

- **Labels required for label-using components.** Components 2 (`label_shared`), 4 (`label_private`), and 5 (contrastive) require `label_key` in `setup_anndata`. Components 1 and 3 (the `group_*` classifiers) work without labels — group identity is always known.
- **Covariate keys required for covariate components.** `disentangle_batch_shared_weight` requires `batch_key`; `disentangle_donor_shared_weight` and `disentangle_donor_private_weight` require `donor_key`.
- **Multimodal fully supported.** Shared components act on the post-PoE shared latent. Private components loop over each modality's private latent, summing per-modality CE terms.

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

### HVG selection

```python
# Compute HVGs separately per group, then keep the union.
adata_hvg = spVIPESmulti.utils.highly_variable_genes_union(
    adata,
    group_key="condition",
    n_top_genes=3000,
)
```

### Storing latent representations

Use `model.embed(...)` for the shortest path (auto-infers groups from
`adata.uns["groups_obs_indices"]`):

```python
payload = model.embed(batch_size=512)
# payload["keys"]["shared"] == "X_spvm_shared"
```

If you need manual control over array post-processing, you can still call
`get_latent_representation(...)` + `store_latents(...)`:

```python
group_indices_list = [list(map(int, g)) for g in adata.uns["groups_obs_indices"]]
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

### Enrichment and interpretation

Run pathway/TF enrichment directly from the model (optional decoupler feature):

```python
network = pd.DataFrame(
    {
        "source": ["TF1", "TF1", "TF1", "TF1", "TF1"],
        "target": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"],
    }
)

res = model.get_enrichment_scores(
    network,
    methods=["ora", "gsea", "ulm"],
    obsm_key="X_spvm_enrichment",
    uns_key="spvm_enrichment",
)

summary = model.summarize_enrichment(res["scores_df"], groupby="groups")
report = model.interpretation_report(
    res["scores_df"],
    groupby="groups",
    label_key="cell_type",  # optional, enables integration metrics in report
)
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

### Model diagnostics and latent interpretation

```python
# Integration metrics from NumPy arrays or via model.evaluate(...)
metrics = model.evaluate(label_key="cell_type", z_shared_key="X_spvm_shared", include_private=True)

# Per-dimension activity/collapse diagnostics
stats = spVIPESmulti.metrics.latent_dimension_stats(payload["shared"])
fig = spVIPESmulti.pl.plot_latent_dimension_stats(stats)

# Decoder-based traversal of z_shared dimensions
traversal = model.traverse_latent(group_idx=0, n_steps=15)
top_genes = spVIPESmulti.traversal.calculate_differential_vars(traversal, top_n=20)
fig = spVIPESmulti.pl.differential_vars_heatmap(traversal)
```

| Function | Module | Description |
|---|---|---|
| `highly_variable_genes_union` | `spVIPESmulti.utils` | Compute per-group HVGs and keep their union |
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
| `get_enrichment_scores` | `spVIPESmulti.model.spVIPESmulti` | Run ORA/GSEA/ULM enrichment with optional decoupler backend |
| `summarize_enrichment` | `spVIPESmulti.model.spVIPESmulti` | Aggregate enrichment scores by any `adata.obs` grouping |
| `interpretation_report` | `spVIPESmulti.model.spVIPESmulti` | Build compact enrichment + integration summary tables |
| `evaluate` | `spVIPESmulti.model.spVIPESmulti` | Compute iLISI, cLISI, kBET acceptance rate, purity, ARI, and private-latent silhouette diagnostics |
| `integration_report` | `spVIPESmulti.metrics` | Standalone integration metric table from arrays |
| `latent_dimension_stats` | `spVIPESmulti.metrics` | Per-dimension std, mean absolute activity, KL, and collapsed-dimension flags |
| `reconstruction_error` | `spVIPESmulti.metrics` | Per-group reconstruction RMSE and Poisson NLL |
| `traverse_latent` | `spVIPESmulti.traversal` / model method | Decoder traversal for interpreting `z_shared` dimensions |
| `calculate_differential_vars` | `spVIPESmulti.traversal` | Rank genes by traversal effect per shared dimension |
| `enrichment_heatmap` | `spVIPESmulti.pl` | Plot per-cell or per-group enrichment heatmaps |
| `interpretation_dashboard` | `spVIPESmulti.pl` | Two-panel shared-embedding + enrichment dashboard |
| `plot_latent_dims_in_umap` | `spVIPESmulti.pl` | One UMAP panel per latent dimension |
| `plot_latent_dims_in_heatmap` | `spVIPESmulti.pl` | Heatmap of mean latent activity by cell type or group |
| `plot_latent_dimension_stats` | `spVIPESmulti.pl` | Barplot of per-dimension activity/collapse diagnostics |
| `show_top_differential_vars` | `spVIPESmulti.pl` | Barplot of top traversal genes for one dimension |
| `differential_vars_heatmap` | `spVIPESmulti.pl` | Traversal-effect heatmap across dimensions and genes |

## Troubleshooting & Hyperparameter Guide

This section covers common training pathologies visible in `spVIPESmulti.pl.training_curves(model)` and how to fix them.

______________________________________________________________________

### `kl_divergence_private_group_N` is orders of magnitude higher than other groups

**Symptom:** One group's private KL divergence reaches values of hundreds or thousands while the others stay in the range 1–20.

**Cause:** The `disentangle_label_private` gradient reversal layer (GRL) can drive a positive feedback loop in the private encoder's log-variance output. The GRL rewards the encoder for producing noisy `z_private` samples (high variance makes labels unclassifiable), but without an upper bound on the log-variance this snowballs — the encoder inflates variance without limit, which satisfies the GRL objective but makes the private latent useless for reconstruction. Groups with higher `group_loss_weights` (i.e. smaller groups with inverse-frequency weighting) are hit hardest because the reconstruction gradient competing against the GRL is proportionally stronger.

**Fixes (in order of impact):**

1. **Reduce `disentangle_label_private_weight`** (no code change needed). The default for `"no_contrastive"` and `"full"` presets is 1.0. Drop it to 0.1–0.3:

   ```python
   model = spVIPESmulti.model.spVIPESmulti(
       adata_spv,
       disentangle_preset="no_contrastive",
       disentangle_label_private_weight=0.1,   # override the preset default of 1.0
   )
   ```

1. **Increase KL warmup** (`n_epochs_kl_warmup`). A longer warmup (150–200 epochs) gives the KL penalty more time to regularise the private posterior before the GRL has fully pushed it toward high variance:

   ```python
   model.train(..., n_epochs_kl_warmup=150)
   ```

1. **Soften `group_loss_weights`**. Inverse-frequency weights (`1/n`) can give small groups an 8× higher effective loss scale. Square-root weights are a less aggressive alternative:

   ```python
   GROUP_LOSS_WEIGHTS = [1 / n**0.5 for n in GROUP_SIZES]
   ```

______________________________________________________________________

### LR scheduler never fires / loss plateau not detected

**Symptom:** Training curves show the learning rate is flat throughout training (with `lr_scheduler_type="plateau"`), even though reconstruction loss is still declining.

**Cause:** `ReduceLROnPlateau` requires the monitored metric to have stopped improving by more than `threshold` for `patience` consecutive checks. If the metric is still slowly declining, the scheduler never triggers.

**Fix:** Switch to a cosine schedule which decays on a fixed timeline regardless of plateau detection:

```python
model.train(
    ...,
    plan_kwargs={
        "lr": 5e-4,
        "lr_scheduler_type": "cosine",
        "lr_min": 1e-5,
    },
)
```

______________________________________________________________________

### Reconstruction loss stalls early; ELBO doesn't improve after KL warmup

**Symptom:** Reconstruction loss drops fast in the first ~50 epochs and then plateaus, while KL keeps rising after warmup ends.

**Causes and fixes:**

| Likely cause | Fix |
|---|---|
| Hidden layer too narrow | Increase `n_hidden` from 128 to 256 (matches the default in `networks.py`) |
| Too few HVGs | Use 3000–5000 genes; very few genes starve the encoder |
| `batch_size` too small | Larger batches (512–2048) reduce gradient noise, especially on GPU |
| KL weight fully active too fast | Increase `n_epochs_kl_warmup` (75→150) |

______________________________________________________________________

### Integration is poor — groups don't overlap in the shared UMAP

**Symptom:** `spVIPESmulti.pl.umap_shared(adata, color="groups")` shows clearly separated clusters per group rather than mixing.

**Causes and fixes:**

1. **Label-PoE quality**: If `label_key` labels are inconsistent across groups (different annotation granularity, or one group has many unlabelled cells), the PoE posteriors are misaligned. Use a coarser, consensus annotation across groups.

1. **`disentangle_group_shared_weight` too low**: Increase this to 1.0–2.0 to push group identity out of `z_shared`:

   ```python
   model = spVIPESmulti.model.spVIPESmulti(
       adata_spv,
       disentangle_preset="no_contrastive",
       disentangle_group_shared_weight=2.0,
   )
   ```

1. **Imbalanced groups**: Apply inverse-frequency `group_loss_weights` so the smallest group does not get overwhelmed during training:

   ```python
   GROUP_SIZES = [len(g) for g in group_indices_list]
   GROUP_LOSS_WEIGHTS = [1 / n**0.5 for n in GROUP_SIZES]   # sqrt weighting
   ```

______________________________________________________________________

### Recommended starting hyperparameters (3-group dataset, ~10k cells)

These are the settings validated on the malaria B-cell dataset (CRXV/NANP/Njunc, 9978 cells, 3 batches):

```python
# Model
model = spVIPESmulti.model.spVIPESmulti(
    adata_spv,
    n_hidden=256,
    n_dimensions_shared=10,
    n_dimensions_private=6,
    dropout_rate=0.1,
    disentangle_preset="no_contrastive",
    disentangle_label_shared_weight=2.0,
    disentangle_label_private_weight=0.1,   # keep low to avoid GRL-driven KL explosion
    use_jeffreys_integ=True,
    jeffreys_integ_weight=0.5,
    use_nf_prior=True,
    nf_type="MAF",
    nf_transforms=3,
    nf_target="shared",
    group_loss_weights=[1 / n**0.5 for n in GROUP_SIZES],   # sqrt weighting
)

# Training
model.train(
    group_indices_list,
    batch_size=1024,
    max_epochs=400,
    train_size=0.9,
    n_epochs_kl_warmup=150,
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_monitor="reconstruction_loss_validation",
    check_val_every_n_epoch=5,
    plan_kwargs={
        "lr": 5e-4,
        "lr_scheduler_type": "cosine",
        "lr_min": 1e-5,
    },
)
```

| Parameter | Recommended | Notes |
|---|---|---|
| `n_hidden` | 256 | 128 is too narrow for 3000 HVGs |
| `n_dimensions_shared` | 10–25 | Scale with dataset complexity |
| `n_dimensions_private` | 5–10 | Scale with expected group-specific variation |
| `n_epochs_kl_warmup` | 100–150 | Longer warmup stabilises private KL |
| `lr` | 5×10⁻⁴ | 10⁻³ can cause instability in early epochs |
| `lr_scheduler_type` | `"cosine"` | Plateau scheduler rarely fires when loss is still declining |
| `batch_size` | 1024 | Larger batches reduce gradient noise on GPU |
| `disentangle_label_private_weight` | 0.1 | Keep low; high values trigger GRL-driven variance explosion |
| `group_loss_weights` | `1/sqrt(n)` | Less extreme than `1/n`; balances groups without over-weighting small ones |

______________________________________________________________________

## Counterfactual Interventions

The `spVIPESmulti.interventions` API exposes a single-modal, diagnostic
encode/edit/decode workflow for trained models. The first supported biological
operator is a centroid shift in latent space; arbitrary latent replacement is
available only as a low-level diagnostic helper.

Counterfactual outputs are associative predictions from the fitted decoder, not
causal claims. Use the reported OOD flags and external benchmarks before drawing
biological conclusions.

```python
import spVIPESmulti.interventions as svi

encoded = svi.encode_cells(model, combined)
direction = (
    encoded["shared"][1].mean(axis=0)
    - encoded["shared"][0].mean(axis=0)
)

result = svi.predict_counterfactual(
    model,
    combined,
    cells=combined.uns["groups_obs_indices"][0][:100],
    group_idx=0,
    intervention="centroid_shift",
    direction=direction,
    reject_ood=True,
)

result.X                    # decoded expression rates in target group gene space
result.info["ood_flags"]    # Mahalanobis, library-ratio, and likelihood-proxy flags
result.info["var_names"]    # target decoder gene names
```

When `condition_key` was registered in `setup_anndata(...)`,
`svi.transfer_condition(...)` can compute the condition centroid direction from
the registered obs column and decode selected cells through a source or target
group decoder.

______________________________________________________________________

## Documentation & Tutorials

- [Enrichment quickstart (ORA/GSEA/ULM)](docs/enrichment_quickstart.md) — Interpretation-first workflow with reporting + plotting helpers
- [Basic Tutorial](docs/notebooks/Tutorial.ipynb) — Complete walkthrough of spVIPESmulti functionality
- [Disentanglement ablation](docs/notebooks/disentangle_ablation.ipynb) — Per-component ablation of the disentanglement objective
- [PBMC CITE-seq vaccination](docs/notebooks/pbmc_citeseq_tutorial.ipynb) — Three time-point integration + multimodal appendix
- [CINEMA-OT + NF prior](docs/notebooks/cinemaot_nf_vignette.ipynb) — Gaussian vs. NSF prior vs. disentanglement
- [Plasmodium liver-stage](docs/notebooks/biolord_comparison_plasmodium_tutorial.ipynb) — Comparison with biolord
- [Malaria B-cell recommended workflow](docs/notebooks/malaria_bcells_recommended.ipynb) — Lightweight end-to-end B-cell workflow from CSV inputs
- [Malaria B-cell ablations](docs/notebooks/malaria_bcells_nodisentangle.ipynb) and [hyperparameter exploration](docs/notebooks/malaria_bcells_hparam_explore.ipynb)
- [Kang IFN-beta workflow](docs/notebooks/kang_ifn_commit_old.ipynb) — IFN-beta benchmark notebook
- [Counterfactual interventions](docs/notebooks/counterfactual_interventions_tutorial.ipynb) — Safe centroid-shift editing with OOD diagnostics
- [Multimodal + NF prior](docs/notebooks/multimodal_nf_tutorial.ipynb) — RNA + protein integration with `prepare_multimodal_adatas`
- [API Documentation][link-api] — Comprehensive API reference

## Support

- [Issue Tracker][issue-tracker] — Report bugs and request features

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

______________________________________________________________________

<!-- Badge references -->

[badge-docs]: https://readthedocs.org/projects/spvipesmulti/badge/?version=latest
[badge-pypi]: https://img.shields.io/pypi/v/spVIPESmulti
[issue-tracker]: https://github.com/mdmanurung/spVIPESmulti/issues
[link-api]: https://spvipesmulti.readthedocs.io/en/latest/api.html
[link-docs]: https://spvipesmulti.readthedocs.io/en/latest/
[link-pypi]: https://pypi.org/project/spVIPESmulti
