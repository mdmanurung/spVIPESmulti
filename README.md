<div align="center">

# spVIPES

**Shared-private Variational Inference with Product of Experts and Supervision**

[![PyPI][badge-pypi]][link-pypi]
[![Documentation][badge-docs]][link-docs]

</div>

---

## About

spVIPES enables robust integration of multi-group single-cell datasets through a principled shared-private latent space decomposition. The method leverages a Product of Experts (PoE) framework to learn both shared biological signals common across datasets and private representations capturing group-specific variations.

An optional **disentanglement objective** (inspired by CellDISECT and Multi-ContrastiveVAE) can additionally enforce that the shared latent encodes biology — and only biology — while the private latent encodes group-specific variation — and only that. See [Disentanglement Objective](#disentanglement-objective) below.

### Integration Strategies

spVIPES provides three complementary approaches for dataset alignment:

| Method                   | Description                                               | Best Use Case                                       |
| ------------------------ | --------------------------------------------------------- | --------------------------------------------------- |
| **Label-based PoE**      | Uses cell type annotations for direct supervision         | High-quality cell type labels available             |
| **OT Paired PoE**        | Direct cell-to-cell correspondences via optimal transport | Known cellular correspondences (e.g., time series)  |
| **OT Cluster-based PoE** | Automated cluster matching with transport plans           | Similar cell populations, no direct correspondences |

> **Note:** The method automatically selects the most appropriate strategy based on available annotations and transport information.

## Installation

### Requirements

-   Python ≥ 3.10
-   scvi-tools ≥ 1.0 (built on `lightning.pytorch`)
-   PyTorch (GPU support strongly recommended)

> **scvi-tools 1.x note.** The deprecated `use_gpu=True` kwarg on `model.train(...)` has been removed; pass `accelerator="gpu", devices=1` (or `"auto"`) inside `**trainer_kwargs` instead. Several private scvi-tools modules removed in 1.x are now vendored under `spVIPES.data`.

### Quick Install

Install the latest stable release from PyPI:

```bash
pip install spVIPES
```

For the development version:

```bash
pip install git+https://github.com/nrclaudio/spVIPES.git@main
```

### GPU Setup (Recommended)

For optimal performance, ensure CUDA-compatible PyTorch is installed:

```bash
# Check GPU availability
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 11.3)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

> See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for version-specific instructions.

## Quick Start

### Basic Workflow

```python
import spVIPES
import scanpy as sc

# Load your multi-group dataset
adata = sc.read_h5ad("data.h5ad")

# Configure integration strategy
spVIPES.model.spVIPES.setup_anndata(
    adata,
    groups_key="dataset",
    label_key="cell_type",  # Optional: for supervised integration
)

# Initialize and train model
model = spVIPES.model.spVIPES(adata)
model.train(max_epochs=200)

# Extract integrated representations
latent = model.get_latent_representation()
adata.obsm["X_spVIPES"] = latent
```

### Integration Strategies

<details>
<summary><b>📋 Label-based Integration</b></summary>

Use when high-quality cell type annotations are available:

```python
spVIPES.model.spVIPES.setup_anndata(
    adata,
    groups_key="dataset",
    label_key="cell_type",
    batch_key="batch",  # Optional batch correction
)
```

</details>

<details>
<summary><b>🔄 Optimal Transport: Paired Cells</b></summary>

For datasets with known cell-to-cell correspondences:

```python
# Assumes transport plan stored in adata.uns["transport_plan"]
spVIPES.model.spVIPES.setup_anndata(
    adata,
    groups_key="dataset",
    transport_plan_key="transport_plan",
    match_clusters=False,
)
```

</details>

<details>
<summary><b>🧩 Optimal Transport: Cluster Matching</b></summary>

For automatic cluster-based alignment:

```python
spVIPES.model.spVIPES.setup_anndata(
    adata,
    groups_key="dataset",
    transport_plan_key="transport_plan",
    match_clusters=True,  # Enables automated cluster matching
)
```

</details>

### Advanced Configuration

```python
# Custom model parameters
model = spVIPES.model.spVIPES(
    adata,
    n_dimensions_shared=25,  # Shared latent dimensions
    n_dimensions_private=10,  # Private latent dimensions
    n_hidden=128,  # Hidden layer size
    dropout_rate=0.1,  # Regularization
)

# Training with custom settings
model.train(
    max_epochs=300,
    batch_size=512,
    early_stopping=True,
    check_val_every_n_epoch=10,
    accelerator="gpu",  # scvi-tools 1.x: replaces the removed use_gpu=True
    devices=1,
)
```

## Disentanglement Objective

spVIPES exposes an optional disentanglement objective inspired by **CellDISECT**'s cross-covariate decoupling MLPs and **Multi-ContrastiveVAE**. It is *not* the original Mutual Information Gap metric (Chen et al. 2018) — what we implement is a mix of:

-   **adversarial domain-invariance losses** via gradient reversal (DANN-style)
-   **supervised classification losses** acting as variational MI lower bounds
-   an optional **prototype InfoNCE** on the shared latent

These four classifiers + one contrastive term together push:

```
z_shared  : encode cell-type label,  not group identity
z_private : encode group identity,   not cell-type label
```

### Quick start with a preset

```python
# Default (no disentanglement, fully backward-compatible):
model = spVIPES.model.spVIPES(adata)

# Full disentanglement objective:
model = spVIPES.model.spVIPES(adata, disentangle_preset="full")

# Preset + per-component override (ablation):
model = spVIPES.model.spVIPES(adata, disentangle_preset="full", contrastive_weight=0.0)
```

Available presets: `"off"` (default), `"full"`, `"shared_only"`, `"private_only"`, `"adversarial_only"`, `"supervised_only"`, `"no_contrastive"`.

### Constraints

-   **Labels required for label-using components.** `disentangle_label_shared_weight`, `disentangle_label_private_weight`, and `contrastive_weight > 0` require `label_key` to be registered with `setup_anndata`. The two group classifiers (`q_group_shared`, `q_group_private`) work without labels — group identity is always known.
-   **Multimodal supported.** As of the multimodal-disentanglement work (P8), the objective applies in multimodal mode too: components 1, 2, 5 act on the post-PoE shared latent and components 3, 4 loop over each modality's private latent. The earlier construction-time `ValueError` for multimodal + disentangle has been removed. See `scripts/validate_disentanglement_multimodal.py` for a systematic preset benchmark.

See [`docs/notebooks/disentangle_ablation.ipynb`](docs/notebooks/disentangle_ablation.ipynb) for a systematic per-component ablation walkthrough.

## Multimodal Integration

`spVIPES.data.prepare_multimodal_adatas` builds a single AnnData from a nested dict of `{group: {modality: AnnData}}`, then `spVIPES.model.spVIPES` learns per-(group, modality) encoders/decoders with a two-level Product of Experts (intra-group across modalities, inter-group across groups). Three multimodal-specific kwargs let you tune training:

```python
# Inspect which (group, modality) pairs hold real data
mask = adata.uns["groups_modality_masks"]  # {group_idx: {modality: bool}}

# Re-balance per-modality reconstruction (e.g. ~1000 HVGs vs. 110 proteins)
model = spVIPES.model.spVIPES(
    adata,
    modality_loss_weights={"rna": 1.0, "protein": 5.0},
)

# Symmetric-KL alignment between group PoE posteriors (unsupervised, complements disentanglement)
model = spVIPES.model.spVIPES(
    adata,
    use_jeffreys_integ=True,
    jeffreys_integ_weight=0.5,
    disentangle_preset="full",  # disentanglement now works in multimodal mode
)
```

See [`docs/notebooks/multimodal_nf_tutorial.ipynb`](docs/notebooks/multimodal_nf_tutorial.ipynb) for an end-to-end CITE-seq example covering `prepare_multimodal_adatas`, `modality_loss_weights`, `use_jeffreys_integ`, and multimodal disentanglement.

## Documentation & Tutorials

📚 **Getting Started**

-   [Basic Tutorial](docs/notebooks/Tutorial.ipynb) — Complete walkthrough of spVIPES functionality
-   [Disentanglement ablation](docs/notebooks/disentangle_ablation.ipynb) — Per-component ablation of the disentanglement objective on the DIALOGUE dataset
-   [DIALOGUE multi-group](docs/notebooks/dialogue_multigroup_vignette.ipynb) — N ≥ 2 group integration
-   [Kidney IRI time-course](docs/notebooks/iri_days_vignette.ipynb) — Three-day post-injury comparison
-   [PBMC CITE-seq vaccination](docs/notebooks/pbmc_citeseq_tutorial.ipynb) — Three time-point integration
-   [CINEMA-OT + NF prior](docs/notebooks/cinemaot_nf_vignette.ipynb) — Gaussian vs NSF prior vs disentanglement
-   [Plasmodium liver-stage](docs/notebooks/biolord_comparison_plasmodium_tutorial.ipynb) — Comparison with biolord
-   [Multimodal + NF prior](docs/notebooks/multimodal_nf_tutorial.ipynb) — RNA + protein integration
-   [API Documentation][link-api] — Comprehensive API reference

## Support & Community

💬 **Get Help**

-   [Issue Tracker][issue-tracker] — Report bugs and request features

## Citation

If you use spVIPES in your research, please cite:

```bibtex
@article{spVIPES2023,
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

[badge-tests]: https://img.shields.io/github/actions/workflow/status/nrclaudio/spVIPES/test.yaml?branch=main
[badge-python]: https://img.shields.io/pypi/pyversions/spVIPES
[badge-pypi]: https://img.shields.io/pypi/v/spVIPES
[badge-docs]: https://readthedocs.org/projects/spvipes/badge/?version=latest
[link-tests]: https://github.com/nrclaudio/spVIPES/actions/workflows/test.yml
[link-python]: https://pypi.org/project/spVIPES
[link-pypi]: https://pypi.org/project/spVIPES
[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/nrclaudio/spVIPES/issues
[changelog]: https://spVIPES.readthedocs.io/latest/changelog.html
[link-docs]: https://spvipes.readthedocs.io/en/latest/
[link-api]: https://spvipes.readthedocs.io/en/latest/api.html
