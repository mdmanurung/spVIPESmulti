# Changelog

All notable changes to this project will be documented in this file.
Maintained by **Mikhael Dito Manurung** (fork of [spVIPES](https://github.com/nrclaudio/spVIPES) by Claudio Novella Rausell).

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [1.0.0] — 2026-05-01

### Changed (breaking)

-   **scvi-tools upgraded to >=1.0,<2** (was pinned to 0.20.x). Minimum Python
    is now 3.10. Several private scvi-tools modules removed in 1.x (`scvi.data._constants`,
    `scvi.data._utils.get_anndata_attribute`, `scvi.dataloaders._anntorchdataset`,
    `scvi.dataloaders._data_splitting.validate_data_split`, `scvi._types`) are now
    vendored under `spVIPESmulti.data`. `pytorch_lightning` was replaced with
    `lightning.pytorch`.
-   **`MultiGroupTrainingMixin.train(...)` no longer accepts `use_gpu`.** scvi-tools 1.x
    routes accelerator selection through `pl.Trainer`. Pass `accelerator="gpu"` (or
    `"cpu"`, `"auto"`) and `devices=...` via `**trainer_kwargs` instead.
-   **`MultiGroupDataSplitter` now stores its registered manager on
    `self.adata_manager`** (was `self.adata`). Downstream code using the splitter as a
    library should rename accordingly.

### Added

-   Basic tool, preprocessing and plotting functions
-   DIALOGUE-style multi-group vignette (`docs/notebooks/dialogue_multigroup_vignette.ipynb`)
    showcasing N ≥ 2 group integration using `clinical.status` from `pt.dt.dialogue_example()`.
-   **Multimodal tutorial extended** — `docs/notebooks/multimodal_nf_tutorial.ipynb`
    now demonstrates the multigrate-inspired multimodal kwargs introduced in
    `1749292` (`groups_modality_masks` introspection,
    `modality_loss_weights`, `use_jeffreys_integ` / `jeffreys_integ_weight`)
    and the multimodal disentanglement objective (P8). The PBMC CITE-seq
    tutorial (`pbmc_citeseq_tutorial.ipynb`) gains an "Optional: multimodal
    mode" appendix that wires up `prepare_multimodal_adatas` on the same
    data as a forward pointer to the dedicated multimodal vignette.
-   **`docs/api.md`** now documents
    `spVIPESmulti.data.prepare_adatas.prepare_multimodal_adatas` alongside the
    single-modality `prepare_adatas`.
-   **Disentanglement objective** (inspired by CellDISECT's cross-covariate
    decoupling MLPs and Multi-ContrastiveVAE). Four auxiliary classifiers + an
    optional contrastive term enforce that each latent space encodes only its
    "owned" factor:

    | Classifier | Input | Goal | Mechanism |
    |---|---|---|---|
    | `q_label_shared` | `z_shared` | preserve cell-type info | supervised CE (MI lower bound) |
    | `q_group_shared` | `z_shared` | erase group identity | adversarial CE via GRL (DANN-style) |
    | `q_group_private` | `z_private` | preserve group info | supervised CE |
    | `q_label_private` | `z_private` | erase cell-type info | adversarial CE via GRL |

    **Note on naming:** these are *not* the original Mutual Information Gap
    metric (Chen et al. 2018), which is a post-hoc disentanglement metric.
    Our losses are a mix of adversarial domain-invariance losses (gradient
    reversal) and supervised classification losses acting as variational MI
    lower bounds.

    All weights default to `0.0` (fully backward-compatible). New
    `spVIPESmultimodule` parameters: `disentangle_group_shared_weight`,
    `disentangle_label_shared_weight`, `disentangle_group_private_weight`,
    `disentangle_label_private_weight`, `contrastive_weight`,
    `contrastive_temperature`.

-   **Prototype supervised contrastive loss** on `z_shared` (optional,
    `contrastive_weight > 0`). EMA prototypes per (group × label) are
    maintained; supervised InfoNCE pulls same-label cells together across
    groups.

-   `gradient_reversal()` utility function and `_GradientReversalFunction`
    added to `spVIPESmulti.module.utils`.

-   **`disentangle_preset` kwarg** on `spVIPESmulti` model, with named
    configurations: `"off"` (default), `"full"`, `"shared_only"`,
    `"private_only"`, `"adversarial_only"`, `"supervised_only"`,
    `"no_contrastive"`. Per-component weight kwargs
    (`disentangle_*_weight`, `contrastive_weight`) now act as overrides on
    top of the preset. `None` (the default) means "use the preset's value";
    numeric values (including `0.0`) override.

-   Ablation walkthrough notebook (`docs/notebooks/disentangle_ablation.ipynb`)
    showing how to enable/disable each component via presets and per-weight
    overrides.

-   `_compute_disentangle_losses()` helper method on `spVIPESmultimodule` —
    extracts the disentanglement loss block from `loss()` for cleaner
    ablation workflow. Each component reads as one isolated
    `if self.q_X is not None` branch.

-   **Multimodal disentanglement support** (PLANS.md P8). The disentanglement
    objective now applies in multimodal mode (`is_multimodal=True`).
    Components 1 (`q_group_shared`), 2 (`q_label_shared`), and 5 (contrastive
    prototypes) operate on the post-PoE shared latent — modality-agnostic by
    construction. Components 3 (`q_group_private`) and 4 (`q_label_private`)
    loop over each modality's private latent in
    `_compute_disentangle_losses`, summing the per-modality cross-entropy
    terms with the same classifier weights (no new parameters).
    `_loss_multimodal` now invokes `_compute_disentangle_losses` before
    returning, and the construction-time `ValueError` that previously
    rejected `disentangle_*_weight > 0` together with multimodal data has
    been removed.

### Fixed

-   **`register_buffer("prototypes", ...)` crash on PyTorch 2.x** in
    `spVIPESmultimodule.__init__`. The previous code assigned `self.prototypes = None`
    before conditionally calling `register_buffer`; on PyTorch >= 2.x this
    raises `KeyError: "attribute 'prototypes' already exists"` on the second
    instantiation in the same process. Fix: always call `register_buffer`
    (with `None` in the off branch) so the attribute is owned by
    `nn.Module._buffers`. Affects any workflow that builds two `spVIPESmulti`
    models in the same process (e.g., `scripts/validate_disentanglement.py`,
    the ablation notebook). (PLANS.md P0, fix 1)
-   **`IndexError: tensors used as indices must be long, int, byte or bool`**
    in the contrastive prototype EMA update path
    (`_compute_disentangle_losses`, component 5). `labels_by_group[g]` is a
    float tensor (categorical codes from scvi-tools' dataloader); the other
    label-using branches all `.long()`-cast before use, but the contrastive
    branch missed the cast and `self.prototypes[g, lbl]` failed on float
    indices. Fix: cast once per group (`labels_g = labels_by_group[g].long()`)
    above the `unique()` loop. Unblocks `disentangle_preset='full'`,
    `'shared_only'`, and `'supervised_only'`, plus any custom config with
    `contrastive_weight > 0`. (PLANS.md P0, fix 2)

### Changed

-   Setting any of `disentangle_label_shared_weight`,
    `disentangle_label_private_weight`, or `contrastive_weight > 0` without
    `use_labels=True` now raises `ValueError` at construction time with a
    clear message. Group classifiers (`q_group_shared`, `q_group_private`)
    continue to work without labels — group identity is always known.
-   **All eight tutorial notebooks** (`Tutorial.ipynb`,
    `dialogue_multigroup_vignette.ipynb`, `iri_days_vignette.ipynb`,
    `cinemaot_nf_vignette.ipynb`, `pbmc_citeseq_tutorial.ipynb`,
    `biolord_comparison_plasmodium_tutorial.ipynb`,
    `multimodal_nf_tutorial.ipynb`, `disentangle_ablation.ipynb`) carry a
    **Requirements / Compatibility** callout pointing out the scvi-tools
    1.x training-kwarg change (`accelerator=` / `devices=` instead of the
    removed `use_gpu=`).
-   **`docs/index.md`** toctree links every shipped tutorial notebook
    (previously linked only `notebooks/example` (nonexistent) and
    `notebooks/dialogue_multigroup_vignette`).
-   **`README.md`** documents the multigrate-inspired multimodal kwargs
    (`modality_loss_weights`, `use_jeffreys_integ`, `jeffreys_integ_weight`)
    and updates the Disentanglement Objective constraint block to reflect
    that multimodal mode is now supported.
-   **Removed** `docs/notebooks/vignette_plan.md` (planning scratchpad
    self-marked for deletion once `multimodal_nf_tutorial.ipynb` landed).

### Notes

-   The six pertpy/local-h5ad-dependent notebooks (`Tutorial.ipynb`,
    `iri_days_vignette.ipynb`,
    `biolord_comparison_plasmodium_tutorial.ipynb`,
    `cinemaot_nf_vignette.ipynb`, `dialogue_multigroup_vignette.ipynb`,
    `disentangle_ablation.ipynb`) received markdown-only edits this round.
    Their existing rendered outputs were not regenerated, because pertpy
    1.x conflicts with the `jax==0.4.27` pin and the splatter / IRI /
    Plasmodium h5ad files are not bundled in-tree. Re-execution of those
    six is tracked separately.
