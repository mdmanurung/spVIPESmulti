# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

-   Basic tool, preprocessing and plotting functions
-   DIALOGUE-style multi-group vignette (`docs/notebooks/dialogue_multigroup_vignette.ipynb`)
    showcasing N ≥ 2 group integration using `clinical.status` from `pt.dt.dialogue_example()`.
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
    `spVIPESmodule` parameters: `disentangle_group_shared_weight`,
    `disentangle_label_shared_weight`, `disentangle_group_private_weight`,
    `disentangle_label_private_weight`, `contrastive_weight`,
    `contrastive_temperature`.

-   **Prototype supervised contrastive loss** on `z_shared` (optional,
    `contrastive_weight > 0`). EMA prototypes per (group × label) are
    maintained; supervised InfoNCE pulls same-label cells together across
    groups.

-   `gradient_reversal()` utility function and `_GradientReversalFunction`
    added to `spVIPES.module.utils`.

-   **`disentangle_preset` kwarg** on `spVIPES` model, with named
    configurations: `"off"` (default), `"full"`, `"shared_only"`,
    `"private_only"`, `"adversarial_only"`, `"supervised_only"`,
    `"no_contrastive"`. Per-component weight kwargs
    (`disentangle_*_weight`, `contrastive_weight`) now act as overrides on
    top of the preset. `None` (the default) means "use the preset's value";
    numeric values (including `0.0`) override.

-   Ablation walkthrough notebook (`docs/notebooks/disentangle_ablation.ipynb`)
    showing how to enable/disable each component via presets and per-weight
    overrides.

-   `_compute_disentangle_losses()` helper method on `spVIPESmodule` —
    extracts the disentanglement loss block from `loss()` for cleaner
    ablation workflow. Each component reads as one isolated
    `if self.q_X is not None` branch.

### Changed

-   Setting any of `disentangle_label_shared_weight`,
    `disentangle_label_private_weight`, or `contrastive_weight > 0` without
    `use_labels=True` now raises `ValueError` at construction time with a
    clear message. Group classifiers (`q_group_shared`, `q_group_private`)
    continue to work without labels — group identity is always known.
-   Setting any `disentangle_*_weight` or `contrastive_weight > 0` together
    with multimodal data (`is_multimodal=True`) now raises `ValueError` at
    model construction. Previously this combination silently bypassed the
    disentanglement losses. Multimodal disentanglement support is tracked as
    P8 in `PLANS.md`.

### Notes

-   The disentanglement objective is currently single-modality only.
    Multimodal mode (`is_multimodal=True`) bypasses the disentanglement
    block; tracked in `PLANS.md` as P8.
