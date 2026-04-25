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
-   **MIG-based disentanglement objective** (inspired by CellDISECT and Multi-ContrastiveVAE).
    Four auxiliary classifiers enforce explicit mutual-information gaps between latent spaces:

    | Classifier | Input | Goal |
    |---|---|---|
    | `q_label_shared` | `z_shared` | Preserve cell-type info (supervised) |
    | `q_group_shared` | `z_shared` | Erase group identity (adversarial, GRL) |
    | `q_group_private` | `z_private` | Preserve group info (supervised) |
    | `q_label_private` | `z_private` | Erase cell-type info (adversarial, GRL) |

    This maximises two information gaps simultaneously:
    `MIG_shared = I(z_shared; label) − I(z_shared; group)` and
    `MIG_private = I(z_private; group) − I(z_private; label)`.

    All weights default to `0.0` (fully backward-compatible). New `spVIPES` constructor
    parameters: `mig_group_shared_weight`, `mig_label_shared_weight`,
    `mig_group_private_weight`, `mig_label_private_weight`, `contrastive_weight`,
    `contrastive_temperature`.

-   **Prototype supervised contrastive loss** on `z_shared` (optional, `contrastive_weight > 0`).
    EMA prototypes per (group × label) are maintained; supervised InfoNCE pulls same-label
    cells together across groups.

-   `gradient_reversal()` utility function and `_GradientReversalFunction` added to
    `spVIPES.module.utils`.
-   **`mig_preset` kwarg** on `spVIPES` model, with named configurations:
    `"off"` (default), `"full"`, `"shared_only"`, `"private_only"`,
    `"adversarial_only"`, `"supervised_only"`, `"no_contrastive"`.
    Per-component weight kwargs (`mig_*_weight`, `contrastive_weight`) now act
    as overrides on top of the preset. `None` (the default) means "use the
    preset's value"; numeric values (including `0.0`) override.
-   Ablation walkthrough notebook (`docs/notebooks/mig_ablation.ipynb`) showing
    how to enable/disable each MIG component via presets and per-weight overrides.
-   `_compute_mig_losses()` helper method on `spVIPESmodule` — extracts the MIG
    loss block from `loss()` for cleaner ablation workflow. Each component
    reads as one isolated `if self.q_X is not None` branch.

### Changed

-   Setting any of `mig_label_shared_weight`, `mig_label_private_weight`, or
    `contrastive_weight > 0` without `use_labels=True` now raises `ValueError`
    at construction time with a clear message. Group classifiers
    (`q_group_shared`, `q_group_private`) continue to work without labels —
    group identity is always known.

### Notes

-   MIG is currently single-modality only. Multimodal mode (`is_multimodal=True`)
    bypasses the MIG block; tracked in `PLANS.md` as P8.
