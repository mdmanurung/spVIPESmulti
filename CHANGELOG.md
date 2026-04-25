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
