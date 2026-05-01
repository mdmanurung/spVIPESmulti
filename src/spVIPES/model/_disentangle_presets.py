"""Named presets for the disentanglement objective.

The disentanglement losses are inspired by CellDISECT's cross-covariate
decoupling MLPs and Multi-ContrastiveVAE. They are not the original
"Mutual Information Gap" metric (Chen et al. 2018), which is a post-hoc
disentanglement *metric*. What we implement here is a mix of:

  * adversarial domain-invariance losses via gradient reversal (DANN-style),
  * supervised classification losses acting as variational MI lower bounds,
  * an optional prototype InfoNCE on the shared latent.

Each preset is a dict of weight overrides. Users select a preset via
``disentangle_preset=...`` on the :class:`~spVIPES.model.spVIPES` model, and
may further override individual weights by passing them explicitly. ``None``
on a per-weight kwarg means "use the preset's value"; a numeric value
(including ``0.0``) overrides the preset for that component.
"""

DISENTANGLE_PRESETS: dict[str, dict[str, float]] = {
    "off": {
        "disentangle_group_shared_weight": 0.0,
        "disentangle_label_shared_weight": 0.0,
        "disentangle_group_private_weight": 0.0,
        "disentangle_label_private_weight": 0.0,
        "contrastive_weight": 0.0,
    },
    "full": {
        # All 4 classifiers + contrastive at sensible defaults.
        "disentangle_group_shared_weight": 1.0,
        "disentangle_label_shared_weight": 1.0,
        "disentangle_group_private_weight": 1.0,
        "disentangle_label_private_weight": 1.0,
        "contrastive_weight": 0.5,
    },
    "shared_only": {
        # Only the z_shared decoupling losses.
        "disentangle_group_shared_weight": 1.0,
        "disentangle_label_shared_weight": 1.0,
        "disentangle_group_private_weight": 0.0,
        "disentangle_label_private_weight": 0.0,
        "contrastive_weight": 0.5,
    },
    "private_only": {
        # Only the z_private decoupling losses.
        "disentangle_group_shared_weight": 0.0,
        "disentangle_label_shared_weight": 0.0,
        "disentangle_group_private_weight": 1.0,
        "disentangle_label_private_weight": 1.0,
        "contrastive_weight": 0.0,
    },
    "adversarial_only": {
        # Only the GRL-based components.
        "disentangle_group_shared_weight": 1.0,
        "disentangle_label_shared_weight": 0.0,
        "disentangle_group_private_weight": 0.0,
        "disentangle_label_private_weight": 1.0,
        "contrastive_weight": 0.0,
    },
    "supervised_only": {
        # Only the non-GRL components (label-shared, group-private, contrastive).
        "disentangle_group_shared_weight": 0.0,
        "disentangle_label_shared_weight": 1.0,
        "disentangle_group_private_weight": 1.0,
        "disentangle_label_private_weight": 0.0,
        "contrastive_weight": 0.5,
    },
    "no_contrastive": {
        # Full disentanglement but with contrastive disabled.
        "disentangle_group_shared_weight": 1.0,
        "disentangle_label_shared_weight": 1.0,
        "disentangle_group_private_weight": 1.0,
        "disentangle_label_private_weight": 1.0,
        "contrastive_weight": 0.0,
    },
}

_REQUIRED_PRESET_KEYS = frozenset({
    "disentangle_group_shared_weight",
    "disentangle_label_shared_weight",
    "disentangle_group_private_weight",
    "disentangle_label_private_weight",
    "contrastive_weight",
})

for _preset_name, _preset_vals in DISENTANGLE_PRESETS.items():
    _missing = _REQUIRED_PRESET_KEYS - _preset_vals.keys()
    if _missing:
        raise RuntimeError(
            f"DISENTANGLE_PRESETS['{_preset_name}'] is missing required keys: {_missing}"
        )
