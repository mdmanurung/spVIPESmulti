"""Named presets for the MIG (Mutual Information Gap) objective.

Each preset is a dict of weight overrides. Users select a preset via
``mig_preset=...`` on the :class:`~spVIPES.model.spVIPES` model, and may
further override individual weights by passing them explicitly. ``None``
on a per-weight kwarg means "use the preset's value"; a numeric value
(including ``0.0``) overrides the preset for that component.
"""

MIG_PRESETS = {
    "off": {
        "mig_group_shared_weight": 0.0,
        "mig_label_shared_weight": 0.0,
        "mig_group_private_weight": 0.0,
        "mig_label_private_weight": 0.0,
        "contrastive_weight": 0.0,
    },
    "full": {
        # All 4 classifiers + contrastive at sensible defaults.
        "mig_group_shared_weight": 1.0,
        "mig_label_shared_weight": 1.0,
        "mig_group_private_weight": 1.0,
        "mig_label_private_weight": 1.0,
        "contrastive_weight": 0.5,
    },
    "shared_only": {
        # Only the z_shared MIG: I(z_shared; label) - I(z_shared; group).
        "mig_group_shared_weight": 1.0,
        "mig_label_shared_weight": 1.0,
        "mig_group_private_weight": 0.0,
        "mig_label_private_weight": 0.0,
        "contrastive_weight": 0.5,
    },
    "private_only": {
        # Only the z_private MIG: I(z_private; group) - I(z_private; label).
        "mig_group_shared_weight": 0.0,
        "mig_label_shared_weight": 0.0,
        "mig_group_private_weight": 1.0,
        "mig_label_private_weight": 1.0,
        "contrastive_weight": 0.0,
    },
    "adversarial_only": {
        # Only the GRL-based components.
        "mig_group_shared_weight": 1.0,
        "mig_label_shared_weight": 0.0,
        "mig_group_private_weight": 0.0,
        "mig_label_private_weight": 1.0,
        "contrastive_weight": 0.0,
    },
    "supervised_only": {
        # Only the non-GRL components (label-shared, group-private, contrastive).
        "mig_group_shared_weight": 0.0,
        "mig_label_shared_weight": 1.0,
        "mig_group_private_weight": 1.0,
        "mig_label_private_weight": 0.0,
        "contrastive_weight": 0.5,
    },
    "no_contrastive": {
        # Full MIG but with contrastive disabled.
        "mig_group_shared_weight": 1.0,
        "mig_label_shared_weight": 1.0,
        "mig_group_private_weight": 1.0,
        "mig_label_private_weight": 1.0,
        "contrastive_weight": 0.0,
    },
}
