"""Safe latent intervention helpers for spVIPESmulti.

The functions in this package expose deterministic encode/edit/decode utilities for
single-modal models. They are diagnostic, associative predictions and should not be
interpreted as causal effects without external interventional validation.
"""

from .counterfactual import (
    CounterfactualResult,
    decode_counterfactual,
    edit_latent,
    encode_cells,
    predict_counterfactual,
    transfer_condition,
)
from .diagnostics import (
    condition_separability,
    integration_report,
    latent_variance_utilization,
    leakage_score,
)
from .latent_operators import (
    condition_centroid_shift,
    latent_arithmetic,
    latent_interpolation,
    latent_replacement,
)
from .protocols import (
    donor_condition_shift,
    private_swap_label_matched,
    private_swap_stratified,
    private_swap_unmatched,
)

__all__ = [
    "CounterfactualResult",
    "condition_centroid_shift",
    "condition_separability",
    "decode_counterfactual",
    "donor_condition_shift",
    "edit_latent",
    "encode_cells",
    "integration_report",
    "latent_arithmetic",
    "latent_interpolation",
    "latent_replacement",
    "latent_variance_utilization",
    "leakage_score",
    "predict_counterfactual",
    "private_swap_label_matched",
    "private_swap_stratified",
    "private_swap_unmatched",
    "transfer_condition",
]
