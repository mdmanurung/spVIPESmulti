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

__all__ = [
    "CounterfactualResult",
    "condition_centroid_shift",
    "condition_separability",
    "decode_counterfactual",
    "edit_latent",
    "encode_cells",
    "integration_report",
    "latent_arithmetic",
    "latent_interpolation",
    "latent_replacement",
    "latent_variance_utilization",
    "leakage_score",
    "predict_counterfactual",
    "transfer_condition",
]
