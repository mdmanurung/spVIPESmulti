from __future__ import annotations

import pytest
import torch

from spVIPESmulti.module.spVIPESmultimodule import _within_stratum_corr_norm


@pytest.mark.skip(reason="NAT-001 pending: native/compiled implementation has not landed")
def test_nat_001_parity_python_vs_native_placeholder() -> None:
    """Exercise the Python path that a future native implementation must match."""

    z_shared = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    z_private = torch.tensor([[0.0, 2.0], [1.0, 1.0], [2.0, 2.0], [3.0, 1.0]])
    strata = torch.tensor([0, 0, 1, 1])

    mean_score, worst_score, excluded = _within_stratum_corr_norm(
        z_shared,
        z_private,
        strata,
        min_cells=2,
    )

    assert mean_score >= 0.0
    assert worst_score >= mean_score
    assert excluded == 0
