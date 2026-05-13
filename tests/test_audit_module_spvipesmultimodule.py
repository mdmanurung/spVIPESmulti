from __future__ import annotations

import numpy as np
import pytest
import torch
from scvi import REGISTRY_KEYS
from torch.distributions import Normal

from spVIPESmulti.module.spVIPESmultimodule import spVIPESmultimodule


class _RecordingDistribution:
    """Distribution stub that records the value passed to log_prob."""

    def __init__(self) -> None:
        self.value: torch.Tensor | None = None

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Record the target tensor and return a zero reconstruction term."""

        self.value = value.detach().clone()
        return torch.zeros_like(value)


def _minimal_uninitialized_module() -> spVIPESmultimodule:
    """Create a loss-only module instance without running the heavy constructor."""

    module = object.__new__(spVIPESmultimodule)
    torch.nn.Module.__init__(module)
    module.is_multimodal = False
    module.groups_var_indices = [np.array([0, 1])]
    module.log_variational_generative = True
    module.validate_observations = False
    module.strict_likelihood_support = False
    module.use_nf_prior = False
    module.nf_target = "shared"
    module.group_loss_weights = None
    module.disentangle_warmup = False
    module.use_jeffreys_integ = False
    module._compute_disentangle_losses = lambda *args: torch.zeros((), dtype=torch.float32)
    return module


def _minimal_constructed_module() -> spVIPESmultimodule:
    """Create a small real module for inference-path tests."""

    return spVIPESmultimodule(
        groups_lengths={0: 2, 1: 2},
        groups_obs_names=[[0, 1], [0, 1]],
        groups_var_names=[["g0", "g1"], ["g0", "g1"]],
        groups_obs_indices=[[0, 1], [0, 1]],
        groups_var_indices=[[0, 1], [0, 1]],
        n_hidden=8,
        n_dimensions_private=2,
        n_dimensions_shared=2,
        use_batch_norm=False,
        use_layer_norm=False,
    )


@pytest.mark.xfail(strict=True, reason="INT-004 pending: single-modal all-zero cells produce -inf library")
def test_single_modal_all_zero_library_is_finite() -> None:
    """All-zero cells should not create non-finite observed library tensors."""

    module = _minimal_constructed_module()
    module.eval()
    x = {0: torch.zeros((2, 2), dtype=torch.float32), 1: torch.ones((2, 2), dtype=torch.float32)}
    batch_index = [torch.zeros((2, 1), dtype=torch.long), torch.zeros((2, 1), dtype=torch.long)]
    groups = [torch.zeros((2, 1), dtype=torch.long), torch.ones((2, 1), dtype=torch.long)]

    outputs = module.inference(x=x, batch_index=batch_index, groups=groups, global_indices=[None, None])

    assert torch.isfinite(outputs["library"][0]).all()


@pytest.mark.xfail(strict=True, reason="INT-003 pending: NB reconstruction uses log1p targets")
def test_negative_binomial_loss_uses_raw_count_targets() -> None:
    """Negative-binomial reconstruction should evaluate log_prob on raw counts."""

    module = _minimal_uninitialized_module()
    recorder = _RecordingDistribution()
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    zeros = torch.zeros((2, 2), dtype=torch.float32)
    qz = Normal(zeros, torch.ones_like(zeros))

    module.loss(
        tensors_by_group=[{REGISTRY_KEYS.X_KEY: x}],
        inference_outputs={
            "private_stats": {0: {"qz": qz, "log_z": zeros}},
            "poe_stats": {0: {"logtheta_qz": qz, "logtheta_log_z": zeros}},
        },
        generative_outputs={"private_poe": {"0": {"px": recorder}}},
        kl_weight=1.0,
    )

    assert recorder.value is not None
    torch.testing.assert_close(recorder.value, x)
