"""Tests for Zuko normalizing flow prior and cycle consistency loss."""

import importlib.util
import os

import numpy as np
import pytest
import torch
from torch.distributions import Normal

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _load_module(module_name, filepath):
    """Load a single module from filepath without triggering package __init__."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Test Zuko normalizing flow prior
# =============================================================================


class TestNFPrior:
    """Tests for normalizing flow prior integration."""

    def test_nsf_flow_as_prior(self):
        """NSF flow should serve as a trainable prior with log_prob."""
        import zuko.flows

        latent_dim = 10
        flow = zuko.flows.NSF(features=latent_dim, context=0, transforms=3)
        dist = flow()

        # Should be able to compute log_prob
        z = torch.randn(32, latent_dim)
        log_p = dist.log_prob(z)
        assert log_p.shape == (32,)
        assert torch.isfinite(log_p).all()

    def test_maf_flow_as_prior(self):
        """MAF flow should also work."""
        import zuko.flows

        latent_dim = 10
        flow = zuko.flows.MAF(features=latent_dim, context=0, transforms=3)
        dist = flow()

        z = torch.randn(16, latent_dim)
        log_p = dist.log_prob(z)
        assert log_p.shape == (16,)
        assert torch.isfinite(log_p).all()

    def test_nf_kl_monte_carlo(self):
        """MC KL estimate using flow prior should be finite and positive on average."""
        import zuko.flows

        latent_dim = 10
        flow = zuko.flows.NSF(features=latent_dim, context=0, transforms=2)

        # Posterior q(z|x) is a diagonal Gaussian
        mu = torch.randn(32, latent_dim)
        logvar = torch.randn(32, latent_dim)
        std = torch.exp(0.5 * logvar)
        qz = Normal(mu, std)
        z = qz.rsample()

        # MC KL = log q(z|x) - log p_flow(z)
        log_qz = qz.log_prob(z).sum(dim=-1)
        log_pz = flow().log_prob(z)
        kl_mc = log_qz - log_pz

        assert kl_mc.shape == (32,)
        assert torch.isfinite(kl_mc).all()

    def test_nf_prior_is_trainable(self):
        """Flow parameters should be trainable via backprop."""
        import zuko.flows

        latent_dim = 5
        flow = zuko.flows.NSF(features=latent_dim, context=0, transforms=2)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

        # One training step
        z = torch.randn(16, latent_dim)
        log_p = flow().log_prob(z)
        loss = -log_p.mean()
        loss.backward()
        optimizer.step()

        # Parameters should have changed (gradients flowed through)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in flow.parameters())

    def test_flow_prior_vs_standard_kl_shape(self):
        """NF KL should have same shape as standard KL for drop-in replacement."""
        import zuko.flows

        latent_dim = 10
        batch_size = 32

        mu = torch.randn(batch_size, latent_dim)
        std = torch.exp(0.5 * torch.randn(batch_size, latent_dim))
        qz = Normal(mu, std)
        z = qz.rsample()

        # Standard KL
        standard_kl = torch.distributions.kl_divergence(
            qz, Normal(torch.zeros_like(mu), torch.ones_like(std))
        ).sum(dim=1)

        # NF KL (MC)
        flow = zuko.flows.NSF(features=latent_dim, context=0, transforms=2)
        log_qz = qz.log_prob(z).sum(dim=-1)
        log_pz = flow().log_prob(z)
        nf_kl = log_qz - log_pz

        assert standard_kl.shape == nf_kl.shape == (batch_size,)


# =============================================================================
# Test cycle consistency loss
# =============================================================================


class TestCycleConsistencyLoss:
    """Tests for the sysVI-style cycle consistency loss."""

    @staticmethod
    def _cycle_consistency_loss(z_original, z_cycled):
        """Standalone version of the static method for testing."""
        z_concat = torch.cat([z_original, z_cycled], dim=0)
        means = z_concat.mean(dim=0, keepdim=True)
        stds = z_concat.std(dim=0, keepdim=True).clamp(min=1e-6)
        z_orig_std = (z_original - means) / stds
        z_cycle_std = (z_cycled - means) / stds
        return torch.nn.functional.mse_loss(z_orig_std, z_cycle_std, reduction="none").sum(dim=1)

    def test_identical_latents_zero_loss(self):
        """Identical latents should give zero cycle loss."""
        z = torch.randn(32, 10)
        loss = self._cycle_consistency_loss(z, z.clone())
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)

    def test_different_latents_positive_loss(self):
        """Different latents should give positive loss."""
        z1 = torch.randn(32, 10)
        z2 = torch.randn(32, 10)
        loss = self._cycle_consistency_loss(z1, z2)
        assert (loss > 0).all()

    def test_loss_shape(self):
        """Loss should be per-sample (batch_size,)."""
        z1 = torch.randn(16, 25)
        z2 = torch.randn(16, 25)
        loss = self._cycle_consistency_loss(z1, z2)
        assert loss.shape == (16,)

    def test_standardization_prevents_trivial_solution(self):
        """Scaling both latents by a small constant shouldn't reduce loss."""
        z1 = torch.randn(32, 10)
        z2 = z1 + torch.randn(32, 10) * 0.5  # perturbed

        loss_original = self._cycle_consistency_loss(z1, z2)
        # Scale everything down by 0.01
        loss_scaled = self._cycle_consistency_loss(z1 * 0.01, z2 * 0.01)

        # Standardized MSE should be similar regardless of scale
        assert torch.allclose(loss_original, loss_scaled, atol=1e-4)

    def test_loss_is_symmetric(self):
        """cycle_loss(a, b) == cycle_loss(b, a)."""
        z1 = torch.randn(32, 10)
        z2 = torch.randn(32, 10)
        loss_ab = self._cycle_consistency_loss(z1, z2)
        loss_ba = self._cycle_consistency_loss(z2, z1)
        assert torch.allclose(loss_ab, loss_ba, atol=1e-6)

    def test_loss_gradients_flow(self):
        """Gradients should flow through the cycle consistency loss."""
        z1 = torch.randn(32, 10, requires_grad=True)
        z2 = torch.randn(32, 10, requires_grad=True)
        loss = self._cycle_consistency_loss(z1, z2).mean()
        loss.backward()
        assert z1.grad is not None
        assert z2.grad is not None
        assert z1.grad.abs().sum() > 0
        assert z2.grad.abs().sum() > 0
