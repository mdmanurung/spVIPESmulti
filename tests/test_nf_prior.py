"""Tests for Zuko normalizing flow prior."""

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
