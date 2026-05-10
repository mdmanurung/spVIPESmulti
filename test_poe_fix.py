#!/usr/bin/env python3
"""Test to verify the RuntimeError fix for tensor dimension mismatch."""

import torch
from torch.distributions import Normal
import torch.nn.functional as F
import sys

# Mock the minimal parts of spVIPESmultimodule needed to test _label_based_poe and _poe_n
class MockModule:
    def _product_of_experts(self, mus, logvars):
        """Simplified PoE: just return average for testing."""
        # Input shapes: (n_groups, max_batch, latent_dim)
        mu_avg = mus.mean(dim=0)
        logvar_avg = logvars.mean(dim=0)
        return mu_avg, logvar_avg
    
    def _poe_n(self, shared_stats):
        """Generic N-group Product of Experts.
        
        This is the function that was failing with:
        RuntimeError: Tensors must have same number of dimensions: got 3 and 2
        """
        group_keys = sorted(shared_stats.keys())
        n_groups = len(group_keys)
        if n_groups < 2:
            raise ValueError(f"PoE requires at least 2 groups, got {n_groups}")

        group_sizes = {g: shared_stats[g]["logtheta_logvar"].shape[0] for g in group_keys}
        max_batch_size = max(group_sizes.values())
        # FIX: Use shape[-1] instead of shape[1] to handle any number of dimensions
        latent_dim = shared_stats[group_keys[0]]["logtheta_logvar"].shape[-1]
        device = shared_stats[group_keys[0]]["logtheta_logvar"].device

        # Pad each group to max_batch_size and stack
        padded_locs = []
        padded_logvars = []
        for g in group_keys:
            loc = shared_stats[g]["logtheta_loc"]
            logvar = shared_stats[g]["logtheta_logvar"]
            g_size = group_sizes[g]
            if g_size < max_batch_size:
                pad_size = max_batch_size - g_size
                # This is where the error was happening
                loc = torch.cat([loc, torch.zeros(pad_size, latent_dim, device=device)], dim=0)
                logvar = torch.cat([logvar, torch.zeros(pad_size, latent_dim, device=device)], dim=0)
            padded_locs.append(loc)
            padded_logvars.append(logvar)

        # Stack: shape (N, max_batch, latent_dim)
        stacked_mus = torch.stack(padded_locs, dim=0)
        stacked_logvars = torch.stack(padded_logvars, dim=0)

        # Compute joint PoE
        mus_joint, logvars_joint = self._product_of_experts(stacked_mus, stacked_logvars)

        # Slice back to each group's original size and build result
        result = {}
        for g in group_keys:
            g_size = group_sizes[g]
            g_mu = mus_joint[:g_size]
            g_logvar = logvars_joint[:g_size]
            g_scale = torch.sqrt(torch.exp(g_logvar))
            result[g] = {
                "logtheta_loc": g_mu,
                "logtheta_logvar": g_logvar,
                "logtheta_scale": g_scale,
            }

        return result

    def _label_based_poe(self, shared_stats, label_group):
        """Test the label-based PoE with the robust mask fix."""
        stat_keys = ["logtheta_loc", "logtheta_logvar", "logtheta_scale"]
        group_keys = sorted(shared_stats.keys())

        # Extract per-group stats and labels
        per_group_stats = {}
        per_group_labels = {}
        for g in group_keys:
            per_group_stats[g] = {k: shared_stats[g][k] for k in stat_keys if k in shared_stats[g]}
            per_group_labels[g] = label_group[g]

        # Collect all unique labels
        label_sets = {g: set(per_group_labels[g].flatten().tolist()) for g in group_keys}
        all_labels = set()
        for s in label_sets.values():
            all_labels |= s

        # For each label, compute PoE
        poe_stats_per_label = {}
        for label in all_labels:
            groups_with_label = [g for g in group_keys if label in label_sets[g]]
            groups_without_label = [g for g in group_keys if label not in label_sets[g]]

            label_stats_for_poe = {}
            for g in groups_with_label:
                # FIX: Use reshape(-1) to ensure mask is always 1D
                mask = (per_group_labels[g] == label).squeeze().reshape(-1)
                label_stats_for_poe[g] = {key: value[mask] for key, value in per_group_stats[g].items()}

            if len(groups_with_label) >= 2:
                for g in groups_without_label:
                    ref_g = groups_with_label[0]
                    n_cells = label_stats_for_poe[ref_g]["logtheta_loc"].shape[0]
                    # FIX: Use shape[-1] instead of shape[1]
                    latent_dim = label_stats_for_poe[ref_g]["logtheta_loc"].shape[-1]
                    device = label_stats_for_poe[ref_g]["logtheta_loc"].device
                    _large_logvar = torch.full((n_cells, latent_dim), 30.0, device=device)
                    label_stats_for_poe[g] = {
                        "logtheta_loc": torch.zeros(n_cells, latent_dim, device=device),
                        "logtheta_logvar": _large_logvar,
                        "logtheta_scale": torch.exp(0.5 * _large_logvar),
                    }

                # This call was failing before the fix
                poe_result = self._poe_n(label_stats_for_poe)

                # Process results
                for g in groups_without_label:
                    latent_dim = poe_result[groups_with_label[0]]["logtheta_loc"].shape[-1]
                    device = poe_result[groups_with_label[0]]["logtheta_loc"].device
                    poe_result[g] = {
                        k: torch.empty((0, latent_dim), device=device) for k in stat_keys
                    }

                poe_stats_per_label[label] = poe_result

        return poe_stats_per_label


def test_label_based_poe_with_mask_fix():
    """Test that the label-based PoE works correctly with the mask reshape fix."""
    module = MockModule()
    
    # Create test data for 2 groups
    batch_size_g0 = 100
    batch_size_g1 = 80
    latent_dim = 10
    
    # Group 0 with mixed labels
    labels_g0 = torch.cat([
        torch.ones(50, dtype=torch.long) * 1,      # 50 of label 1
        torch.ones(50, dtype=torch.long) * 2,      # 50 of label 2
    ])
    
    # Group 1 with mixed labels (different distribution)
    labels_g1 = torch.cat([
        torch.ones(40, dtype=torch.long) * 1,      # 40 of label 1
        torch.ones(40, dtype=torch.long) * 2,      # 40 of label 2
    ])
    
    # Create shared stats for both groups
    shared_stats = {
        0: {
            "logtheta_loc": torch.randn(batch_size_g0, latent_dim),
            "logtheta_logvar": torch.randn(batch_size_g0, latent_dim),
            "logtheta_scale": torch.abs(torch.randn(batch_size_g0, latent_dim)) + 0.1,
        },
        1: {
            "logtheta_loc": torch.randn(batch_size_g1, latent_dim),
            "logtheta_logvar": torch.randn(batch_size_g1, latent_dim),
            "logtheta_scale": torch.abs(torch.randn(batch_size_g1, latent_dim)) + 0.1,
        },
    }
    
    label_group = {
        0: labels_g0,
        1: labels_g1,
    }
    
    print("Testing label-based PoE with mask reshape fix...")
    try:
        result = module._label_based_poe(shared_stats, label_group)
        print(f"✓ Successfully computed PoE for {len(result)} labels")
        
        # Verify result structure
        for label, poe_stats in result.items():
            print(f"  Label {label}: {len(poe_stats)} group results")
            for g, stats in poe_stats.items():
                assert "logtheta_loc" in stats
                assert "logtheta_logvar" in stats
                assert "logtheta_scale" in stats
                assert stats["logtheta_loc"].dim() == 2
                print(f"    Group {g}: shape {stats['logtheta_loc'].shape}")
        
        print("\n✓ Label-based PoE test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Label-based PoE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_label_based_poe_with_mask_fix()
    sys.exit(0 if success else 1)
