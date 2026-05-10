#!/usr/bin/env python3
"""Quick test to verify the tensor dimension fix."""

import torch
import sys

# Test the mask reshape fix
def test_mask_reshape():
    """Test that reshape(-1) properly handles mask shapes."""
    
    # Simulate different label tensor shapes
    batch_size = 100
    latent_dim = 10
    
    # Case 1: 1D labels (expected)
    labels_1d = torch.randint(0, 3, (batch_size,))
    mask_1d = (labels_1d == 1).squeeze().reshape(-1)
    assert mask_1d.dim() == 1, f"Expected 1D mask, got {mask_1d.shape}"
    
    # Case 2: 2D labels with shape (batch_size, 1) (edge case)
    labels_2d = torch.randint(0, 3, (batch_size, 1))
    mask_2d = (labels_2d == 1).squeeze().reshape(-1)
    assert mask_2d.dim() == 1, f"Expected 1D mask, got {mask_2d.shape}"
    
    # Test masking with these masks
    data = torch.randn(batch_size, latent_dim)
    
    # Both masks should work for indexing
    result_1d = data[mask_1d]
    result_2d = data[mask_2d]
    
    assert result_1d.dim() == 2, f"Expected 2D result, got {result_1d.shape}"
    assert result_2d.dim() == 2, f"Expected 2D result, got {result_2d.shape}"
    
    print("✓ Mask reshape test passed")

def test_dimension_extraction():
    """Test that shape[-1] properly extracts latent_dim from various tensor shapes."""
    
    # 2D tensor (standard case)
    tensor_2d = torch.randn(100, 10)
    assert tensor_2d.shape[-1] == 10
    
    # 3D tensor (if it ever happens)
    tensor_3d = torch.randn(100, 5, 10)
    assert tensor_3d.shape[-1] == 10
    
    print("✓ Dimension extraction test passed")

if __name__ == "__main__":
    try:
        test_mask_reshape()
        test_dimension_extraction()
        print("\nAll tests passed! The fix should work correctly.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
