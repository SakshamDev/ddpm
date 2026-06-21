"""
Unit tests for the U-Net architecture.
"""

import pytest
import torch
from ddpm.unet import UNet

def test_tiny_unet_shape():
    """Verify that a tiny UNet outputs the correct shape and preserves batch/spatial dims."""
    B, C, H, W = 4, 3, 32, 32
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    
    # Instantiate a tiny UNet (runs fast on CPU)
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mults=(1, 2),
        attention_resolutions=(1,),
        num_res_blocks=1,
        time_emb_dim=128
    )
    
    out = model(x, t)
    
    assert out.shape == (B, C, H, W), f"Expected {(B, C, H, W)}, got {out.shape}"


def test_unet_device_transfer():
    """Verify that UNet correctly moves all internal structures to target device."""
    model = UNet(
        base_channels=16,
        channel_mults=(1, 2),
        attention_resolutions=(),
        num_res_blocks=1,
        time_emb_dim=64
    )
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        return # Skip if no accelerator
        
    model = model.to(device)
    
    x = torch.randn(2, 3, 16, 16).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    
    # This will fail if internal constants (like sinusoidal freq bands) aren't on device
    out = model(x, t)
    assert out.device.type == device.type


def test_parameter_count():
    """Verify that the tiny UNet is small enough for fast prototyping (< 1M params)."""
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mults=(1, 2),
        attention_resolutions=(1,),
        num_res_blocks=1,
        time_emb_dim=128
    )
    
    params = sum(p.numel() for p in model.parameters())
    assert params < 1_000_000, f"Tiny UNet is too big: {params} parameters"
