"""
Unit tests for forward diffusion.
"""

import pytest
import torch
from ddpm.noise_schedule import NoiseSchedule
from ddpm.forward_diffusion import extract, q_sample

def test_extract_shapes():
    a = torch.arange(10).float()
    t = torch.tensor([1, 5, 9])
    x_shape = (3, 3, 32, 32)
    
    out = extract(a, t, x_shape)
    
    assert out.shape == (3, 1, 1, 1)
    assert out[0, 0, 0, 0].item() == 1.0
    assert out[1, 0, 0, 0].item() == 5.0
    assert out[2, 0, 0, 0].item() == 9.0


def test_q_sample_limits():
    B, C, H, W = 4, 3, 8, 8
    x_start = torch.ones(B, C, H, W)
    noise = torch.zeros(B, C, H, W)  # zero noise for predictable math
    schedule = NoiseSchedule(timesteps=1000)
    
    # At t=0, signal should be very high (x_t ≈ x_start)
    t_zero = torch.zeros(B, dtype=torch.long)
    x_t_zero = q_sample(x_start, t_zero, schedule, noise=noise)
    
    # √ᾱ_0 is very close to 1
    assert torch.allclose(x_t_zero, x_start * schedule.sqrt_alphas_cumprod[0])
    
    # At t=999, signal should be very low (x_t ≈ noise, which is 0 here)
    t_end = torch.full((B,), 999, dtype=torch.long)
    x_t_end = q_sample(x_start, t_end, schedule, noise=noise)
    
    # Result should be close to zero since signal is suppressed and noise=0
    assert torch.abs(x_t_end).max() < 0.05
