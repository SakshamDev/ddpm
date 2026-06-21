"""
Unit tests for the noise schedule.
"""

import pytest
import torch
from ddpm.noise_schedule import NoiseSchedule

def test_linear_schedule_bounds():
    schedule = NoiseSchedule(schedule_type="linear", timesteps=1000)
    
    assert schedule.betas.shape == (1000,)
    # Linear schedule should exactly match bounds
    assert torch.allclose(schedule.betas[0], torch.tensor(0.0001))
    assert torch.allclose(schedule.betas[-1], torch.tensor(0.02))
    
    # ᾱ_t should be strictly decreasing
    diffs = schedule.alphas_cumprod[1:] - schedule.alphas_cumprod[:-1]
    assert torch.all(diffs < 0), "alphas_cumprod must be strictly decreasing"


def test_cosine_schedule_bounds():
    schedule = NoiseSchedule(schedule_type="cosine", timesteps=1000)
    
    assert schedule.betas.shape == (1000,)
    # Beta should be clamped to 0.999 max
    assert torch.max(schedule.betas) <= 0.999
    
    # ᾱ_t should be strictly decreasing
    diffs = schedule.alphas_cumprod[1:] - schedule.alphas_cumprod[:-1]
    assert torch.all(diffs < 0), "alphas_cumprod must be strictly decreasing"


def test_buffer_registration():
    schedule = NoiseSchedule()
    # Ensure they are registered as buffers, not parameters
    params = list(schedule.parameters())
    assert len(params) == 0
    
    buffers = dict(schedule.named_buffers())
    assert "betas" in buffers
    assert "alphas_cumprod" in buffers
    
    # Moving module should move buffers
    if torch.cuda.is_available():
        schedule.to("cuda")
        assert schedule.betas.device.type == "cuda"
