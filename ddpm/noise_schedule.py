"""
Noise schedules for DDPM.

Defines the variance schedule (beta) and derived quantities (alpha, alpha_bar)
used in the forward diffusion process.

References:
  - Ho et al. (2020) Eq. 4: Linear schedule
  - Nichol & Dhariwal (2021) Eq. 17: Cosine schedule
"""

import math
import torch
import torch.nn as nn


class NoiseSchedule(nn.Module):
    """
    Computes and holds all diffusion schedule tensors.
    
    Subclasses nn.Module and uses register_buffer so that when the parent U-Net
    is moved to GPU (model.to(device)), these schedule tensors move with it,
    preventing device mismatch errors during training.
    """

    def __init__(self, schedule_type: str = "cosine", timesteps: int = 1000):
        super().__init__()
        self.timesteps = timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            betas = self._linear_beta_schedule(timesteps)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register tensors as buffers (state that is NOT updated by optimizer)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        
        # Precompute terms used in forward diffusion
        # √ᾱ_t
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        # √(1 - ᾱ_t)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _linear_beta_schedule(self, timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
        """
        Ho et al. (2020) linear schedule.
        """
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Nichol & Dhariwal (2021) cosine schedule.
        Ensures signal degrades more smoothly across all timesteps.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        
        # f(t) = cos^2( (t/T + s) / (1 + s) * pi/2 )
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        # Normalize so f(0) = 1
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # β_t = 1 - (ᾱ_t / ᾱ_{t-1})
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clamp beta to max 0.999 to prevent numerical instability near t=T
        return torch.clamp(betas, min=0.0001, max=0.999)
