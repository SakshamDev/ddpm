"""
Forward diffusion process.

Implements the closed-form jump from clean image to noisy image at any timestep t.
References: Ho et al. (2020) Eq. 11
"""

import torch

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract values from a 1D tensor `a` at indices `t` and reshape them to
    broadcast correctly with a tensor of shape `x_shape`.
    
    Args:
        a: 1D tensor of schedule values (e.g., shape (T,))
        t: 1D tensor of timesteps (e.g., shape (B,))
        x_shape: Target shape for broadcasting (e.g., (B, C, H, W))
        
    Returns:
        Tensor of shape (B, 1, 1, 1) containing the extracted values.
    """
    b = t.shape[0]
    out = a.gather(-1, t)
    # Reshape to (B, 1, 1, 1) for broadcasting against (B, C, H, W)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise_schedule, noise: torch.Tensor = None) -> torch.Tensor:
    """
    Forward diffusion process.
    
    Jumps directly from clean image x_0 to noisy image x_t:
    x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
    
    Args:
        x_start: Clean images, shape (B, C, H, W)
        t: Timesteps, shape (B,)
        noise_schedule: Instance of ddpm.noise_schedule.NoiseSchedule
        noise: Optional pre-generated Gaussian noise ε. If None, generated automatically.
        
    Returns:
        Noisy images at timestep t, shape (B, C, H, W)
    """
    if noise is None:
        noise = torch.randn_like(x_start)
        
    # Extract √ᾱ_t and √(1 - ᾱ_t) and reshape to (B, 1, 1, 1)
    sqrt_alphas_cumprod_t = extract(noise_schedule.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
