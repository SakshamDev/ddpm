"""
Sampling utilities for DDPM.
Implements ancestral sampling, DDIM, and Classifier-Free Guidance (CFG).
"""

import torch
from ddpm.forward_diffusion import extract

@torch.no_grad()
def p_sample_ddpm(model, x, t, t_index, schedule, classes=None, cfg_scale=1.0):
    """
    Standard DDPM sampling step (Ancestral Sampling).
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta) + sigma_t * z
    """
    B = x.shape[0]
    
    # Extract terms for the current timestep
    betas_t = extract(schedule.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(1.0 / torch.sqrt(schedule.alphas), t, x.shape)
    
    # Model prediction (with CFG if enabled)
    if cfg_scale > 1.0 and classes is not None:
        # For CFG, we do two forward passes: one conditional, one unconditional
        # We can batch them together for efficiency
        x_in = torch.cat([x, x])
        t_in = torch.cat([t, t])
        
        # null class is assumed to be 10 for CIFAR-10
        null_classes = torch.full_like(classes, 10)
        c_in = torch.cat([classes, null_classes])
        
        pred_noise_both = model(x_in, t_in, classes=c_in)
        pred_noise_cond, pred_noise_uncond = pred_noise_both.chunk(2)
        
        # Extrapolate
        pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
    else:
        pred_noise = model(x, t, classes=classes)
        
    # Compute mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
    
    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        # using fixed variance beta_t
        variance = betas_t
        return model_mean + torch.sqrt(variance) * noise


@torch.no_grad()
def p_sample_ddim(model, x, t, t_next, schedule, classes=None, cfg_scale=1.0, eta=0.0):
    """
    DDIM sampling step. Allows skipping timesteps.
    """
    # Extract terms
    alpha_bar_t = extract(schedule.alphas_cumprod, t, x.shape)
    alpha_bar_next = extract(schedule.alphas_cumprod, t_next, x.shape) if t_next[0] >= 0 else torch.ones_like(alpha_bar_t)
    
    if cfg_scale > 1.0 and classes is not None:
        x_in = torch.cat([x, x])
        t_in = torch.cat([t, t])
        null_classes = torch.full_like(classes, 10)
        c_in = torch.cat([classes, null_classes])
        
        pred_noise_both = model(x_in, t_in, classes=c_in)
        pred_noise_cond, pred_noise_uncond = pred_noise_both.chunk(2)
        pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
    else:
        pred_noise = model(x, t, classes=classes)
        
    # Predict x0
    pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
    
    # Direction pointing to x_t
    sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next))
    dir_xt = torch.sqrt(1 - alpha_bar_next - sigma**2) * pred_noise
    
    # Random noise
    noise = sigma * torch.randn_like(x)
    
    x_prev = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt + noise
    return x_prev


@torch.no_grad()
def generate_images(model, schedule, num_images=16, classes=None, cfg_scale=3.0, sampler="ddpm", steps=1000):
    """
    Generate images from pure noise.
    """
    model.eval()
    device = next(model.parameters()).device
    shape = (num_images, 3, 32, 32)
    x = torch.randn(shape, device=device)
    
    if sampler == "ddpm":
        # Ancestral sampling
        for i in reversed(range(0, schedule.timesteps)):
            t = torch.full((num_images,), i, device=device, dtype=torch.long)
            x = p_sample_ddpm(model, x, t, i, schedule, classes=classes, cfg_scale=cfg_scale)
            
    elif sampler == "ddim":
        # DDIM skipping steps
        times = torch.linspace(schedule.timesteps - 1, 0, steps, dtype=torch.long, device=device)
        times_next = torch.cat([times[1:], torch.tensor([-1], device=device)])
        
        for i in range(len(times)):
            t = torch.full((num_images,), times[i].item(), device=device, dtype=torch.long)
            t_next = torch.full((num_images,), times_next[i].item(), device=device, dtype=torch.long)
            x = p_sample_ddim(model, x, t, t_next, schedule, classes=classes, cfg_scale=cfg_scale)
            
    # Scale from [-1, 1] to [0, 1]
    x = (x + 1) * 0.5
    return x.clamp(0, 1)
