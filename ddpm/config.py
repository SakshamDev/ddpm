"""
Configuration for DDPM training and sampling.

Uses a dataclass so every hyperparameter is:
  1. Typed and documented
  2. Serializable to/from JSON (for checkpoint reproducibility)
  3. Printable for experiment logging

Design decision: dataclass over argparse/yaml because it's simpler,
IDE-friendly, and sufficient for a single-model project.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class DDPMConfig:
    """All hyperparameters for the DDPM project.

    Grouped by concern. Values set here are the final target config;
    during development/testing, override at construction time
    (e.g., DDPMConfig(base_channels=32, timesteps=100) for toy runs).
    """

    # --- Model architecture ---
    image_size: int = 32                          # CIFAR-10 is 32×32
    image_channels: int = 3                       # RGB
    base_channels: int = 128                      # C in the U-Net
    channel_multipliers: list = field(            # multipliers per resolution level
        default_factory=lambda: [1, 2, 2, 2]      # → channels: [128, 256, 256, 256]
    )
    num_res_blocks: int = 2                       # ResBlocks per resolution level
    attention_resolutions: list = field(           # spatial sizes where we apply self-attention
        default_factory=lambda: [16]               # 16×16 only (256 tokens — manageable)
    )
    num_classes: int = 10                         # CIFAR-10 classes (for CFG)
    dropout: float = 0.1                          # dropout in ResBlocks

    # --- Diffusion process ---
    timesteps: int = 1000                         # T — number of diffusion steps
    schedule_type: str = "cosine"                 # "cosine" (Nichol & Dhariwal) or "linear"

    # --- Training ---
    batch_size: int = 128
    learning_rate: float = 2e-4                   # Adam lr (Ho et al.)
    total_steps: int = 200_000
    ema_decay: float = 0.9999                     # EMA decay rate
    label_dropout: float = 0.1                    # probability of dropping class label (for CFG)
    grad_clip: float = 1.0                        # gradient clipping norm
    use_amp: bool = False                         # automatic mixed precision (Milestone 5)

    # --- Checkpointing ---
    checkpoint_dir: str = "/content/drive/MyDrive/ddpm_checkpoints"  # Google Drive path
    checkpoint_every: int = 10_000                # save every N steps
    log_every: int = 100                          # print loss every N steps

    # --- Sampling ---
    sampler: str = "ddpm"                         # "ddpm" or "ddim"
    ddim_steps: int = 50                          # number of DDIM steps (if using DDIM)
    ddim_eta: float = 0.0                         # DDIM stochasticity (0 = deterministic)
    guidance_scale: float = 3.0                   # classifier-free guidance scale (w)

    # --- Data ---
    data_dir: str = "./data"                      # where to download CIFAR-10
    num_workers: int = 2                          # DataLoader workers

    # --- Reproducibility ---
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization (checkpoint storage)."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DDPMConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> str:
        """One-line summary for experiment logging."""
        return (
            f"ch={self.base_channels} mult={self.channel_multipliers} "
            f"T={self.timesteps} sched={self.schedule_type} "
            f"bs={self.batch_size} lr={self.learning_rate} "
            f"steps={self.total_steps} amp={self.use_amp}"
        )
