"""
Utility functions used across the DDPM project.

Contents:
  - Device detection (CPU / MPS / CUDA) — write once, works everywhere
  - Reproducibility seed setting
  - Git commit hash capture for experiment logging
"""

import os
import random
import subprocess

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the best available device.

    Priority: CUDA (Colab T4) > MPS (M2 MacBook) > CPU.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    Also enables deterministic cuDNN behavior (slightly slower but reproducible).

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode: reproducible but ~10% slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    """Get the current git commit hash (short form).

    Returns 'unknown' if not in a git repo or git is unavailable.
    Used for experiment logging — every run records its code version.

    Returns:
        str: Short git commit hash, or 'unknown'.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def is_working_tree_clean() -> bool:
    """Check if the git working tree has uncommitted changes.

    Prints a warning if the tree is dirty — experiments should ideally
    be run on committed code so the git hash is meaningful.

    Returns:
        bool: True if clean, False if dirty or not a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        clean = result.returncode == 0 and result.stdout.strip() == ""
        if not clean:
            print("⚠️  WARNING: Git working tree has uncommitted changes.")
            print("   Experiment git hash may not fully represent the code used.")
        return clean
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model.

    Used to verify the U-Net hits our ~35M target.

    Args:
        model: A PyTorch module.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
