"""
Checkpoint save/load/resume for DDPM training.

Design principles (from Operating Rule #9):
  1. Checkpoint discipline from session 1 — resumability is not an afterthought.
  2. Atomic writes — save to temp file, then os.rename() to prevent corruption
     if Colab disconnects mid-save.
  3. Graceful schema evolution — load_checkpoint uses .get() with defaults so
     checkpoints from earlier milestones still load in later code.
  4. latest.pt — always maintained as a copy of the most recent checkpoint.

Checkpoint contents:
  - model_state_dict      (always present)
  - optimizer_state_dict   (always present)
  - step                   (always present)
  - config                 (always present — frozen hyperparams for this run)
  - seed                   (always present)
  - loss_history           (always present — last N losses for continuity check)
  - ema_state_dict         (optional — added in Milestone 4)
  - scaler_state_dict      (optional — added in Milestone 5, AMP)
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    seed: int,
    loss_history: list,
    ema_model: Optional[torch.nn.Module] = None,
    scaler=None,
) -> str:
    """Save a training checkpoint with atomic write protection.

    Writes to a temporary file first, then renames — so if Colab crashes
    mid-save, we never end up with a corrupt checkpoint file.

    Also maintains 'latest.pt' as a copy of the most recent checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoints (Google Drive path).
        model: The model to save.
        optimizer: The optimizer to save (includes momentum buffers, etc.).
        step: Current training step (1-indexed).
        config: Config dict (from DDPMConfig.to_dict()).
        seed: Random seed used for this run.
        loss_history: List of recent loss values (for continuity verification on resume).
        ema_model: Optional EMA model (added in Milestone 4).
        scaler: Optional GradScaler (added in Milestone 5 for AMP).

    Returns:
        str: Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build the checkpoint dict
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config,
        "seed": seed,
        "loss_history": loss_history,
    }

    # Optional fields — only included if provided
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # Target filename
    filename = f"step_{step:07d}.pt"
    target_path = os.path.join(checkpoint_dir, filename)

    # --- Atomic write: save to temp file, then rename ---
    # We write the temp file in the SAME directory as the target so that
    # os.rename() is an atomic operation (same filesystem).
    fd, tmp_path = tempfile.mkstemp(dir=checkpoint_dir, suffix=".pt.tmp")
    try:
        os.close(fd)  # close the file descriptor; torch.save opens its own
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, target_path)  # atomic on same filesystem
    except Exception:
        # Clean up temp file if something went wrong
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    # --- Maintain latest.pt ---
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    shutil.copy2(target_path, latest_path)

    print(f"💾 Checkpoint saved: {target_path} (step {step})")
    return target_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
    ema_model: Optional[torch.nn.Module] = None,
    scaler=None,
) -> dict:
    """Load a checkpoint and restore model/optimizer state.

    Handles missing optional keys gracefully — checkpoints from earlier
    milestones (without EMA, without AMP scaler) will still load fine
    in later code.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: Model to load weights into (must have matching architecture).
        optimizer: Optional optimizer to restore state into.
        device: Device to map tensors to.
        ema_model: Optional EMA model to restore.
        scaler: Optional GradScaler to restore.

    Returns:
        dict: The full checkpoint dict, so callers can access 'step',
              'config', 'loss_history', etc.
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # --- Required fields ---
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # --- Optional fields (graceful degradation) ---
    if ema_model is not None and "ema_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
    elif ema_model is not None:
        print("⚠️  No EMA state in checkpoint — EMA model not restored.")

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    elif scaler is not None:
        print("⚠️  No scaler state in checkpoint — GradScaler not restored.")

    step = checkpoint.get("step", 0)
    print(f"✅ Resumed from step {step}")
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in a directory.

    Looks for 'latest.pt' first (fast path). If not found, scans for
    step_*.pt files and returns the one with the highest step number.

    Args:
        checkpoint_dir: Directory to search.

    Returns:
        str or None: Path to the latest checkpoint, or None if no
                     checkpoints exist.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    # Fast path: latest.pt exists
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path):
        return latest_path

    # Fallback: scan for step_*.pt files
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith("step_") and f.endswith(".pt")]
    )

    if not checkpoint_files:
        return None

    return os.path.join(checkpoint_dir, checkpoint_files[-1])
