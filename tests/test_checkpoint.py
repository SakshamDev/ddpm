"""
Unit tests for ddpm/checkpoint.py.

Tests the three public functions:
  - save_checkpoint: atomic write, correct contents, latest.pt maintenance
  - load_checkpoint: round-trip fidelity, graceful handling of missing optional keys
  - find_latest_checkpoint: correct selection of highest-step file, latest.pt fast path

These tests run on CPU with tiny models — no GPU needed.
"""

import os
import shutil
import tempfile

import pytest
import torch
import torch.nn as nn

from ddpm.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Minimal model for checkpoint testing. Two parameters: weight + bias."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def checkpoint_dir():
    """Create a temporary directory for checkpoints, cleaned up after the test."""
    tmp_dir = tempfile.mkdtemp(prefix="ddpm_test_ckpt_")
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture
def model_and_optimizer():
    """Create a fresh model and optimizer for each test."""
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


# ---------------------------------------------------------------------------
# Tests: save_checkpoint
# ---------------------------------------------------------------------------

class TestSaveCheckpoint:

    def test_creates_file(self, checkpoint_dir, model_and_optimizer):
        """save_checkpoint should create a step_*.pt file."""
        model, optimizer = model_and_optimizer
        path = save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=100, config={"test": True}, seed=42, loss_history=[0.5, 0.4],
        )
        assert os.path.exists(path)
        assert "step_0000100.pt" in path

    def test_creates_latest(self, checkpoint_dir, model_and_optimizer):
        """save_checkpoint should also create/update latest.pt."""
        model, optimizer = model_and_optimizer
        save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=100, config={}, seed=42, loss_history=[],
        )
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        assert os.path.exists(latest_path)

    def test_latest_updates_on_newer_save(self, checkpoint_dir, model_and_optimizer):
        """latest.pt should always reflect the most recent checkpoint."""
        model, optimizer = model_and_optimizer

        save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=100, config={}, seed=42, loss_history=[0.5],
        )
        save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=200, config={}, seed=42, loss_history=[0.3],
        )

        latest = torch.load(
            os.path.join(checkpoint_dir, "latest.pt"),
            map_location="cpu",
            weights_only=False,
        )
        assert latest["step"] == 200

    def test_checkpoint_contents(self, checkpoint_dir, model_and_optimizer):
        """Checkpoint should contain all required fields."""
        model, optimizer = model_and_optimizer
        path = save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=50, config={"lr": 2e-4}, seed=42, loss_history=[0.6, 0.5],
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert ckpt["step"] == 50
        assert ckpt["config"] == {"lr": 2e-4}
        assert ckpt["seed"] == 42
        assert ckpt["loss_history"] == [0.6, 0.5]

    def test_no_temp_files_left(self, checkpoint_dir, model_and_optimizer):
        """After a successful save, no .tmp files should remain."""
        model, optimizer = model_and_optimizer
        save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=100, config={}, seed=42, loss_history=[],
        )
        tmp_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0


# ---------------------------------------------------------------------------
# Tests: load_checkpoint
# ---------------------------------------------------------------------------

class TestLoadCheckpoint:

    def test_round_trip_weights(self, checkpoint_dir, model_and_optimizer):
        """Model weights should be identical after save → load."""
        model, optimizer = model_and_optimizer

        # Capture original weights
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}

        path = save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=100, config={}, seed=42, loss_history=[],
        )

        # Create a fresh model with different weights
        fresh_model = TinyModel()
        assert not all(
            torch.equal(fresh_model.state_dict()[k], original_weights[k])
            for k in original_weights
        ), "Fresh model should have different random weights"

        # Load checkpoint into fresh model
        load_checkpoint(path, fresh_model)

        # Verify weights match
        for key in original_weights:
            assert torch.equal(
                fresh_model.state_dict()[key], original_weights[key]
            ), f"Weight mismatch for {key}"

    def test_round_trip_optimizer_state(self, checkpoint_dir):
        """Optimizer state (momentum buffers) should survive round-trip."""
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Do a few training steps to populate optimizer state (momentum buffers)
        for _ in range(5):
            x = torch.randn(2, 4)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save optimizer state for comparison
        original_opt_state = {
            k: {sk: sv.clone() if isinstance(sv, torch.Tensor) else sv
                 for sk, sv in v.items()}
            for k, v in optimizer.state_dict()["state"].items()
        }

        path = save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=5, config={}, seed=42, loss_history=[],
        )

        # Fresh model + optimizer
        fresh_model = TinyModel()
        fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)

        load_checkpoint(path, fresh_model, fresh_optimizer)

        # Verify optimizer state matches
        loaded_opt_state = fresh_optimizer.state_dict()["state"]
        for param_key in original_opt_state:
            for state_key, original_val in original_opt_state[param_key].items():
                if isinstance(original_val, torch.Tensor):
                    assert torch.allclose(
                        loaded_opt_state[param_key][state_key], original_val
                    ), f"Optimizer state mismatch: param {param_key}, state {state_key}"

    def test_step_restored(self, checkpoint_dir, model_and_optimizer):
        """The step counter should be restored correctly."""
        model, optimizer = model_and_optimizer
        path = save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=12345, config={}, seed=42, loss_history=[],
        )
        ckpt = load_checkpoint(path, TinyModel())
        assert ckpt["step"] == 12345

    def test_graceful_missing_ema(self, checkpoint_dir, model_and_optimizer):
        """Loading a checkpoint without EMA state should not crash."""
        model, optimizer = model_and_optimizer
        path = save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=100, config={}, seed=42, loss_history=[],
            # No ema_model passed → no ema_state_dict in checkpoint
        )

        fresh_model = TinyModel()
        ema_model = TinyModel()  # Caller passes EMA model, but checkpoint lacks it

        # This should print a warning but NOT raise
        ckpt = load_checkpoint(path, fresh_model, ema_model=ema_model)
        assert "ema_state_dict" not in ckpt

    def test_loss_history_preserved(self, checkpoint_dir, model_and_optimizer):
        """Loss history should survive the round-trip for continuity checks."""
        model, optimizer = model_and_optimizer
        losses = [0.8, 0.6, 0.5, 0.45, 0.42]
        path = save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=500, config={}, seed=42, loss_history=losses,
        )
        ckpt = load_checkpoint(path, TinyModel())
        assert ckpt["loss_history"] == losses


# ---------------------------------------------------------------------------
# Tests: find_latest_checkpoint
# ---------------------------------------------------------------------------

class TestFindLatestCheckpoint:

    def test_returns_none_for_empty_dir(self, checkpoint_dir):
        """Should return None if no checkpoints exist."""
        assert find_latest_checkpoint(checkpoint_dir) is None

    def test_returns_none_for_nonexistent_dir(self):
        """Should return None if the directory doesn't exist."""
        assert find_latest_checkpoint("/nonexistent/path/abc123") is None

    def test_finds_latest_pt(self, checkpoint_dir, model_and_optimizer):
        """Should return latest.pt when it exists."""
        model, optimizer = model_and_optimizer
        save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=100, config={}, seed=42, loss_history=[],
        )

        result = find_latest_checkpoint(checkpoint_dir)
        assert result is not None
        assert result.endswith("latest.pt")

    def test_finds_highest_step_without_latest(self, checkpoint_dir, model_and_optimizer):
        """If latest.pt is missing, should find the highest step_*.pt file."""
        model, optimizer = model_and_optimizer

        save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=100, config={}, seed=42, loss_history=[],
        )
        save_checkpoint(
            checkpoint_dir, model, optimizer,
            step=200, config={}, seed=42, loss_history=[],
        )

        # Remove latest.pt to force the fallback path
        os.remove(os.path.join(checkpoint_dir, "latest.pt"))

        result = find_latest_checkpoint(checkpoint_dir)
        assert result is not None
        assert "step_0000200.pt" in result
