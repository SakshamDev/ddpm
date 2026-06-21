"""
Phase C — Checkpoint Recovery Drill

This is the M0 capstone: validates that our checkpoint infrastructure works
end-to-end before we trust it with real training runs.

Drill steps:
  1. Train a tiny conv autoencoder for 200 steps
  2. Save checkpoint via ddpm/checkpoint.py
  3. Kill the "session" (fresh model + optimizer from scratch)
  4. Resume from checkpoint
  5. Train 200 more steps (steps 201–400)
  6. Verify all pass criteria

Pass criteria:
  ✓ Loss at step 201 is within ±5% of loss at step 200 (no discontinuity)
  ✓ Model weights match after save → load (torch.allclose)
  ✓ Optimizer state (momentum) restored (torch.allclose)
  ✓ Step counter resumes from 201, not 0
  ✓ latest.pt is maintained correctly
  ✓ No temp files left behind

Run with:
  source venv/bin/activate && python m0_exercises/c_checkpoint_drill.py
"""

import os
import shutil
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from ddpm.utils import set_seed


# ── Tiny autoencoder (mirrors a simplified U-Net: encoder → bottleneck → decoder) ──

class TinyAutoencoder(nn.Module):
    """3-layer conv autoencoder. ~11K params. Structurally similar to a U-Net
    (encoder → bottleneck → decoder) but tiny enough for instant CPU training."""

    def __init__(self):
        super().__init__()
        # Encoder: (3, 32, 32) → (16, 16, 16) → (32, 8, 8)
        self.enc1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)   # (3,32,32) → (16,16,16)
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # (16,16,16) → (32,8,8)
        # Decoder: (32, 8, 8) → (16, 16, 16) → (3, 32, 32)
        self.dec1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # (32,8,8) → (16,16,16)
        self.dec2 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)   # (16,16,16) → (3,32,32)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        h = F.silu(self.enc1(x))   # (B, 16, 16, 16)
        h = F.silu(self.enc2(h))   # (B, 32, 8, 8)
        h = F.silu(self.dec1(h))   # (B, 16, 16, 16)
        h = self.dec2(h)           # (B, 3, 32, 32) — no activation (reconstruction)
        return h


def make_synthetic_data(batch_size=8, num_batches=50):
    """Generate synthetic (3, 32, 32) data. Same seed = same data every time."""
    set_seed(42)
    batches = []
    for _ in range(num_batches):
        batches.append(torch.randn(batch_size, 3, 32, 32))
    return batches


def train_steps(model, optimizer, data, start_step, end_step):
    """Train for [start_step, end_step) and return per-step losses."""
    model.train()
    losses = []
    for step in range(start_step, end_step):
        batch = data[step % len(data)]
        recon = model(batch)
        loss = F.mse_loss(recon, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if step % 50 == 0 or step == end_step - 1:
            print(f"  Step {step:4d} | Loss: {loss.item():.6f}")
    return losses


def main():
    print("=" * 60)
    print("Phase C — Checkpoint Recovery Drill")
    print("=" * 60)

    # Temp directory for drill checkpoints (cleaned up at end)
    ckpt_dir = tempfile.mkdtemp(prefix="ddpm_drill_")
    all_passed = True

    try:
        # ── Phase 1: Train for 200 steps ──
        print("\n── Phase 1: Train for 200 steps ──")
        set_seed(42)
        model = TinyAutoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        data = make_synthetic_data()

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count:,}")

        losses_phase1 = train_steps(model, optimizer, data, 0, 200)
        loss_at_200 = losses_phase1[-1]

        # ── Phase 2: Save checkpoint at step 200 ──
        print("\n── Phase 2: Save checkpoint ──")
        saved_weights = {k: v.clone() for k, v in model.state_dict().items()}
        saved_opt_state = {
            k: {sk: sv.clone() if isinstance(sv, torch.Tensor) else sv
                for sk, sv in v.items()}
            for k, v in optimizer.state_dict()["state"].items()
        }

        save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            step=200,
            config={"lr": 1e-3, "drill": True},
            seed=42,
            loss_history=losses_phase1[-10:],  # last 10 losses
        )

        # ── Phase 3: Simulate disconnect (fresh everything) ──
        print("\n── Phase 3: Simulate disconnect ──")
        del model, optimizer  # destroy everything
        print("  ✓ Model and optimizer destroyed")

        # ── Phase 4: Resume from checkpoint ──
        print("\n── Phase 4: Resume from checkpoint ──")
        latest_path = find_latest_checkpoint(ckpt_dir)

        # Check 5: latest.pt exists and is found
        if latest_path is None:
            print("  ❌ FAIL: find_latest_checkpoint returned None")
            all_passed = False
        else:
            print(f"  ✓ Found checkpoint: {os.path.basename(latest_path)}")

        # Create FRESH model and optimizer (different random init)
        fresh_model = TinyAutoencoder()
        fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)

        ckpt = load_checkpoint(
            latest_path, fresh_model, fresh_optimizer, device=torch.device("cpu")
        )

        # Check 3: Weights match
        print("\n── Verification Checks ──")
        weights_match = all(
            torch.allclose(fresh_model.state_dict()[k], saved_weights[k])
            for k in saved_weights
        )
        if weights_match:
            print("  ✓ Check 1: Model weights restored exactly")
        else:
            print("  ❌ FAIL: Model weights don't match after reload")
            all_passed = False

        # Check 4: Step counter
        resumed_step = ckpt["step"]
        if resumed_step == 200:
            print(f"  ✓ Check 2: Step counter = {resumed_step} (correct)")
        else:
            print(f"  ❌ FAIL: Step counter = {resumed_step}, expected 200")
            all_passed = False

        # Check: Optimizer state
        loaded_opt_state = fresh_optimizer.state_dict()["state"]
        opt_match = True
        for param_key in saved_opt_state:
            for state_key, original_val in saved_opt_state[param_key].items():
                if isinstance(original_val, torch.Tensor):
                    loaded_val = loaded_opt_state[param_key][state_key]
                    if not torch.allclose(original_val, loaded_val):
                        opt_match = False
                        break
        if opt_match:
            print("  ✓ Check 3: Optimizer state (momentum buffers) restored")
        else:
            print("  ❌ FAIL: Optimizer state doesn't match")
            all_passed = False

        # Check: Loss history preserved
        if ckpt["loss_history"] == losses_phase1[-10:]:
            print("  ✓ Check 4: Loss history preserved in checkpoint")
        else:
            print("  ❌ FAIL: Loss history mismatch")
            all_passed = False

        # Check: No temp files
        tmp_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".tmp")]
        if len(tmp_files) == 0:
            print("  ✓ Check 5: No temp files left behind")
        else:
            print(f"  ❌ FAIL: {len(tmp_files)} temp files found")
            all_passed = False

        # ── Phase 5: Train 200 more steps ──
        print("\n── Phase 5: Resume training (steps 201–400) ──")
        data = make_synthetic_data()  # same data (same seed)
        losses_phase2 = train_steps(
            fresh_model, fresh_optimizer, data,
            start_step=resumed_step, end_step=resumed_step + 200
        )
        loss_at_201 = losses_phase2[0]

        # Check 1: Loss continuity
        pct_change = abs(loss_at_201 - loss_at_200) / loss_at_200
        if pct_change < 0.05:
            print(f"\n  ✓ Check 6: Loss continuity — step 200: {loss_at_200:.6f}, "
                  f"step 201: {loss_at_201:.6f} (Δ = {pct_change:.2%})")
        else:
            print(f"\n  ❌ FAIL: Loss discontinuity — step 200: {loss_at_200:.6f}, "
                  f"step 201: {loss_at_201:.6f} (Δ = {pct_change:.2%}, threshold 5%)")
            all_passed = False

        # ── Final verdict ──
        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 CHECKPOINT RECOVERY DRILL: ALL 6 CHECKS PASSED")
            print("   Checkpoint infrastructure is validated.")
            print("   Ready to proceed to Milestone 1.")
        else:
            print("❌ CHECKPOINT RECOVERY DRILL: SOME CHECKS FAILED")
            print("   Fix the issues above before proceeding.")
        print("=" * 60)

    finally:
        # Clean up temp checkpoint dir
        shutil.rmtree(ckpt_dir)
        print(f"\n🧹 Cleaned up temp dir: {ckpt_dir}")


if __name__ == "__main__":
    main()
