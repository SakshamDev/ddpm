# Task Plan — DDPM From Scratch

> **Purpose:** Living plan tracking the current milestone, its status, and what's next.  
> Updated only at milestone boundaries — not mid-sprint.

---

## Current Milestone

**Milestone 0 — PyTorch Competency Assessment**  
**Status:** `NOT STARTED`

Evaluate fluency in: tensor ops, broadcasting, Dataset/DataLoader, nn.Module, autograd, training loops, GPU transfers.  
If proficient → skip to Milestone 1. Otherwise → design shortest possible PyTorch sprint.

---

## Roadmap

| # | Milestone | Status | Notes |
|---|-----------|--------|-------|
| 0 | PyTorch competency assessment | `NOT STARTED` | Gate for all subsequent work |
| 1 | Forward diffusion & math | `BLOCKED` | Awaiting M0 result |
| 2 | Tiny U-Net prototype (MNIST-scale) | `BLOCKED` | |
| 3 | Full U-Net at CIFAR-10 scale + class embeddings | `BLOCKED` | |
| 4 | Training loop + EMA + checkpointing | `BLOCKED` | |
| 5 | Mixed precision + T4 batch-size tuning | `BLOCKED` | |
| 6 | Full 200K-step CIFAR-10 training (w/ label dropout for CFG) | `BLOCKED` | |
| 7 | DDPM ancestral + DDIM samplers | `BLOCKED` | |
| 8 | Classifier-free guidance (sampling-time only) | `BLOCKED` | |
| 9 | FID evaluation + Gradio demo | `BLOCKED` | |
| 10 | Documentation + repo finalization | `BLOCKED` | |

---

## Next Action

Confirm with the user whether they need the PyTorch fluency sprint (M0) or can proceed directly to Milestone 1.
