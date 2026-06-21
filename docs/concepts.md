# Concepts & Understanding Checkpoints

> **Purpose:** Glossary of every concept introduced in this project, plus a log of understanding checkpoints.  
> For each concept: definition, your restatement in your own words, date, and confidence level.  
> A concept is not "learned" until you've restated it and it's logged here.

---

## Format

### Concept Name

- **Definition:** One-paragraph explanation.
- **Your restatement:** *(filled in during understanding checkpoint)*
- **Date:** YYYY-MM-DD
- **Confidence:** `high` / `medium` / `low`

---

*(Concepts taught in Phase B sprint — 2026-06-21. Restatements pending.)*

### B1 — Tensor Creation, Indexing, Reshaping

- **Definition:** PyTorch tensors are multi-dimensional arrays with automatic differentiation support. Images are `(B, C, H, W)` — channels first. Key ops: `torch.randn` for noise, `tensor[indices]` for batch lookups into schedules, `.reshape(B, C, 1, 1)` to add spatial dims for broadcasting.
- **Your restatement:** *(fill in)*
- **Date:** 2026-06-21
- **Confidence:** —

### B2 — Broadcasting

- **Definition:** PyTorch's rule for multiplying tensors of different shapes: align from the right, size-1 dims expand, mismatched non-1 dims error. The critical DDPM pattern: reshape `(B,)` → `(B,1,1,1)` to multiply per-sample schedule values against `(B,C,H,W)` images.
- **Your restatement:** *(fill in)*
- **Date:** 2026-06-21
- **Confidence:** —

### B3 — Dataset & DataLoader

- **Definition:** `Dataset` defines how to load one sample (`__getitem__`). `DataLoader` wraps it with batching, shuffling, and parallel workers. CIFAR-10 images are normalized to `[-1, 1]` via `transforms.Normalize((0.5,), (0.5,))`. `drop_last=True` ensures consistent batch sizes.
- **Your restatement:** *(fill in)*
- **Date:** 2026-06-21
- **Confidence:** —

### B4 — nn.Module

- **Definition:** Base class for all neural network components. Register layers in `__init__`, define computation in `forward()`. Use `nn.ModuleList` (not Python lists) for collections. `model.parameters()` yields all trainable weights for the optimizer.
- **Your restatement:** *(fill in)*
- **Date:** 2026-06-21
- **Confidence:** —

### B5 — Autograd

- **Definition:** PyTorch's automatic differentiation engine. `loss.backward()` computes gradients for all parameters. `optimizer.step()` updates them. `optimizer.zero_grad()` clears accumulated gradients (forgetting this = exploding updates). `torch.no_grad()` disables tracking for inference/EMA. `.detach()` removes a tensor from the graph.
- **Your restatement:** *(fill in)*
- **Date:** 2026-06-21
- **Confidence:** —

### B6 — Training Loop + Checkpointing

- **Definition:** The canonical cycle: load batch → forward → loss → backward → clip gradients → optimizer step → zero grad → periodically save checkpoint. `model.train()`/`model.eval()` toggles dropout/norm behavior. `state_dict()` captures all parameters for save/load.
- **Your restatement:** *(fill in)*
- **Date:** 2026-06-21
- **Confidence:** —

### B7 — Device Management

- **Definition:** Tensors and models must be on the same device (CPU/CUDA/MPS). Use `model.to(device)` and `tensor.to(device)`. Use `torch.randn_like(x)` (inherits device) instead of `torch.randn(x.shape)` (defaults to CPU). Device-agnostic pattern: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
- **Your restatement:** *(fill in)*
- **Date:** 2026-06-21
- **Confidence:** —

### Milestone 1 — Noise Schedule & Forward Diffusion

- **Definition:** The forward process adds noise to an image over $T$ steps. The noise schedule dictates how much noise is added at each step, defining $\bar{\alpha}_t$ (the fraction of original signal remaining). The cosine schedule ensures a smoother, more uniform degradation compared to the linear schedule.
- **Your restatement:** cosine ensures it denoises uniformly
- **Date:** 2026-06-21
- **Confidence:** high
