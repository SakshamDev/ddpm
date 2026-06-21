# Decision Log

> **Purpose:** Record every non-default design decision with alternatives considered and rationale.  
> This is the "why" record — if someone asks "why did you do X instead of Y?", the answer is here.

---

## Format

### Decision Title

- **Date:** YYYY-MM-DD
- **Decision:** What we chose.
- **Alternatives considered:** What else we could have done.
- **Rationale:** Why we chose this option.
- **Reversible?** Yes / No / Partially

---

## D001 — Predict noise (ε) instead of clean image (x₀)

- **Date:** 2026-06-21 (design-time, not yet implemented)
- **Decision:** The U-Net will predict the noise ε added at each timestep, following Ho et al. (2020).
- **Alternatives:** Predict x₀ directly; predict v (velocity parameterization, Salimans & Ho 2022).
- **Rationale:** ε-prediction is the standard in Ho et al. Simplifies the loss to a direct MSE between predicted and actual noise. v-prediction is newer and useful for high-resolution / very high timestep counts, but adds complexity without clear benefit for CIFAR-10 at T=1000.
- **Reversible?** Yes — only changes the loss target and sampling formula.

## D002 — Cosine noise schedule instead of linear

- **Date:** 2026-06-21 (design-time)
- **Decision:** Use the cosine schedule from Nichol & Dhariwal (2021) instead of the linear schedule from Ho et al. (2020).
- **Alternatives:** Linear schedule (Ho et al.); sigmoid schedule.
- **Rationale:** The linear schedule destroys too much information too quickly at low-resolution (32×32). The cosine schedule provides a smoother degradation curve, which empirically improves sample quality on CIFAR-10.
- **Reversible?** Yes — schedule is a standalone module.

## D003 — Classifier-free guidance designed in from the start

- **Date:** 2026-06-21 (design-time)
- **Decision:** Build the class-embedding pathway and label-dropout mechanism into the architecture from Milestone 3, rather than retrofitting it later.
- **Alternatives:** Train unconditional first, then retrain with conditioning.
- **Rationale:** Avoids a second full training run. Label dropout (randomly replacing the class label with a null token during training) teaches the model both conditional and unconditional generation in a single run. This is the approach from Ho & Salimans (2021).
- **Reversible?** Partially — removing the class pathway later is possible but wasteful.

## D004 — Self-attention at 16×16 only

- **Date:** 2026-06-21 (design-time)
- **Decision:** Apply self-attention only at the 16×16 resolution level, not at 32×32, 8×8, or 4×4.
- **Alternatives:** Attention at all resolutions; attention at 16×16 and 8×8.
- **Rationale:** Self-attention is O(n²) in spatial tokens. At 32×32 (1024 tokens), it's expensive; at 16×16 (256 tokens) it's manageable. At 8×8 and 4×4 the spatial dimensions are too small for attention to add much. This matches common practice in DDPM implementations for CIFAR-10.
- **Reversible?** Yes — attention blocks are modular.
