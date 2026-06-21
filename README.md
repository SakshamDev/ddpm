# DDPM From Scratch — CIFAR-10

A from-scratch implementation of Denoising Diffusion Probabilistic Models (DDPM) on CIFAR-10, following:

- **Ho et al. (2020)** — DDPM core
- **Nichol & Dhariwal (2021)** — Cosine noise schedule
- **Song et al. (2021)** — DDIM sampling
- **Ho & Salimans (2021)** — Classifier-free guidance

## Architecture

- ~35M-parameter U-Net
- Base channels: 128, multipliers: [1, 2, 2, 2]
- Self-attention at 16×16 only, 2 ResBlocks per resolution
- T=1000 timesteps, cosine schedule, ε-prediction
- Classifier-free guidance via label dropout (single training run)

## Development Workflow

```
MacBook Air (M2) ──git push──▶ GitHub ──git clone──▶ Google Colab (T4)
   develop/test                  source of truth           train
```

### Local Development (MacBook Air)

```bash
# First-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/ -v

# Development cycle
git add . && git commit -m "description" && git push
```

### Colab Training

1. Open `notebooks/train_colab.ipynb`
2. Verify T4 GPU with `!nvidia-smi`
3. Run all cells — the notebook clones/pulls the repo, installs deps, and auto-resumes training

**Never edit code in Colab cells.** All logic lives in the `ddpm/` package.

### Checkpoints

Checkpoints save to Google Drive (`My Drive/ddpm_checkpoints/`) so they survive Colab disconnects. The training loop auto-resumes from the latest checkpoint.

### Experiment Logging

Every training run is logged in `experiments/exp_log.md` with:
- Run ID (`run_YYYYMMDD_HHMMSS`)
- Git commit hash (captured at training start)
- Config summary, steps completed, final loss, FID (if evaluated), notes

**Rule:** One change at a time between runs.

## Project Structure

```
ddpm/              Python package — ALL logic here
tests/             Unit tests — run locally before pushing
notebooks/         Colab thin-shell notebooks
scripts/           CLI entry points
docs/              Learning notes, paper notes, decisions
experiments/       Experiment log
m0_exercises/      PyTorch learning exercises (temporary)
```

## Roadmap

See `task_plan.md` for current milestone status.
