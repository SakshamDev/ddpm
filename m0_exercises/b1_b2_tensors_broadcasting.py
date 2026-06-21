"""
Phase B — Exercise B1: Tensors (Creation, Indexing, Reshaping)
Phase B — Exercise B2: Broadcasting

Run with:  source venv/bin/activate && python m0_exercises/b1_b2_tensors_broadcasting.py

Each exercise has:
  - A task description in the docstring
  - Skeleton code with TODOs for you to fill in
  - assert statements that verify your answer

If all asserts pass, you'll see "✅ All exercises passed!"
If one fails, you'll see which exercise and what went wrong.
"""

import torch

print("=" * 60)
print("B1 — Tensors: Creation, Indexing, Reshaping")
print("=" * 60)


# ──────────────────────────────────────────────────────────────
# Exercise 1: Create a batch of "images" and extract a channel
# ──────────────────────────────────────────────────────────────
print("\n📝 Exercise 1: Create a batch of images")

# Task: Create a random tensor representing a batch of 4 RGB images,
#        each 32×32 pixels. Use torch.randn.
# Expected shape: (4, 3, 32, 32)  →  (batch, channels, height, width)
#
# Then extract the RED channel (channel index 0) of the SECOND image
# (batch index 1). The result should have shape (32, 32).

# TODO: Create the image batch
images = ...  # YOUR CODE HERE — shape should be (4, 3, 32, 32)

# TODO: Extract red channel of second image
red_channel = ...  # YOUR CODE HERE — shape should be (32, 32)

assert images.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {images.shape}"
assert red_channel.shape == (32, 32), f"Expected (32, 32), got {red_channel.shape}"
print("✅ Exercise 1 passed!")


# ──────────────────────────────────────────────────────────────
# Exercise 2: Index into a schedule with a batch of timesteps
# ──────────────────────────────────────────────────────────────
print("\n📝 Exercise 2: Schedule indexing")

# Task: Create a 1D tensor of 1000 linearly-spaced values from 0.0001 to 0.02.
#        This simulates a noise schedule β₁, β₂, ..., β_T.
#
#        Then create a batch of 4 random timesteps (integers 0–999).
#        Use these timesteps to look up the corresponding schedule values.
#
# Hint: torch.linspace for the schedule, torch.randint for timesteps.
# The result should have shape (4,) — one schedule value per sample.

# TODO: Create the schedule
schedule = ...  # YOUR CODE HERE — shape (1000,)

# TODO: Create random timesteps
timesteps = ...  # YOUR CODE HERE — shape (4,), integers in [0, 999]

# TODO: Look up schedule values for these timesteps
values = ...  # YOUR CODE HERE — shape (4,)

assert schedule.shape == (1000,), f"Expected (1000,), got {schedule.shape}"
assert timesteps.shape == (4,), f"Expected (4,), got {timesteps.shape}"
assert timesteps.dtype in (torch.int32, torch.int64), f"Timesteps should be integers, got {timesteps.dtype}"
assert values.shape == (4,), f"Expected (4,), got {values.shape}"
# Verify the lookup is correct: each value should equal schedule[t]
for i in range(4):
    assert values[i] == schedule[timesteps[i]], f"values[{i}] doesn't match schedule[timesteps[{i}]]"
print("✅ Exercise 2 passed!")


# ──────────────────────────────────────────────────────────────
# Exercise 3: Reshape for timestep embedding injection
# ──────────────────────────────────────────────────────────────
print("\n📝 Exercise 3: Reshape for injection")

# Task: You have a timestep embedding of shape (4, 256) — one 256-dim
#        vector per sample in a batch of 4.
#
#        Reshape it to (4, 256, 1, 1) so it can be added to a feature
#        map of shape (4, 256, 16, 16) via broadcasting.
#
# This is the EXACT pattern used in the U-Net's ResBlock to inject
# timestep information into convolutional feature maps.

embedding = torch.randn(4, 256)  # (batch=4, embed_dim=256)

# TODO: Reshape to (4, 256, 1, 1)
embedding_4d = ...  # YOUR CODE HERE

assert embedding_4d.shape == (4, 256, 1, 1), f"Expected (4, 256, 1, 1), got {embedding_4d.shape}"
# Verify data is unchanged (just reshaped, not copied or modified)
assert torch.equal(embedding_4d.squeeze(), embedding), "Data should be identical, just reshaped"
print("✅ Exercise 3 passed!")


print("\n" + "=" * 60)
print("B2 — Broadcasting")
print("=" * 60)


# ──────────────────────────────────────────────────────────────
# Exercise 4: Broadcasting failure and fix
# ──────────────────────────────────────────────────────────────
print("\n📝 Exercise 4: Broadcasting — fail then fix")

# Task: You have schedule values of shape (4,) and images of shape
#        (4, 3, 32, 32). Multiplying them directly will FAIL.
#
# Step 1: Try the multiplication and catch the error.
# Step 2: Fix it by reshaping the schedule values to (4, 1, 1, 1).

scale = torch.rand(4)            # (4,) — one value per sample
images = torch.randn(4, 3, 32, 32)  # (4, 3, 32, 32)

# Step 1: Verify that direct multiplication fails
failed = False
try:
    bad_result = scale * images  # This SHOULD fail
except RuntimeError:
    failed = True
assert failed, "Direct multiplication should have raised RuntimeError!"
print("  ✓ Confirmed: (4,) * (4,3,32,32) fails as expected")

# Step 2: Fix it
# TODO: Reshape scale so the multiplication works
scale_4d = ...  # YOUR CODE HERE — shape should be (4, 1, 1, 1)

result = scale_4d * images

assert scale_4d.shape == (4, 1, 1, 1), f"Expected (4, 1, 1, 1), got {scale_4d.shape}"
assert result.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {result.shape}"
# Verify: the first pixel of the first image should be scaled by scale[0]
assert torch.allclose(result[0, 0, 0, 0], scale[0] * images[0, 0, 0, 0]), "Scaling incorrect"
print("✅ Exercise 4 passed!")


# ──────────────────────────────────────────────────────────────
# Exercise 5: Implement toy q_sample (forward diffusion)
# ──────────────────────────────────────────────────────────────
print("\n📝 Exercise 5: Toy q_sample")

# Task: Implement the forward diffusion sampling equation:
#
#   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
#
# Where:
#   x_0:           clean images,    shape (B, C, H, W)  = (4, 3, 32, 32)
#   alpha_bar_t:   schedule values, shape (B,)           = (4,)
#   epsilon:       random noise,    shape (B, C, H, W)  = (4, 3, 32, 32)
#
# You need to:
#   1. Compute sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t)
#   2. Reshape both to (B, 1, 1, 1) for broadcasting
#   3. Apply the formula
#
# This is THE core equation of DDPM forward diffusion. You'll implement
# the real version in Milestone 1.

B = 4
x_0 = torch.randn(B, 3, 32, 32)                # clean images
alpha_bar = torch.tensor([0.99, 0.75, 0.25, 0.01])  # schedule values (high→low noise)
epsilon = torch.randn(B, 3, 32, 32)             # noise

# TODO: Implement q_sample
# Step 1: compute sqrt(alpha_bar) and sqrt(1 - alpha_bar)
sqrt_alpha_bar = ...       # YOUR CODE HERE — shape (4,)
sqrt_one_minus = ...       # YOUR CODE HERE — shape (4,)

# Step 2: reshape both to (B, 1, 1, 1) for broadcasting
sqrt_alpha_bar_4d = ...    # YOUR CODE HERE — shape (4, 1, 1, 1)
sqrt_one_minus_4d = ...    # YOUR CODE HERE — shape (4, 1, 1, 1)

# Step 3: apply the forward diffusion formula
x_t = ...                  # YOUR CODE HERE — shape (4, 3, 32, 32)

assert x_t.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {x_t.shape}"

# Sanity check: when alpha_bar ≈ 1, x_t ≈ x_0 (almost no noise)
# Sample 0 has alpha_bar = 0.99, so x_t[0] should be very close to x_0[0]
diff_clean = (x_t[0] - x_0[0]).abs().mean().item()
assert diff_clean < 0.5, f"When alpha_bar≈1, x_t should ≈ x_0. Mean diff = {diff_clean:.4f}"

# Sanity check: when alpha_bar ≈ 0, x_t ≈ epsilon (almost pure noise)
# Sample 3 has alpha_bar = 0.01, so x_t[3] should be very close to epsilon[3]
diff_noisy = (x_t[3] - epsilon[3]).abs().mean().item()
assert diff_noisy < 0.5, f"When alpha_bar≈0, x_t should ≈ ε. Mean diff = {diff_noisy:.4f}"

print("✅ Exercise 5 passed!")
print(f"   alpha_bar=0.99 → mean|x_t - x_0| = {diff_clean:.4f}  (should be small)")
print(f"   alpha_bar=0.01 → mean|x_t - ε|   = {diff_noisy:.4f}  (should be small)")


# ──────────────────────────────────────────────────────────────
# All done!
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🎉 All B1 + B2 exercises passed!")
print("=" * 60)
print("\nKey takeaways:")
print("  • Images in PyTorch are (B, C, H, W) — channels first")
print("  • torch.randn for noise, torch.linspace for schedules")
print("  • tensor[indices] for batch lookups into schedules")
print("  • Reshape (B,) → (B,1,1,1) before multiplying with (B,C,H,W)")
print("  • Broadcasting aligns from the right; dims of size 1 expand")
print("  • The q_sample formula is the heart of forward diffusion")
