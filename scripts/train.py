"""
Training script for DDPM on CIFAR-10.
Implements the training loop with Exponential Moving Average (EMA),
Mixed Precision Training (AMP), and Classifier-Free Guidance (CFG) label dropout.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler

from ddpm.unet import UNet
from ddpm.noise_schedule import NoiseSchedule
from ddpm.forward_diffusion import q_sample
from ddpm.ema import EMA
from ddpm.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


def train(args):
    # Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = args.use_amp
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False  # MPS amp support is experimental/limited
    else:
        device = torch.device("cpu")
        use_amp = False
    
    print(f"Using device: {device}")

    # Dataset & DataLoader (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download data
    os.makedirs("./data", exist_ok=True)
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)

    # Model Setup
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 2, 2),  # Typical for CIFAR-10 32x32 -> 16 -> 8 -> 4
        attention_resolutions=(1,),  # Apply attention at 16x16
        num_res_blocks=2,
        time_emb_dim=args.base_channels * 4,
        num_classes=10  # CIFAR-10 classes
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # EMA Setup
    ema = EMA(beta=0.9999)
    ema_model = ema.create_ema_model(model).to(device)

    # Noise Schedule
    schedule = NoiseSchedule(schedule_type="cosine", timesteps=args.timesteps).to(device)

    # AMP Scaler
    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    # Resume from checkpoint if provided
    start_step = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest_ckpt = find_latest_checkpoint(args.checkpoint_dir)
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt_data = load_checkpoint(latest_ckpt, model, optimizer, device, ema_model=ema_model, scaler=scaler)
        start_step = ckpt_data["step"]

    # Training Loop
    model.train()
    data_iter = iter(dataloader)
    
    # Loss history for checkpointing
    loss_history = []

    for step in range(start_step, args.total_steps):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, labels = next(data_iter)

        images = images.to(device)
        labels = labels.to(device)
        B = images.shape[0]

        # Classifier-Free Guidance: Randomly drop labels with p=0.1
        # We use num_classes (10) as the "null" class for unconditional generation
        if args.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=device) < args.cfg_drop_prob
            labels = torch.where(drop_mask, torch.tensor(10, device=device), labels)

        # 1. Sample random timesteps
        t = torch.randint(0, args.timesteps, (B,), device=device).long()
        
        # 2. Sample random noise
        noise = torch.randn_like(images)

        # 3. Forward diffusion: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise
        x_t = q_sample(images, t, schedule, noise=noise)

        # 4. Predict noise (Forward pass with AMP if enabled)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(device_type=device.type):
                predicted_noise = model(x_t, t, classes=labels)
                loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predicted_noise = model(x_t, t, classes=labels)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

        # Update EMA
        ema.update_model_average(ema_model, model)
        
        loss_history.append(loss.item())
        if len(loss_history) > 100:
            loss_history.pop(0)

        # Logging
        if step % args.log_every == 0:
            avg_loss = sum(loss_history) / len(loss_history)
            print(f"Step {step:6d}/{args.total_steps} | Loss: {avg_loss:.4f}")

        # Checkpointing
        if step > 0 and step % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                step=step,
                config=vars(args),
                seed=42,
                loss_history=loss_history,
                ema_model=ema_model,
                scaler=scaler
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--total_steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--use_amp", action="store_true", help="Enable Mixed Precision")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    
    args = parser.parse_args()
    train(args)
