"""
U-Net architecture for DDPM.

This is the core neural network that predicts the noise ε given x_t and t.
It features:
- Sinusoidal timestep embeddings
- ResNet blocks with GroupNorm and SiLU
- Self-attention blocks at lower resolutions
- U-Net down/up architecture with skip connections

This implementation is parameterizable to easily instantiate "tiny" versions
for fast CPU debugging (e.g., base_channels=32, 1-2 downsamples) or full
versions for CIFAR-10 training (base_channels=128, 4 downsamples).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes the scalar timestep t into a high-dimensional vector.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """
    A standard block: Conv2d -> GroupNorm -> SiLU
    """
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResnetBlock(nn.Module):
    """
    Combines two Blocks with a residual connection.
    Injects the timestep embedding via addition after the first block.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = Block(in_channels, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)
        
        # 1x1 conv if input and output channels differ, else Identity
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        
        # Inject time embedding (reshape to match spatial dims)
        time_emb = self.mlp(time_emb)
        h = h + time_emb[..., None, None]
        
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """
    Standard scaled dot-product self-attention applied spatially.
    """
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, channels)
        self.proj_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Normalize and compute Q, K, V
        qkv = self.proj_qkv(self.group_norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention: (B, heads, HW, C/heads)
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W).transpose(-1, -2)
        k = k.view(B, self.num_heads, head_dim, H * W)  # Not transposed, ready for matmul
        v = v.view(B, self.num_heads, head_dim, H * W).transpose(-1, -2)
        
        # Attention scores: Q * K^T / sqrt(d)
        attn = torch.matmul(q, k) * (head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply to V
        out = torch.matmul(attn, v)
        
        # Reshape back to image dimensions
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        
        # Final projection and residual connection
        return x + self.proj_out(out)


class Downsample(nn.Module):
    """Halves spatial dimensions (e.g., 32x32 -> 16x16)"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Doubles spatial dimensions (e.g., 16x16 -> 32x32)"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Full UNet Architecture for DDPM.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        attention_resolutions: tuple = (2,),  # Indices in channel_mults where attention applies
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        
        final_out_channels = out_channels
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        # Track channel counts for skip connections
        channels = [base_channels]
        now_channels = base_channels
        
        # ----- Downward Path -----
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResnetBlock(now_channels, out_channels, time_emb_dim)]
                now_channels = out_channels
                if i in attention_resolutions:
                    layers.append(AttentionBlock(now_channels))
                self.downs.append(nn.ModuleList(layers))
                channels.append(now_channels)
                
            if i != len(channel_mults) - 1:
                self.downs.append(nn.ModuleList([Downsample(now_channels)]))
                channels.append(now_channels)
                
        # ----- Middle Blocks -----
        self.mid_block1 = ResnetBlock(now_channels, now_channels, time_emb_dim)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResnetBlock(now_channels, now_channels, time_emb_dim)

        # ----- Upward Path -----
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            # +1 because we also process the downsample/init_conv skip connection
            for _ in range(num_res_blocks + 1):
                skip_channels = channels.pop()
                layers = [ResnetBlock(now_channels + skip_channels, out_channels, time_emb_dim)]
                now_channels = out_channels
                if i in attention_resolutions:
                    layers.append(AttentionBlock(now_channels))
                self.ups.append(nn.ModuleList(layers))
                
            if i != 0:
                self.ups.append(nn.ModuleList([Upsample(now_channels)]))
                
        assert len(channels) == 0, "Channel tracking logic mismatch"
                
        # ----- Final Output -----
        self.final_res_block = ResnetBlock(now_channels, now_channels, time_emb_dim)
        self.final_conv = nn.Conv2d(now_channels, final_out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy images, shape (B, C, H, W)
            time: Timesteps, shape (B,)
            
        Returns:
            Predicted noise, shape (B, C, H, W)
        """
        t = self.time_mlp(time)
        x = self.init_conv(x)
        
        skips = [x]
        
        # Down
        for block in self.downs:
            for layer in block:
                if isinstance(layer, ResnetBlock):
                    x = layer(x, t)
                else:
                    x = layer(x)
            skips.append(x)
            
        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # Up
        for block in self.ups:
            if isinstance(block[0], ResnetBlock):
                skip = skips.pop()
                x = torch.cat((x, skip), dim=1)
                
            for layer in block:
                if isinstance(layer, ResnetBlock):
                    x = layer(x, t)
                else:
                    x = layer(x)
                    
        # Final
        x = self.final_res_block(x, t)
        return self.final_conv(x)
