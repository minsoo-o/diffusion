import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attentions import SelfAttention, CrossAttention


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.t_proj = nn.Linear(t_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x, t):
        residial = x

        x = self.norm1(x)
        x = self.conv1(x)
        x = F.gelu(x)

        t = self.t_proj(t)
        t = F.gelu(t)

        h = x + t.unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.conv2(h)
        return h + self.residual_layer(residial)


class AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, y_emb_dim=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.in_norm = nn.GroupNorm(1, channels, eps=1e-6)
        self.in_conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.norm1 = nn.LayerNorm(channels)
        self.attn1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.norm2 = nn.LayerNorm(channels)
        self.attn2 = CrossAttention(n_head, channels, y_emb_dim, in_proj_bias=False)
        self.norm3 = nn.LayerNorm(channels)
        self.fc1  = nn.Linear(channels, 4 * channels * 2)
        self.fc2 = nn.Linear(4 * channels, channels)

        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, y):
        residue_long = x

        x = self.in_norm(x)
        x = self.in_conv(x)
        
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))   # (n, c, hw)
        x = x.transpose(-1, -2)  # (n, hw, c)

        residue_short = x
        x = self.norm1(x)
        x = self.attn1(x)
        x += residue_short

        residue_short = x
        x = self.norm2(x)
        x = self.attn2(x, y)
        x += residue_short

        residue_short = x
        x = self.norm3(x)
        x, gate = self.fc1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.fc2(x)
        x += residue_short

        x = x.transpose(-1, -2)  # (n, c, hw)
        x = x.view((n, c, h, w))    # (n, c, h, w)

        return self.out_conv(x) + residue_long