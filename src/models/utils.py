import numpy as np
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, t_emb_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(t_emb_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)