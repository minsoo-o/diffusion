import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        weight = torch.einsum("BTH,BSH->BTS", q, k)  # Inner product of Q and K, a tensor

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight = F.softmax(weight / math.sqrt(self.d_head), dim=-1)  # Softmax of scoremats
        output = torch.einsum("BTS,BSH->BTH", weight, v)  # Weighted average value vectors by attnmats

        output = self.out_proj(output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        weight = torch.einsum("BTH,BSH->BTS", q, k)  # Inner product of Q and K, a tensor
        weight = F.softmax(weight / math.sqrt(self.d_head), dim=-1)  # Softmax of scoremats
        output = torch.einsum("BTS,BSH->BTH", weight, v)  # Weighted average value vectors by attnmats

        output = self.out_proj(output)
        return output