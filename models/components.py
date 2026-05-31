import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SquaredReLUFeedForward(nn.Module):
    """Squared ReLU FeedForward layer (Primer-style)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Activation is (max(0, x))^2
        return self.down_proj(self.dropout(torch.square(F.relu(self.up_proj(x)))))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU FeedForward layer."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(hidden))


class LayerScale(nn.Module):
    """Per-dimension learnable residual branch scale."""

    def __init__(self, d_model: int, init_value: float):
        super().__init__()
        self.scale = nn.Parameter(torch.full((d_model,), float(init_value)))

    def forward(self, x):
        return x * self.scale
