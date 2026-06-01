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
    """SwiGLU feed-forward layer for a gated MLP variant."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Match the squared-ReLU FFN budget approximately:
        # 2 projections in + 1 projection out versus 1 + 1.
        inner_dim = max(1, int(round(d_ff * 2 / 3)))
        self.gate_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.up_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))
