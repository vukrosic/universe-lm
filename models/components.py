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


class ReLU2FeedForward(nn.Module):
    """153 — Squared-ReLU FFN activation, Primer / Mercury Coder style.

    Identical param count and shape to `SquaredReLUFeedForward` (two
    projections, no gate) — the lever is purely the activation
    formulation. `relu2(x) = x * F.relu(x)` is mathematically equal to
    `(max(0, x))^2` for any real x; we use the `x * relu(x)` form so
    the forward graph is visibly distinct from
    `SquaredReLUFeedForward`'s `torch.square(F.relu(...))` (helpful for
    grep and for confirming the branch was actually taken at run time).
    At init with normal-distributed pre-activations both
    formulations produce zero-mean, similar-variance outputs.
    See `autoresearch/ideas/153-relu2-ffn/idea.md`.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.up_proj(x)
        return self.down_proj(self.dropout(h * F.relu(h)))


class SaturatingReLUFeedForward(nn.Module):
    """#93 Anti-outlier FFN. squared_relu AMPLIFIES large activations (x^2),
    manufacturing the massive-activation channels that hurt L2 normalization.
    This replaces the square with a smooth soft-cap: c * tanh(relu(x) / c).
    Linear for small activations (preserves signal), saturating at +c for
    large ones (compresses outliers at their source). c is a learnable scalar
    (init 4). Same 2-projection shape/param-count as squared_relu."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.cap = nn.Parameter(torch.tensor(4.0))

    def forward(self, x):
        h = F.relu(self.up_proj(x))
        c = self.cap.abs() + 1e-4
        h = c * torch.tanh(h / c)
        return self.down_proj(self.dropout(h))


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


class GELUFeedForward(nn.Module):
    """Standard GELU FeedForward layer.

    #60 — fresh activation axis. Plain GELU on a single up-projection,
    # no gating, no squaring. Different operating point from squared_relu
    # (Primer-style) and swiglu (Llama-style). Tests whether the FFN
    # activation is itself a real architecture lever — a question we
    # haven't cleanly answered yet because SwiGLU and squared_relu
    # differ in BOTH activation and number of projections.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.gelu(self.up_proj(x))))
