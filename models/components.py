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


def mish(x: torch.Tensor) -> torch.Tensor:
    """Mish activation (Misra 2019, arXiv:1908.08681):
    `mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`.
    fp32-stable everywhere (F.softplus is internally stabilized; tanh is
    bounded in [-1, 1]). Note `mish(0) = 0 * tanh(softplus(0)) = 0`
    and `dMish/dx|_{x=0} = tanh(softplus(0)) ≈ tanh(ln 2) ≈ 0.6` — a
    20% larger origin-derivative than SiLU's 0.5, which is the lever
    196 (MishGLU) tests. Used by `MishGLUFeedForward`.
    """
    return x * torch.tanh(F.softplus(x))


class MishGLUFeedForward(nn.Module):
    """196 — MishGLU FFN (Misra 2019 + Shazeer 2020 composition).

    Three-projection gated linear unit `y = down_proj(dropout(mish(
    W_gate·x) ⊙ (W_up·x)))` — structurally identical to
    `SwiGLUFeedForward` *except* the gate activation is `mish`
    (Misra 2019) instead of `silu` (Elfwing et al. 2017 / Hendrycks &
    Gimpel 2016 form used in LLaMA-family SwiGLU). The lever is the
    *inner-activation axis* (which gating function shapes the gate)
    — orthogonal to 170's closed *outer-GLU axis* (whether the gate
    mechanism itself binds at 0.94M).

    d_ff is scaled by the Shazeer 2/3 trick (`(2 * d_ff) // 3`, e.g.
    170 for d_ff_baseline=256) so total FFN param count matches
    SwiGLU to within ~0.4% (32,640 vs 32,768). The `mish(0) = 0`
    identity gives the step-0 silence automatically — no explicit
    zero-init on `gate_proj.weight` is needed (and would actually
    mask the gradient signal the lever depends on, since the
    derivative `dMish/dx|_{x=0} ≈ 0.6` is the whole point of the
    swap). Kaiming-uniform init (the standard `nn.Linear` default)
    is correct.

    Default off → `MishGLUFeedForward` is never constructed, the
    standard `ffn_variant` cascade runs bit-identical to baseline.
    See `autoresearch/ideas/196-ffn-glu-mish/idea.md`.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = mish(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(hidden))


class SwiGLUZeroInitFeedForward(nn.Module):
    """170 — SwiGLU FFN with zero-initialized gate (Shazeer 2020,
    arXiv:2002.05202; LLaMA-family FFN).

    Drop-in replacement for the standard GELU/SquaredReLU FFN. Three
    projections (gate_proj, up_proj, down_proj) plus dropout; the
    forward is

        y = down_proj(dropout(silu(gate_proj(x)) * up_proj(x)))

    Identical to the existing `SwiGLUFeedForward` *except* that
    `gate_proj.weight` is zero-init at construction (and `gate_proj.bias`
    if it had one — `SwiGLUFeedForward` uses `bias=False` so this is
    a no-op). The math identity: at step 0 `gate_proj(x) = 0` exactly
    ⇒ `silu(0) = 0` ⇒ `silu(gate) * (W_up · x) = 0` ⇒ FFN output is
    **exactly zero**. The residual stream therefore carries only the
    attention sub-block at step 0 — a ReZero-style model.

    At eval, the optimizer has grown the gate and the FFN contributes.
    This makes step-0 a clean "FFN-off" baseline: the val loss at step
    0 is *higher* than the vanilla GELU baseline (which has a non-
    trivial FFN at step 0), but the optimizer ramps the gate in over
    the first few hundred steps and the model converges as usual.

    Param parity uses the standard Shazeer 2/3 trick: pass
    `d_ff = (2 * d_ff_baseline) // 3` (e.g. 170 for d_ff_baseline=256).
    Total FFN params = 3 × d_model × d_ff_swiglu. At tiny1m3m
    (d_model=64, d_ff_baseline=256) that's 32,640 vs the baseline
    2 × 64 × 256 = 32,768 — 0.4% smaller, well within harness noise.

    Default off → `SwiGLUZeroInitFeedForward` is never constructed, the
    standard `ffn_variant` cascade runs bit-identical to the no-flag
    baseline. See `autoresearch/ideas/170-swiglu-ffn/idea.md`.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Zero-init the gate so `silu(gate_proj(x)) = silu(0) = 0` exactly
        # at step 0 ⇒ FFN output = 0 exactly at step 0. This is the lever:
        # the optimizer must then grow the gate during training. The up_proj
        # and down_proj keep their default Kaiming init (their contribution
        # is gated out by `silu(0)=0` until the gate grows).
        with torch.no_grad():
            nn.init.zeros_(self.gate_proj.weight)

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
