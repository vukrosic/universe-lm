import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import SquaredReLUFeedForward, SwiGLUFeedForward, GELUFeedForward, SaturatingReLUFeedForward
from .fire_pe import FIREBias
from .cope import CoPE
from .deberta import DeBERTaRelativePositionBias
from .fox import FoX
from .canon_conv import CanonConv
from .soft_moe import SoftMoEFFN
from .mod_router import MoDRouter
from .short_conv import ShortConv1D
from .switch_ffn import SwitchFFN
from .expert_choice_moe import ExpertChoiceMoE


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(
            dim=dim, max_seq_len=max_seq_len, base=base
        )

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)


# ============================================================================
# 022 — Softpick (Zuhri/Fuadi/Aji 2025, arXiv:2504.20966)
# Rectified-softmax attention normalization:
#   softpick(x_i) = relu(exp(x_i) − 1) / (Σ_j |exp(x_j) − 1| + ε)
# Permits zero total attention mass → kills the attention-sink
# pathology without adding a learnable sink token. The `exp` is
# computed in **fp32** with a per-row max-subtraction stability
# trick (closed-form identity: numerator and denominator both scale
# by exp(M), so subtracting M = per-row max bounds exp(·) ≤ 1 and
# the function value is unchanged). Without this trick, scores >
# ~88 overflow fp32 → relu(inf)/inf = NaN; r2 evidence.md showed
# mid-training NaN at step 400 from exactly this. `mask` (bool,
# broadcast-compatible with `scores`) zeros out both numerator and
# denominator on masked positions — the bug class the spec calls
# out at `idea.md:32-45`. ε=1e-6 is the paper default, pinned.
# See `autoresearch/ideas/022-softpick-attention/plan.md`.
# ============================================================================
def softpick(scores: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Rectified-softmax attention normalization (max-stabilized).

    Numerical-stability identity:
        relu(exp(x) − 1) / Σ_j |exp(x_j) − 1|
      ≡ relu(exp(x − M) − exp(−M)) / Σ_j |exp(x_j − M) − exp(−M)|
    for any M (both numerator and denominator scale by exp(M); the
    relu sign is preserved because exp(x − M) > exp(−M) iff x > 0).
    We pick M = per-row max over UNMASKED positions, clamped to ≥ 0.
    Clamping at 0 keeps the identity exact when no positive scores
    exist (M_true ≤ 0 ⇒ subtracting 0 is a no-op and we recover the
    original `exp(x) − 1`); when some positive scores exist, the
    clamp is inactive and we get full overflow safety (exp(x − M) ≤ 1,
    exp(−M) ≤ 1, both bounded). Fully-masked rows would get
    M = −inf from masked_fill; the clamp_min(0) forces M = 0 and the
    mask multiply zeroes the row anyway.

    Args:
        scores: attention logits, shape [B, H, T_q, T_k]. T_q may be 1
            (decode); T_k is the sequence length. The query axis is
            not reduced over, so this is also valid for cross-attn.
        mask: bool tensor, broadcast-compatible with `scores`. True =
            attend, False = mask out. Both numerator and denominator
            are zeroed on False entries.
        eps: denominator guard (paper default 1e-6).

    Returns:
        Tensor of the same shape and dtype as `scores`. Each row sums
        to ≤ 1 (≤, not ==, because softpick permits zero total mass
        when every score is ≤ 0).
    """
    s = scores.to(torch.float32)
    # M = per-row max over UNMASKED positions. masked_fill with −inf
    # keeps the −1e9 sentinel (or any masked-out value) from inflating
    # M; .amax then ignores those positions. clamp_min(0) preserves
    # identity when M_true ≤ 0 and prevents the −inf → +inf blow-up
    # in exp(−M) for a fully-masked row.
    M = s.masked_fill(~mask, float("-inf")).amax(dim=-1, keepdim=True).clamp_min(0.0)
    z = (s - M).exp() - (-M).exp()
    m = mask.to(z.dtype)
    num = torch.relu(z) * m
    den = z.abs() * m
    return (num / (den.sum(dim=-1, keepdim=True) + eps)).to(scores.dtype)


# ============================================================================
# #90 Invented residual-stream normalizations. Drop-in for RMSNorm; all are
# O(d) per token with a learnable per-channel gain `g`. Selected via the
# `norm_type` config flag; the internal Q/K norms are left untouched.
# ============================================================================
class PeakNorm(nn.Module):
    """L-infinity norm: divide by the largest-magnitude activation (no sqrt)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        denom = x.abs().amax(dim=-1, keepdim=True) + self.eps
        return self.weight * (x / denom)


class ManhattanNorm(nn.Module):
    """L1 / mean-absolute-deviation norm (no square, no sqrt)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        denom = x.abs().mean(dim=-1, keepdim=True) + self.eps
        return self.weight * (x / denom)


class SquashNorm(nn.Module):
    """DyT-style reduction-free 'norm': g * tanh(alpha * x). No cross-feature
    reduction at all (fully element-wise), so it's the cheapest of the set."""
    def __init__(self, dim: int, alpha_init: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x)


class CenterNorm(nn.Module):
    """Mean-only norm: subtract the feature mean, scale by g (no variance)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x - x.mean(dim=-1, keepdim=True))


class ManifoldNorm(nn.Module):
    """Fractional-power RMS: x / rms(x)**rho, with rho = sigmoid(raw) in (0,1)
    learnable. rho=1 -> RMSNorm (full unit-sphere projection); rho=0 -> no
    normalization (g*x). Inits at rho~=0.98 so it starts as RMSNorm and learns
    HOW MUCH to normalize the residual stream."""
    def __init__(self, dim: int, eps: float = 1e-6, rho_init: float = 4.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.rho_raw = nn.Parameter(torch.tensor(float(rho_init)))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        rho = torch.sigmoid(self.rho_raw)
        return self.weight * (x / rms.pow(rho))


class PNorm(nn.Module):
    """Generalized Lp norm: x / (mean(|x|^p))^(1/p) * g. Unifies the family —
    p=1 is ManhattanNorm, p=2 is RMSNorm, p->inf approaches PeakNorm. Lets us
    sweep the single 'p' knob to find the best aggregate. Scale-invariant and
    all-dimensions by construction (the two properties that mattered)."""
    def __init__(self, dim: int, p: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.p = float(p)
        self.eps = eps

    def forward(self, x):
        denom = (x.abs().pow(self.p).mean(dim=-1, keepdim=True) + self.eps).pow(1.0 / self.p)
        return self.weight * (x / denom)


class CenteredL1Norm(nn.Module):
    """L1 analogue of LayerNorm: subtract the mean, then divide by the mean
    absolute deviation. Robust (L1) scale + centering. Scale-invariant."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xc = x - x.mean(dim=-1, keepdim=True)
        denom = xc.abs().mean(dim=-1, keepdim=True) + self.eps
        return self.weight * (xc / denom)


class ClipNorm(nn.Module):
    """#94 Winsorized RMSNorm: clip |x| to k*mean|x| per token (removing the
    massive-activation outliers DIRECTLY), then RMS-normalize the clipped
    vector. Tests whether the outliers are harmful (clip helps) or functional
    (clip hurts). k via 'clipnorm<k>' in the norm_type string (default 3)."""
    def __init__(self, dim: int, k: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.k = float(k)
        self.eps = eps

    def forward(self, x):
        lim = self.k * x.abs().mean(dim=-1, keepdim=True) + self.eps
        xc = torch.maximum(torch.minimum(x, lim), -lim)
        rms = torch.sqrt(xc.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (xc / rms)


class ChannelScaleNorm(nn.Module):
    """#95 Learnable per-channel PRE-scale, then RMSNorm. The pre-scale lets
    the model down-weight specific outlier channels BEFORE they dominate the
    denominator (the post-norm gain cannot — it acts after the division).
    Init identity, so it starts exactly as RMSNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.pre = nn.Parameter(torch.ones(dim))
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x * self.pre
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class ScaleNorm(nn.Module):
    """Scalar-gain RMSNorm.

    Keeps the RMS denominator but collapses the learned gain to a single
    scalar parameter. Identity at init (`g = 1.0`).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class MedianNorm(nn.Module):
    """#96 Divide by the MEDIAN absolute activation — the maximally outlier-
    robust scale (50% breakdown point). Probes the robustness ceiling past L1."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        med = x.abs().median(dim=-1, keepdim=True).values
        return self.weight * (x / (med + self.eps))


_NORM_REGISTRY = {
    "peak": PeakNorm,
    "manhattan": ManhattanNorm,
    "squash": SquashNorm,
    "center": CenterNorm,
    "manifold": ManifoldNorm,
    "centeredl1": CenteredL1Norm,
    "channelscale": ChannelScaleNorm,
    "scalenorm": ScaleNorm,
    "median": MedianNorm,
}


class RMSNorm(nn.Module):
    """RMSNorm with a family of cheap, mostly identity-init levers
    (docs/research/rmsnorm/plan.md). Base op: y = g · x / √(mean(x²) + eps).

    Default kwargs == plain learnable RMSNorm (per-channel `weight`, init 1) so
    existing configs/checkpoints are byte-identical. Each lever below adds ONE
    param or ONE free trick; all are no-ops at their init value (so step-0 == the
    plain-RMSNorm baseline) except the ones flagged `# own control`.

    Levers (one `norm_type` string each, parsed in `make_norm`):
      N1 reparam_gain  g = 1 + g0           (g0=0)
      N2 bias          + b                  (b=0, d_model)
      N3 temp          / τ                  (τ=1 scalar)
      N4 partial_mix   + λ·x                 (λ=0 scalar)
      N5 floor         √(mean+eps+c²)        (c=0 scalar)
      N6 gain_init     weight init 0.5       (# own control)
      N7 softplus_gain g = softplus(w)       (w≈0.5413 → g≈1)
      N8 partial_vec   + λ⊙x                 (λ=0 vector)
      N9 groups        per-group rms         (~identity)
      N10 stopgrad     detach(rms)
      N11 asym         g₊ / g₋ by sign       (both 1)
      N12 clamp_a      g ∈ [1−a, 1+a]
      N14 learnable_eps eps is a param       (init eps)
      N15 dyntanh      g·tanh(α·x), no stats (# own control)
      N16 double_norm  rms(rms(x))
      N17 centermix    (x − μ·x̄)/rms         (μ=0 scalar; RMS↔LayerNorm knob)
    """
    def __init__(self, dim: int, eps: float = 1e-6, use_reparam_gain: bool = False,
                 use_bias: bool = False, use_temp: bool = False,
                 use_partial_mix: bool = False, use_partial_vec: bool = False,
                 use_floor: bool = False, gain_init: float = 1.0,
                 use_softplus_gain: bool = False, groups: int = 0,
                 stopgrad: bool = False, use_asym: bool = False,
                 clamp_a: float = 0.0, learnable_eps: bool = False,
                 use_dyntanh: bool = False, double_norm: bool = False,
                 use_centermix: bool = False):
        super().__init__()
        self.eps = eps
        self.use_reparam_gain = use_reparam_gain
        self.use_softplus_gain = use_softplus_gain
        self.use_asym = use_asym
        self.use_bias = use_bias
        self.use_temp = use_temp
        self.use_partial_mix = use_partial_mix
        self.use_partial_vec = use_partial_vec
        self.use_floor = use_floor
        self.groups = groups
        self.stopgrad = stopgrad
        self.clamp_a = clamp_a
        self.learnable_eps = learnable_eps
        self.use_dyntanh = use_dyntanh
        self.double_norm = double_norm
        self.use_centermix = use_centermix

        # ---- gain parameterization (mutually exclusive) ----
        if use_asym:  # N11: separate gain for x>0 and x<=0
            self.g_pos = nn.Parameter(torch.ones(dim))
            self.g_neg = nn.Parameter(torch.ones(dim))
        elif use_softplus_gain:  # N7: g = softplus(w), w s.t. g≈1
            w0 = float(torch.log(torch.expm1(torch.tensor(1.0))))  # ≈0.5413
            self.weight = nn.Parameter(torch.full((dim,), w0))
        elif use_reparam_gain:  # N1: g = 1 + g0
            self.g0 = nn.Parameter(torch.zeros(dim))
        else:  # standard / N6 scaled init
            self.weight = nn.Parameter(torch.full((dim,), float(gain_init)))

        # ---- extra per-lever params (identity-init) ----
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(dim))          # N2
        if use_temp:
            self.log_tau = nn.Parameter(torch.zeros(1))         # N3, τ=exp(0)=1
        if use_partial_mix:
            self.mix_lambda = nn.Parameter(torch.zeros(1))      # N4
        if use_partial_vec:
            self.mix_lambda_vec = nn.Parameter(torch.zeros(dim))  # N8
        if use_floor:
            self.floor_c = nn.Parameter(torch.zeros(1))         # N5
        if learnable_eps:
            self.eps_param = nn.Parameter(torch.tensor(float(eps)))  # N14
        if use_dyntanh:
            self.dyt_alpha = nn.Parameter(torch.ones(1))        # N15
        if use_centermix:
            self.center_mu = nn.Parameter(torch.zeros(1))       # N17

    def _gain(self, x):
        if self.use_asym:
            g = torch.where(x > 0, self.g_pos, self.g_neg)
        elif self.use_softplus_gain:
            g = F.softplus(self.weight)
        elif self.use_reparam_gain:
            g = 1.0 + self.g0
        else:
            g = self.weight
        if self.clamp_a > 0:  # N12
            g = g.clamp(1.0 - self.clamp_a, 1.0 + self.clamp_a)
        return g

    def _rms(self, x):
        eps = self.eps_param.clamp_min(0.0) if self.learnable_eps else self.eps
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        if self.use_floor:
            ms = ms + self.floor_c.pow(2)
        rms = torch.sqrt(ms + eps)
        if self.stopgrad:
            rms = rms.detach()
        return rms

    def forward(self, x):
        if self.use_dyntanh:  # N15: normless elementwise op (own control)
            return self._gain(x) * torch.tanh(self.dyt_alpha * x)

        num = x
        if self.use_centermix:  # N17: continuous RMS→LayerNorm
            num = x - self.center_mu * x.mean(dim=-1, keepdim=True)

        if self.groups and self.groups > 1:  # N9: per-group rms
            *lead, d = x.shape
            g = self.groups
            xg = num.reshape(*lead, g, d // g)
            rms = torch.sqrt(xg.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            if self.stopgrad:
                rms = rms.detach()
            x_norm = (xg / rms).reshape(*lead, d)
        else:
            rms = self._rms(x)
            x_norm = num / rms
            if self.double_norm:  # N16: normalize twice
                rms2 = torch.sqrt(x_norm.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                x_norm = x_norm / rms2

        if self.use_temp:  # N3
            x_norm = x_norm / torch.exp(self.log_tau)

        y = self._gain(x) * x_norm
        if self.use_partial_mix:  # N4
            y = y + self.mix_lambda * x
        if self.use_partial_vec:  # N8
            y = y + self.mix_lambda_vec * x
        if self.use_bias:  # N2
            y = y + self.bias
        return y


def make_norm(dim: int, norm_type: str = "rmsnorm", use_layernorm: bool = False):
    """Factory for residual-stream norms. Custom names win; otherwise fall back
    to LayerNorm (if use_layernorm) or RMSNorm. The string suffix
    "rmsnorm_reparam_gain" activates the N1 lever on the new RMSNorm."""
    nt = (norm_type or "rmsnorm").lower()
    # RMSNorm lever family (docs/research/rmsnorm/plan.md): "rmsnorm_<lever>"
    # strings map to RMSNorm kwargs. Each is identity-init unless flagged.
    rms_kw = {}
    if nt.startswith("rmsnorm_"):
        lever = nt[len("rmsnorm_"):]
        _RMS_LEVERS = {
            "reparam_gain": dict(use_reparam_gain=True),     # N1
            "bias": dict(use_bias=True),                     # N2
            "temp": dict(use_temp=True),                     # N3
            "partial_mix": dict(use_partial_mix=True),       # N4
            "floor": dict(use_floor=True),                   # N5
            "scaled_init": dict(gain_init=0.5),              # N6 (own control)
            "softplus_gain": dict(use_softplus_gain=True),   # N7
            "partial_vec": dict(use_partial_vec=True),       # N8
            "stopgrad": dict(stopgrad=True),                 # N10
            "asym": dict(use_asym=True),                     # N11
            "learnable_eps": dict(learnable_eps=True),       # N14
            "dyntanh": dict(use_dyntanh=True),               # N15 (own control)
            "double": dict(double_norm=True),                # N16
            "centermix": dict(use_centermix=True),           # N17
        }
        if lever in _RMS_LEVERS:
            rms_kw = _RMS_LEVERS[lever]
            nt = "rmsnorm"
        elif lever.startswith("group"):  # N9: rmsnorm_group<G>
            try:
                rms_kw = dict(groups=int(lever[len("group"):]))
            except ValueError:
                rms_kw = dict(groups=4)
            nt = "rmsnorm"
        elif lever.startswith("clamp"):  # N12: rmsnorm_clamp<a*100>, e.g. clamp50→a=0.5
            try:
                rms_kw = dict(clamp_a=float(lever[len("clamp"):]) / 100.0)
            except ValueError:
                rms_kw = dict(clamp_a=0.5)
            nt = "rmsnorm"
    if nt.startswith("pnorm"):
        try:
            p = float(nt[len("pnorm"):])
        except ValueError:
            p = 2.0
        return PNorm(dim, p)
    if nt.startswith("clipnorm"):
        try:
            k = float(nt[len("clipnorm"):])
        except ValueError:
            k = 3.0
        return ClipNorm(dim, k)
    if nt in _NORM_REGISTRY:
        return _NORM_REGISTRY[nt](dim)
    if nt == "layernorm" or use_layernorm:
        return nn.LayerNorm(dim, elementwise_affine=True)
    if nt == "rmsnorm":
        # Custom RMSNorm with optional lever kwargs. With rms_kw empty it is
        # mathematically equivalent to nn.RMSNorm (state_dict 'weight' key).
        return RMSNorm(dim, **rms_kw)
    return nn.RMSNorm(dim)


class FocalModulationBlock(nn.Module):
    """148 — Focal Modulation Networks (Yang et al. 2022,
    arXiv:2203.11926, NeurIPS 2022).

    Drop-in replacement for the attention sub-block. Three stages:
      1. **Hierarchical Context Aggregation** — stack of depthwise
         causal Conv1d at multiple kernel sizes (default 3, 5, 7) on
         the time axis. Identity-init (center tap = 1, rest = 0) so
         each conv is a pass-through at step 0. The multi-scale
         context is the *sum* of conv outputs.
      2. **Gather** — `nn.Linear(d_model, d_model, bias=True)` that
         projects the multi-scale context into modulation space.
         **Zero-init** (W=0, b=0) so the modulation signal is exactly
         `0` at step 0 regardless of the conv outputs.
      3. **Modulate** — `x ← x + σ(W_g x + b_g) * (W_q x ⊙ W_h · z)`,
         where `z` is the gathered context, `W_q` is xavier-init, and
         `W_h`, `W_g` are zero-init with `bias_g = -10.0` so the gate
         starts at ≈ 0. With the gather+h_proj zero-init, the
         modulation contribution is **exactly 0 at step 0** ⇒ output
         `= x` bit-identical to baseline (with `use_focal_mod=False`,
         the module is never built and the MHA path is untouched).

    Args:
        d_model: channel dim of the residual stream (B, T, d_model).
        kernels: tuple of depthwise Conv1d kernel sizes (default
            (3, 5, 7)). At least one element required.
        dropout: dropout applied to the block output (default 0.1).

    Forward:
        x: [B, T, d_model]
        Returns: [B, T, d_model] = x + dropout(g · (W_q x ⊙ W_h z))
        where `z = gather(sum_k DWConv_k(left_pad(x, k-1, time)))`.
    """

    def __init__(
        self,
        d_model: int,
        kernels: tuple = (3, 5, 7),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        # Clamp kernels to ints ≥ 1.
        self.kernels = tuple(max(1, int(k)) for k in kernels)
        if not self.kernels:
            raise ValueError("FocalModulationBlock requires at least one kernel size")

        # (1) Hierarchical context aggregation: one depthwise Conv1d
        # per kernel. `groups=d_model` ⇒ depthwise; `bias=False` because
        # the gate absorbs any constant offset. Identity-init: center
        # tap = 1, rest = 0 — so each conv is a pass-through at step 0
        # and the multi-scale context equals `count(kernels) · x` at
        # init. (Doesn't matter for step-0 parity because the gather
        # linear is zero-init, but identity init is more stable at
        # training start.)
        self.context_convs = nn.ModuleList()
        for k in self.kernels:
            conv = nn.Conv1d(
                d_model, d_model, kernel_size=k, padding=0,
                groups=d_model, bias=False,
            )
            with torch.no_grad():
                w = torch.zeros(d_model, 1, k)
                w[:, 0, k // 2] = 1.0
                conv.weight.copy_(w)
            self.context_convs.append(conv)

        # (2) Gather: linear from d_model to d_model. Zero-init ⇒
        # `z = 0` at step 0 exactly. This is the *single* parameter
        # that controls step-0 identity.
        self.gather = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.gather.weight)
        nn.init.zeros_(self.gather.bias)

        # (3) Modulate. Three linears:
        #   - q_proj: W_q x (xavier-init — the only non-zero weight)
        #   - h_proj: W_h z (zero-init — combined with zero-init gather
        #     gives h_mod = 0 exactly at step 0)
        #   - gate_proj: W_g x + b_g (zero-init weight, bias=-10 so
        #     σ(0+b) ≈ 4.5e-5 even if x is not zero-mean)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        self.h_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.h_proj.weight)
        nn.init.zeros_(self.h_proj.bias)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -10.0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model] → conv1d expects [B, d_model, T]
        h = x.transpose(1, 2)
        # Sum of causal depthwise convs at multiple scales. Left-pad
        # by (k-1, 0) on the time axis so position t can only attend
        # to positions ≤ t (we don't set `padding=k//2` on the conv —
        # that would pad both sides and leak future tokens).
        context = x  # [B, T, d_model] (residual start at step 0 ≡ x)
        for k, conv in zip(self.kernels, self.context_convs):
            h_pad = F.pad(h, (k - 1, 0))
            context = context + conv(h_pad).transpose(1, 2)
        # (2) Gather: z = gather(context). Zero-init ⇒ z = 0 at step 0.
        z = self.gather(context)
        # (3) Modulate: output = x + σ(W_g x + b_g) * (W_q x ⊙ W_h z).
        # At step 0: z = 0 ⇒ W_h z = 0 ⇒ (W_q x ⊙ 0) = 0 ⇒ σ(·)·0 = 0
        # ⇒ output = x. Bit-identical to baseline at step 0.
        q = self.q_proj(x)
        h_mod = self.h_proj(z)
        gate = torch.sigmoid(self.gate_proj(x))
        out = x + gate * (q * h_mod)
        return self.dropout(out)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
        use_attn_output_gate: bool = False,
        use_value_channel_gate: bool = False,
        use_attn_output_channel_gate: bool = False,
        use_exclusive_self_attn: bool = False,
        use_kda_channel_gate: bool = False,
        use_value_embed: bool = False,
        value_embed_rank: int | None = None,
        use_query_embed: bool = False,
        use_key_embed: bool = False,
        use_output_embed: bool = False,
        use_q_gain: bool = False,
        use_k_gain: bool = False,
        use_deep_value_embed: bool = False,
        deep_value_embed_hidden: int | None = None,
        use_qk_norm_post_rope: bool = False,
        use_sliding_window: bool = False,
        sliding_window_size: int = 512,
        use_nope: bool = False,
        rope_base: int = 10000,
        use_layernorm: bool = False,
        use_tied_qk: bool = False,
        use_mla: bool = False,
        mla_latent_dim: int | None = None,
        attention_dilation: int = 1,
        use_post_norm: bool = False,
        use_linear_attn: bool = False,
        use_diff_attn: bool = False,
        use_nsa_global: bool = False,
        nsa_block: int = 64,
        use_hybrid_heads: bool = False,
        # 009 FIRE positional encoding (Li et al., NeurIPS 2023,
        # arXiv:2306.02613): drop-in for RoPE. Content-aware additive
        # logit bias γ(|t-s|) · f([φ(x_t); φ(x_s)]). Default off →
        # baseline path bit-identical. f is zero-init so step-0 bias = 0.
        use_fire_pe: bool = False,
        fire_pe_d_phi: int = 4,
        # 024 — Gated Attention (Qiu et al. 2025, arXiv:2505.06708):
        # per-head *scalar* input-conditional sigmoid gate on the head
        # output `o_h = A_h V_h`, post-AV, pre-merge. `o_h ← o_h · 2·σ(W·x+b)`
        # with `W : nn.Linear(d_model, n_heads)` (per-head scalar, NOT
        # the vector form — vector is 42% of the 0.94M model). Gate
        # input `x` is the sublayer input residual (pre-LN), NOT `o_h`
        # itself (circularity). W=0, b=0 init → 2·σ(0) = 1.0 exactly at
        # step 0, so step-0 ≡ baseline. Default off → baseline path
        # bit-identical. Categorically distinct from the pre-existing
        # `use_attn_output_gate` (which is a per-head *learnable scalar
        # gain* `o_h *= (1 + g_h)`, not input-conditional, ReZero-style).
        # See `autoresearch/ideas/024-gated-attention/plan.md`.
        use_gated_attn: bool = False,
        # 013 — CoPE (Golovneva et al. 2024, arXiv:2405.18719):
        # content-aware positional bias replacing RoPE. When on, the
        # Rotary construction is gated off and the attention path is
        # forced to manual (CoPE bias is added to the [B,H,T,T] score
        # tensor). Default off → pre-norm baseline path bit-identical.
        # use_qk_norm_post_rope must be off when use_cope is on
        # (asserted in forward).
        use_cope: bool = False,
        # 020 — Forgetting Transformer (FoX, Lin et al., arXiv:2503.02130):
        # per-head, per-token learnable forget gate that multiplicatively
        # decays the attention probabilities post-softmax, then
        # row-renormalizes. Conservative extension of softmax attention
        # (softmax stays). Identity-init: W_f=0, b_f=+10 → D ≈ 1 within
        # 9% over the full T=2048 context, so the model has to *learn*
        # to forget from scratch. Default off → baseline path
        # bit-identical. Forces the manual attention path (the
        # post-softmax multiply can't go through SDPA's flash kernel).
        use_fox: bool = False,
        # 022 — Softpick (Zuhri/Fuadi/Aji 2025, arXiv:2504.20966):
        # replace `torch.softmax` with `softpick(scores, mask, eps)` in
        # the FIRE manual-path branch. Function swap; no params, no
        # init, no schedule. ε=1e-6 is the paper default, pinned.
        # Default off → softmax baseline path bit-identical. The mask
        # argument is the same `window` tensor already used for
        # `masked_fill` in the FIRE branch, so masked positions
        # contribute zero to both numerator and denominator.
        use_softpick: bool = False,
        # 025 — Scalable-Softmax (SSMax, Nakanishi 2025, arXiv:2501.19399):
        # per-head learnable scalar s_h that multiplies the attention
        # logits by `s_h · log(n)` pre-softmax, where n is the per-query
        # causal key count. Restores per-position sharpness at long
        # range so the softmax distribution does not flatten toward
        # uniform as context grows. Init s_h = 1.0 (the paper's natural
        # starting point). At flag-on, n > 1, the forward is NOT bit-
        # identical to vanilla softmax — the log(n) scaling IS the
        # mechanism, so this is explicitly justified, not a bug.
        # Default off → baseline path bit-identical. Forces the manual
        # attention path (the score-side multiply can't go through
        # SDPA's flash kernel). See
        # `autoresearch/ideas/025-scalable-softmax/plan.md`.
        use_ssmax: bool = False,
        # 021 — Value Residual Learning (Zhou/Wu/Jiang 2024,
        # arXiv:2410.17897). Cross-layer V shortcut: stash the
        # post-W_V, post-GQA, post-transpose V at layer 0; in every
        # later layer l > 0, blend `V_l ← (1-λ_l)·V_l + λ_l·V_1`
        # right after the transpose at `models/layers.py:1479`
        # (post-`repeat_interleave` GQA expansion, post-transpose;
        # both V and V_1 are shape `[B, n_heads, T, d_k]` here,
        # invariant to GQA settings). `λ_l = nn.Parameter(torch.zeros(()))`
        # per-block (0-dim scalar). λ=0 init ⇒ `(1-0)·V + 0·V_1 = V`
        # bit-identical to baseline at step 0 within fp32 rounding
        # noise (one extra multiply-add). `v_residual` is a forward-
        # pass-local stash on this MHA (`self._v_residual`), passed
        # by `MinimalLLM.forward` to blocks 1..N-1 as a positional
        # kwarg. `.detach()` on the stash ⇒ the layer-l blend's
        # gradient does not flow back into layer-0 W_V (each layer's
        # W_V trains on its own attention path; the cross-layer
        # shortcut only learns the blend weight). Default off →
        # baseline path bit-identical (no Parameter created, no
        # stash, no blend). See
        # `autoresearch/ideas/021-value-residual/plan.md`.
        use_value_residual: bool = False,
        # 129 — YOCO shared KV (Sun et al. 2024, arXiv:2405.05254).
        # When set, the MHA skips its W_K, W_V slices of the merged
        # qkvo_proj and reads K, V from the supplied `shared_kv`
        # tuple `(K_g, V_g)`, each of shape `[B, T, kv_size]`. The
        # cross-layer shared KV is computed ONCE on the lower half's
        # final residual stream by `models/yoco.py:GlobalKVHead`,
        # and passed into every upper-half `YOCOLlamaBlock` via
        # the `shared_kv` kwarg on `TransformerBlock.forward`.
        # Inside the MHA, `K_g` still goes through k_norm + RoPE;
        # `V_g` is used as-is. Q is computed normally from x and
        # goes through q_norm + RoPE per the standard path. With
        # `use_shared_kv=False` (default) the MHA never reads
        # `shared_kv` and the baseline K, V projection path is
        # bit-identical. See `autoresearch/ideas/129-yoco/idea.md`.
        use_shared_kv: bool = False,
        # 134 — Mega EMA on V (Ma et al. 2022, arXiv:2209.10655). When
        # set, the V stream is concatenated with `V_ema = W_V @ u` where
        # `u_t = β·u_{t-1} + (1-β)·x_t` is a per-channel exponential
        # moving average over the residual stream input. `β ∈ [0, 1]`
        # is parametrized as `σ(mega_beta_raw)` so it stays bounded
        # during training; raw is zero-init ⇒ β=0.5 at step 0 (the
        # natural "half-smoothed" midpoint). At tiny1m3m the doubled
        # V stream has shape `[B, T, 2·kv_size] = [B, T, d_model]`
        # (since 2·kv_size = 2·n_kv_heads·d_k = n_heads·d_k = d_model),
        # so the standard head reshape still works (n_kv_heads effective
        # doubles to match n_heads). The first-half (V_raw) and second-
        # half (V_ema) compete via softmax over the doubled K dim.
        # Cost: 1 scalar/layer (12 at tiny1m3m, negligible). NOT
        # byte-identical to baseline at step 0 — the concat doubles the
        # V stream. Default off → baseline path bit-identical.
        # See `autoresearch/ideas/134-mega-ema/idea.md`.
        use_mega: bool = False,
        mega_beta: float = 0.9,
        mega_use_input: bool = True,
        norm_type: str = "rmsnorm",
        qk_norm_type: str = "rmsnorm",
        v_norm_type: str = "",
        # #16 QK-Norm (Dehghani et al. 2023, ViT-22B, arXiv:2302.05442):
        # override the default Q/K norm (RMSNorm) with `nn.LayerNorm(d_head)`
        # on the head-dim axis, before the attention dot product. Bounds
        # the per-head logit `Q·K/√d_head` to `|·| ≤ √d_head`. Init γ=1, β=0
        # → identity at step 0. Default off → Q/K stay on RMSNorm, the
        # current baseline path is bit-identical. Only the Q/K norms are
        # affected; the residual stream norms stay on `norm_type`/
        # `use_layernorm`. See autoresearch/ideas/016-qk-norm/plan.md.
        use_qk_layernorm: bool = False,
        # 029 — V-Norm (Wortsman et al. 2023, arXiv:2309.14322):
        # per-head `nn.LayerNorm(d_head)` on V along `d_head` before the
        # AV product, symmetric partner of 016's QK-Norm. Bounds the
        # per-head V vector magnitude so outlier V entries do not
        # dominate the AV aggregation. Independent module (no weight
        # sharing with q_norm/k_norm). v_norm_type takes precedence
        # when also set (explicit > implicit). Default off → no v_norm
        # module is built; baseline path bit-identical. See
        # autoresearch/ideas/029-v-norm/plan.md.
        use_v_layernorm: bool = False,
        use_multiscale_heads: bool = False,
        use_parallel_block: bool = False,
        use_attn_sink: bool = False,
        # Query-tweaks (29 experiments, 6 batches — see plan.md).
        q_norm_type: str = "rmsnorm",
        use_alibi_bias: bool = False,
        use_q_temp_token: bool = False,
        use_cosine_attn: bool = False,
        use_qk_bilinear: bool = False,
        use_talking_heads_q: bool = False,
        use_per_head_rope_base: bool = False,
        partial_rotary_p: float = 1.0,
        use_q_expansion: bool = False,
        use_decoupled_content_pos: bool = False,
        use_antisym_qk: bool = False,
        use_q_per_head_bias: bool = False,
        use_q_per_channel_gain: bool = False,
        use_q_hd_gain: bool = False,
        use_q_norm_gate: bool = False,
        use_q_lowrank_refine: bool = False,
        q_lowrank_refine_rank: int = 8,
        use_q_layerscale: bool = False,
        use_q_softplus_gain: bool = False,
        use_q_head_mix: bool = False,
        use_q_time_conv: bool = False,
        use_q_ema_smooth: bool = False,
        q_ema_alpha: float = 0.0,
        use_q_feature_map: bool = False,
        q_feature_map_hidden: int = 64,
        use_q_per_token_rope: bool = False,
        q_per_token_rope_hidden: int = 32,
        use_q_noise_reg: bool = False,
        # O1 TalkingHeadsOut (docs/research/attention_output/plan.md,
        # Batch 1). Cross-head mix on the *post-softmax* attention
        # output (sibling of Q5 talking_heads_q, but on the output
        # side of softmax). M init I → einsum is a no-op at step 0.
        use_talking_heads_out: bool = False,
        # 147 — DropKey (Xu et al. 2022, arXiv:2207.01058). Per-head,
        # per-token Bernoulli mask on K during training, applied AFTER
        # RoPE + GQA repeat_interleave (so K is in [B, n_heads, T, d_k]
        # layout). Mask shape `[B, n_heads, T, 1]`; elements drawn
        # i.i.d. Bernoulli(1 - drop_key_rate); rescale by `1/(1-p)`
        # so the expected K magnitude matches the un-masked baseline
        # (inverted-dropout convention, matches `F.dropout` and
        # modded-nanogpt value-residual rescale). Inference
        # (`self.training == False`) and `drop_key_rate=0` both skip
        # the mask ⇒ forward graph bit-identical to baseline. Distinct
        # from value-side regularizers (use_value_channel_gate is
        # additive on V, use_kda_channel_gate is bounded multiplicative
        # on V) and from score-side regularizers (use_fox is post-
        # softmax A·D, use_ssmax is logit-temperature). The lever is
        # *where* the random gate fires (on K, not V or A) — that is
        # the structural choice with a different inductive bias. See
        # `autoresearch/ideas/147-dropkey/idea.md`.
        use_drop_key: bool = False,
        drop_key_rate: float = 0.1,
        # O-family: a single cheap op on the post-softmax attention output
        # [B,H,T,D] (pre head-merge). One string selects the lever; params are
        # built in _init_output_op and applied at the [B,H,T,D] choke point.
        # See docs/research/attention_output/plan.md.
        out_op: str = "",
    ):
        super().__init__()
        # #75 Post-norm: when set, the norm is applied AFTER the
        # residual addition instead of before. Implementation:
        # compute (norm, residual) inside the function but apply
        # the norm to (x + sublayer_out) before returning.
        self.use_post_norm = use_post_norm
        # #79 LayerNorm vs RMSNorm. When set, every nn.RMSNorm in
        # the block is replaced with nn.LayerNorm (with learned
        # scale + bias). Tests whether the choice of norm is a
        # real architecture lever.
        self.use_layernorm = use_layernorm
        # #80 Linear attention (Performer-style): when set, replace
        # softmax attention with positive-feature kernel attention
        # using phi(x) = elu(x) + 1.
        self.use_linear_attn = use_linear_attn
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        
        # ============ MERGED QKVO PROJECTION ============
        # Instead of 4 separate Linear layers, use single merged projection
        q_size = d_model
        kv_size = self.n_kv_heads * self.d_k
        o_size = d_model
        
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_size = q_size + 2 * kv_size  # Q + K + V sizes
        
        # Single parameter tensor for all projections
        # Shape: [Q_size + K_size + V_size + O_size, d_model]
        # #72 Tied QK: when set, Q and K share a single W (PaLM-style).
        # We allocate a SEPARATE parameter and don't use the Q/K slices
        # of the merged qkvo_proj in that case.
        self.use_tied_qk = use_tied_qk
        if use_tied_qk:
            self.qk_proj = nn.Parameter(torch.empty(q_size + kv_size, d_model))
        # #73 MLA: latent K,V with down/up projections. Separate
        # parameters; the standard Q/K/V slices of qkvo_proj are unused
        # when MLA is on.
        self.use_mla = use_mla
        self.mla_latent_dim = mla_latent_dim if mla_latent_dim is not None else max(8, d_model // 4)
        if use_mla:
            self.mla_dkv = nn.Parameter(torch.empty(self.mla_latent_dim, d_model))
            self.mla_uk  = nn.Parameter(torch.empty(kv_size, self.mla_latent_dim))
            self.mla_uv  = nn.Parameter(torch.empty(kv_size, self.mla_latent_dim))
        self.qkvo_proj = nn.Parameter(
            torch.empty(q_size + 2 * kv_size + o_size, d_model)
        )
        
        # Initialize all weights with std=0.02
        with torch.no_grad():
            torch.nn.init.normal_(self.qkvo_proj, mean=0.0, std=0.02)
            if use_tied_qk:
                torch.nn.init.normal_(self.qk_proj, mean=0.0, std=0.02)
            if use_mla:
                torch.nn.init.normal_(self.mla_dkv, mean=0.0, std=0.02)
                torch.nn.init.normal_(self.mla_uk,  mean=0.0, std=0.02)
                torch.nn.init.normal_(self.mla_uv,  mean=0.0, std=0.02)
        # ================================================
        
        # #91 Robust QK-norm: q_norm/k_norm can use any invented norm (e.g.
        # pnorm1.5) so the attention-logit dot product is outlier-robust.
        # #16 QK-Norm (Dehghani et al. 2023, ViT-22B): the Q/K norms
        # default to RMSNorm (`qk_norm_type="rmsnorm"`), but when
        # `use_qk_layernorm=True` we override to `nn.LayerNorm(d_head)` —
        # a stricter bound on per-head logit magnitude. The override is
        # OR'd with the global `use_layernorm` so either flag flips the
        # Q/K norm; the residual stream norms stay on `norm_type`/
        # `use_layernorm` (the Q/K override is strictly local).
        _qk_use_ln = bool(self.use_layernorm) or bool(use_qk_layernorm)
        self.q_norm = make_norm(self.d_k, qk_norm_type, _qk_use_ln)
        self.k_norm = make_norm(self.d_k, qk_norm_type, _qk_use_ln)
        # #92 Robust V-norm: optionally normalize V (per head) before the
        # softmax-weighted sum, so outlier value channels don't dominate the
        # aggregated output. Off by default ("" / "none").
        self.use_v_norm = v_norm_type not in ("", "none", None)
        if self.use_v_norm:
            self.v_norm = make_norm(self.d_k, v_norm_type, self.use_layernorm)
        # 029 — V-Norm override: when `use_v_layernorm=True` and the
        # closed-#92 v_norm_type is off, build a per-head
        # `nn.LayerNorm(d_k)` on V (γ=1, β=0 default → identity at
        # step 0). Mirrors the use_qk_layernorm override above. The
        # v_norm_type branch above takes precedence when also set
        # (explicit > implicit — the closed-#92 lever wins).
        elif use_v_layernorm:
            self.use_v_norm = True
            self.v_norm = nn.LayerNorm(self.d_k)

        # 013 — CoPE: gate the Rotary construction. When on, RoPE is
        # replaced by a content-aware bias (CoPE) added to the
        # attention logits. When off, the baseline Rotary is built
        # unchanged. Saves a few KB/block of cache when CoPE is on.
        self.use_cope = use_cope
        if self.use_cope:
            self.rotary = None
            self.cope = CoPE(d_model, n_heads, max_seq_len)
        else:
            self.rotary = Rotary(self.d_k, max_seq_len, base=rope_base)
            self.cope = None
        # 020 — Forgetting Transformer (FoX). Built unconditionally when
        # the flag is on; never called when off. FoX is a CONSERVATIVE
        # extension of softmax — the softmax stays, the projection
        # stays, the V path is unchanged. The per-head gate W_f is
        # zero-init and b_f = +10 (see `models/fox.py` for the
        # identity-init derivation at T=2048), so step-0 D is within
        # 9% of all-ones over the full context — the row-renorm after
        # D*attn_w makes the attention output within ~1e-2 of the
        # no-FoX baseline (test (e) in plan.md pins this).
        self.use_fox = use_fox
        if self.use_fox:
            self.fox = FoX(d_model, n_heads)
        # 022 — Softpick flag. Stored on self so the FIRE branch can
        # branch on it at the swap site. No module to build — the
        # `softpick` helper at the top of this file is the entire
        # implementation.
        self.use_softpick = use_softpick
        # 025 — SSMax: one learnable per-head scalar s_h (init 1.0).
        # Built lazily; the parameter is never referenced when the
        # flag is off, so the baseline path is bit-identical. The
        # multiplier is `s_h · log(n)` where n = (i+1) is the per-
        # query causal key count. log_n is a length-T vector computed
        # in the forward (no precomputed buffer — T is fixed at
        # construction time but the multiplier is essentially free).
        self.use_ssmax = use_ssmax
        if self.use_ssmax:
            self.ssmax_s = nn.Parameter(torch.ones(self.n_heads))
        # 021 — Value Residual: per-block 0-dim scalar `lambda_v`
        # (init 0 ⇒ identity blend at step 0). `_v_residual` is a
        # forward-pass-local stash on this MHA (set by layer 0,
        # read by `MinimalLLM.forward` and passed to layers 1..N-1).
        # Initialized to `None` so the attribute always exists for
        # the `block.attention._v_residual` readout in the model
        # forward loop, even before the first forward pass.
        self.use_value_residual = use_value_residual
        if self.use_value_residual:
            self.lambda_v = nn.Parameter(torch.zeros(()))
            self._v_residual = None
        # 129 — YOCO: when set, the MHA reads shared K, V from the
        # `shared_kv` kwarg on `forward`, skipping the W_K, W_V
        # slices of the merged qkvo_proj. Default off → the W_K,
        # W_V slices are used as in the standard path; baseline
        # forward is bit-identical. See
        # `models/yoco.py` and `autoresearch/ideas/129-yoco/idea.md`.
        self.use_shared_kv = use_shared_kv
        self.dropout = dropout
        self.use_attn_output_gate = use_attn_output_gate
        if self.use_attn_output_gate:
            self.attn_output_gate = nn.Parameter(torch.zeros(n_heads))
        self.use_value_channel_gate = use_value_channel_gate
        if self.use_value_channel_gate:
            self.value_channel_gate = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.use_attn_output_channel_gate = use_attn_output_channel_gate
        if self.use_attn_output_channel_gate:
            self.attn_output_channel_gate = nn.Parameter(torch.zeros(n_heads, self.d_k))
        # #107 Exclusive self-attn: subtract the projection of the head
        # output onto its current-token value vector. Per-head scalar gate
        # is zero-init so step 0 is the baseline graph.
        self.use_exclusive_self_attn = use_exclusive_self_attn
        if self.use_exclusive_self_attn:
            self.exclusive_self_attn = nn.Parameter(torch.zeros(n_heads))
        # 109 — KDA channel gate: per-(head, channel) *bounded* diagonal
        # `2·σ(g)` gain on V before the AV product. KDA's
        # `Γ = diag(γ_1, …, γ_d)` decay, ported to the V stream. Param
        # shape `(n_heads, d_k)`, zero-init ⇒ `2·σ(0) = 1.0` exactly
        # ⇒ step-0 ≡ baseline when flag on at step 0, AND when flag
        # off (no Parameter created, no application site taken). The
        # closed `use_value_channel_gate` uses the unbounded
        # `1 + g` form; the bounded `2·σ(g)` parametrization here is
        # what makes this lever distinct (per-channel gains live in
        # (0, 2) — bounded during training, prevents drift to
        # extremes). n_heads × d_k scalars per layer = 64 at
        # tiny1m3m (H=4, d_k=16). See
        # `autoresearch/ideas/109-kda-channel-gate/idea.md`.
        self.use_kda_channel_gate = use_kda_channel_gate
        if self.use_kda_channel_gate:
            self.kda_channel_gate = nn.Parameter(torch.zeros(n_heads, self.d_k))
        # 024 — Gated Attention: per-head *scalar* input-conditional
        # sigmoid gate on `o_h = A_h V_h`. `nn.Linear(d_model, n_heads)`,
        # both weight and bias zero-init. At init: `2·σ(0) = 1.0` exactly
        # → step-0 ≡ baseline. Cheap (H·d_model + H params per layer,
        # 1.3% of the 0.94M model at tiny1m3m).
        self.use_gated_attn = use_gated_attn
        if self.use_gated_attn:
            self.gated_attn_proj = nn.Linear(d_model, n_heads, bias=True)
            nn.init.zeros_(self.gated_attn_proj.weight)
            nn.init.zeros_(self.gated_attn_proj.bias)

        # #29 value embeddings: project the (factorized) token embedding into the
        # V subspace and add it to the values. Raw zero-init weight (not nn.Linear)
        # so it (a) starts as an exact baseline, (b) trains immediately via Muon,
        # and (c) draws no RNG at init — keeping every other weight bit-identical to
        # the control run, so the screen isolates the mechanism, not a re-seed.
        self.use_value_embed = use_value_embed
        if self.use_value_embed:
            assert value_embed_rank is not None, "value_embed_rank required"
            self.value_embed_proj = nn.Parameter(torch.zeros(self.kv_size, value_embed_rank))
        # #45 deep value embeddings: 2-layer non-linear V projection.
        # V += GELU(ve @ W1) @ W2.
        # F.linear(ve, W1) computes ve @ W1.T, so W1 is stored as
        # [hidden, emb_rank] (like Linear's weight convention).
        # Same for W2: stored as [kv_size, hidden].
        # Both zero-init so step 0 = exact baseline.
        # Cost per layer: hidden × emb_rank + kv_size × hidden.
        # For Screen10M20M (emb_rank=48, hidden=96, kv_size=48): 9,216.
        # Total: 24 × 9,216 = 221,184 params (+2.9%).
        self.use_deep_value_embed = use_deep_value_embed
        if self.use_deep_value_embed:
            assert value_embed_rank is not None, "value_embed_rank required for deep V-embed"
            assert deep_value_embed_hidden is not None, "deep_value_embed_hidden required"
            self.deep_value_embed_W1 = nn.Parameter(
                torch.zeros(deep_value_embed_hidden, value_embed_rank)
            )
            self.deep_value_embed_W2 = nn.Parameter(
                torch.zeros(self.kv_size, deep_value_embed_hidden)
            )
        # #30 query embeddings: same trick on Q. Tests whether V's win is
        # V-specific or generalizes to "token identity straight into attention."
        self.use_query_embed = use_query_embed
        if self.use_query_embed:
            assert value_embed_rank is not None, "value_embed_rank required (used for Q too)"
            self.query_embed_proj = nn.Parameter(torch.zeros(self.q_size, value_embed_rank))
        # #31 key embeddings: same trick on K. K goes through RoPE after the
        # injection, so the projection's term gets positionally rotated — a
        # different operating point from V (no RoPE) or Q (also RoPE'd).
        self.use_key_embed = use_key_embed
        if self.use_key_embed:
            assert value_embed_rank is not None, "value_embed_rank required (used for K too)"
            self.key_embed_proj = nn.Parameter(torch.zeros(self.kv_size, value_embed_rank))
        # #33 output embeddings: same trick, applied AFTER the O projection
        # (output side of attention, not input side). This is the
        # modded-nanogpt speedrun "value embeddings" position — the token
        # identity bypasses the attention computation entirely and lands
        # straight in the residual. Tests "is V-embed winning because V is
        # a unique position, or because any token-signal-to-residual helps?"
        # Shape: [d_model, emb_rank] (one d_model, not kv_size — the output
        # is full d_model, not per-head).
        self.use_output_embed = use_output_embed
        if self.use_output_embed:
            assert value_embed_rank is not None, "value_embed_rank required (used for O too)"
            self.output_embed_proj = nn.Parameter(torch.zeros(self.d_model, value_embed_rank))
        # #37 per-head Q-gain: a learnable per-head scalar that multiplies
        # the Q vector after norm+RoPE. Zero-init so the model starts as
        # exact baseline (1 + 0 = 1). Equivalent to a per-head
        # temperature on the attention scores. Known modded-nanogpt
        # speedrun trick (q_gain in the parameter-golf baseline). Cost:
        # n_heads scalars per layer = 6 × 24 = 144 total extra params.
        # Non-embed lever: changes the attention math, not the inputs.
        self.use_q_gain = use_q_gain
        if self.use_q_gain:
            self.q_gain = nn.Parameter(torch.zeros(self.n_heads))
        # #42 per-head K-gain: symmetric to q_gain, but on K. Tests
        # whether scaling K helps as much as scaling Q, and whether
        # both are additive (V+q+k_gain might beat V+q_gain).
        self.use_k_gain = use_k_gain
        if self.use_k_gain:
            self.k_gain = nn.Parameter(torch.zeros(self.n_heads))
        # 134 — Mega EMA: per-channel β scalar, parametrized via sigmoid
        # so β ∈ [0, 1] is bounded during training. raw init 0 ⇒
        # β=0.5 at step 0 (midpoint between the paper's β=0 "no
        # smoothing" and β=1 "constant EMA" extremes). The same W_V
        # slice of `qkvo_proj` is reused for the EMA projection
        # (zero new projection params — the EMA input `u` flows
        # through the existing V weight matrix). Only one extra
        # parameter: d_model scalars per layer for the β buffer.
        self.use_mega = use_mega
        if self.use_mega:
            self.mega_beta_raw = nn.Parameter(torch.zeros(d_model))
            self.mega_use_input = mega_use_input
            # At tiny1m3m, 2·n_kv_heads must equal n_heads for the
            # head reshape to work without GQA bookkeeping changes.
            # We assert this and document the constraint — other
            # scales need extra plumbing.
            assert 2 * self.n_kv_heads == self.n_heads, (
                f"use_mega requires 2·n_kv_heads == n_heads "
                f"(got n_kv_heads={self.n_kv_heads}, n_heads={self.n_heads}); "
                f"the doubled V stream only bit-fits when n_kv_heads = n_heads / 2"
            )
        # #49 QK-norm-post-RoPE: apply RMSNorm to Q,K AFTER RoPE (modded-
        # nanogpt trick) instead of the default BEFORE RoPE. Different
        # mathematical operating point. Flag-only, no extra params.
        self.use_qk_norm_post_rope = use_qk_norm_post_rope
        # #53 NoPE: skip the rotary positional embedding entirely. The
        # Q,K tensors still go through RMSNorm (norm is the Q/K
        # magnitude stabilizer, separate concern from position), but
        # the rotary is bypassed.
        self.use_nope = use_nope
        # 009 FIRE positional encoding — build the FIREBias module
        # unconditionally (zero FLOPs when use_fire_pe=False, the
        # forward branch is gated). f weights are zero-init so step-0
        # bias = 0 even with the flag on.
        self.use_fire_pe = use_fire_pe
        self.fire_bias = FIREBias(
            d_model=d_model, n_heads=n_heads, max_seq_len=max_seq_len,
            d_phi=fire_pe_d_phi,
        )
        # #63 RoPE base: control the wavelength of the rotary. The
        # default base=10000 is GPT-Neo style; Llama uses 500000 which
        # extends the useful positional range. Tests whether the
        # default decay is hurting at our seq_len=2048.
        self.rope_base = rope_base
        # #51 sliding-window attention: build a [T, T] causal-local
        # boolean mask once at init and reuse. True = attend,
        # False = mask out. Built for max_seq_len; the SDPA call slices
        # the upper-left [seq_len, seq_len] submatrix. causal AND
        # local-window — both are required, otherwise the mask lets
        # each position attend to its future.
        # #74 dilated attention: same as SWA but the window consists
        # of every `attention_dilation`-th position (dilation=1 is
        # the contiguous SWA case; dilation=2 takes every other
        # position in the window range; etc.).
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.attention_dilation = max(1, int(attention_dilation))
        if self.use_sliding_window:
            idx = torch.arange(max_seq_len)
            diff = idx[:, None] - idx[None, :]
            if self.attention_dilation == 1:
                # contiguous SWA — original mask
                self.register_buffer(
                    "_sliding_window_mask",
                    (diff >= 0) & (diff < sliding_window_size),
                    persistent=False,
                )
            else:
                # dilated: keep positions j where diff is a multiple
                # of dilation AND within the window
                self.register_buffer(
                    "_sliding_window_mask",
                    (diff >= 0)
                    & (diff < sliding_window_size)
                    & ((diff % self.attention_dilation) == 0),
                    persistent=False,
                )

        # #87 Differential Attention (Microsoft DIFF Transformer, adapted):
        # split each head's d_k in half and compute TWO softmax maps; the
        # output is map1 - lambda * map2, which cancels common-mode
        # attention noise. We exploit SDPA's linearity in V:
        #   (softmax1 - lam*softmax2) @ V == sdpa(Q1,K1,V) - lam*sdpa(Q2,K2,V)
        # so two flash-attn calls suffice (no [T,T] materialization). lambda
        # is a learnable per-head scalar (paper init 0.5); a per-head RMSNorm
        # + (1-lam_init) scale stabilizes the subtracted output. Requires an
        # even d_k. Not identity-init (it's a genuinely different operator).
        self.use_diff_attn = use_diff_attn
        if self.use_diff_attn:
            assert self.d_k % 2 == 0, "diff-attn needs even d_k"
            self._diff_lambda_init = 0.5
            self.diff_lambda = nn.Parameter(
                torch.full((self.n_heads,), self._diff_lambda_init)
            )
            self.diff_norm = nn.RMSNorm(self.d_k)
        # #88 NSA-style compressed-global branch (DeepSeek Native Sparse
        # Attention, adapted): keep the cheap local window, and ADD a global
        # branch where each query attends to block-mean-pooled K/V summaries
        # (block size nsa_block). Gives every token full-context reach at
        # O(T * T/block) cost. The per-head gate is ZERO-INIT, so step 0 is
        # exactly the local-attention baseline and the global branch earns
        # its weight during training. Block-causal: a query only sees blocks
        # that ended at or before its position (no intra-block future leak).
        self.use_nsa_global = use_nsa_global
        self.nsa_block = max(1, int(nsa_block))
        if self.use_nsa_global:
            self.nsa_gate = nn.Parameter(torch.zeros(self.n_heads))
        # #89 Hybrid heads (DeepSeek-V4 hybrid attention at head granularity):
        # the first half of the heads attend within a local window, the second
        # half attend over the full causal context — in EVERY layer. Zero extra
        # params; a single SDPA call with a per-head [H,T,T] boolean mask. The
        # window reuses sliding_window_size (default 512).
        self.use_hybrid_heads = use_hybrid_heads
        if self.use_hybrid_heads:
            idx = torch.arange(max_seq_len)
            diff = idx[:, None] - idx[None, :]
            causal = diff >= 0
            local = causal & (diff < sliding_window_size)
            n_local = self.n_heads // 2
            per_head = torch.stack(
                [local if h < n_local else causal for h in range(self.n_heads)],
                dim=0,
            )  # [H, T, T] — True = attend
            self.register_buffer("_hybrid_head_mask", per_head, persistent=False)
        # #97 Multi-scale heads: each head gets a DIFFERENT sliding-window size,
        # geometrically spread around sliding_window_size (head h window =
        # w * 2^(h - H//2), e.g. for w=384/4 heads -> 96/192/384/768). Tests
        # whether receptive-field DIVERSITY beats a single uniform window
        # (SWA384 won uniformly; full attention was flat). Zero extra params.
        self.use_multiscale_heads = use_multiscale_heads
        if self.use_multiscale_heads:
            idx = torch.arange(max_seq_len)
            diff = idx[:, None] - idx[None, :]
            causal = diff >= 0
            masks = []
            for h in range(self.n_heads):
                wh = max(1, int(sliding_window_size * (2.0 ** (h - self.n_heads // 2))))
                masks.append(causal & (diff < wh))
            self.register_buffer("_multiscale_mask", torch.stack(masks, dim=0), persistent=False)
        # #99 Attention sink slot (softmax-off-by-one): append a zero key/value
        # so a query can attend to "nothing" (denominator gets a +1 term)
        # instead of being forced to dump probability mass on a real token.
        # Attention sinks are where massive activations originate, so this
        # attacks the outlier problem at its source. Zero extra params.
        self.use_attn_sink = use_attn_sink
        # 147 — DropKey (Xu et al. 2022). Per-head Bernoulli gate on K
        # during training. Stored on self; the actual mask sample and
        # apply is in `forward()` right after `K.transpose(1, 2)`. No
        # extra params — pure stochastic regularizer. Default off →
        # forward graph bit-identical to baseline.
        self.use_drop_key = use_drop_key
        self.drop_key_rate = float(drop_key_rate)

        # ============================================================================
        # Query-tweaks: 29 mechanisms (Batches 1-6, see plan.md). All
        # parameters stored here; the actual math runs in forward().
        # ============================================================================
        # Q-side normalization. Defaults to qk_norm_type at the
        # LLMConfig level, so existing configs are bit-identical.
        # Batch 4 sets this flag directly per experiment.
        # #16 QK-Norm: the Q-side override at line 772 REPLACES the
        # line-556 q_norm (this attribute is what `forward()` uses).
        # Apply the same LayerNorm override here so `use_qk_layernorm=True`
        # actually flips BOTH Q and K to LayerNorm — without this the
        # Q-side `make_norm` would silently leave Q on RMSNorm.
        self.q_norm = make_norm(self.d_k, q_norm_type, _qk_use_ln)

        # Q1 ALiBi-style per-head distance bias. scores += -m_h*(i-j).
        # m_h init 0 → step-0 == baseline.
        self.use_alibi_bias = use_alibi_bias
        if use_alibi_bias:
            self.alibi_slope = nn.Parameter(torch.zeros(self.n_heads))
        # Q2 Token-conditioned per-head temperature. Q *= (1 + tanh(x·w_h)).
        # w_h init 0 → tanh(0) = 0 → step-0 == baseline.
        self.use_q_temp_token = use_q_temp_token
        if use_q_temp_token:
            self.q_temp_w = nn.Parameter(torch.zeros(self.n_heads, d_model))
        # Q3 Cosine attention. L2-normalize Q and K; per-head learnable τ.
        # τ init 1 / sqrt(d_k) (the standard inverse-temperature), so
        # step-0 score scale ≈ the standard softmax-QK scale.
        self.use_cosine_attn = use_cosine_attn
        if use_cosine_attn:
            self.cosine_tau = nn.Parameter(torch.full((self.n_heads,), 1.0))
        # Q4 Per-channel relevance. score = Q^T diag(d_h) K. d_h init 1.
        self.use_qk_bilinear = use_qk_bilinear
        if use_qk_bilinear:
            self.qk_bilinear_d = nn.Parameter(torch.ones(self.n_heads, self.d_k))
        # Q5 Talking-heads on Q. n_h × n_h mix on attention logits pre-softmax.
        # M init I → step-0 == baseline.
        self.use_talking_heads_q = use_talking_heads_q
        if use_talking_heads_q:
            self.talking_heads_M = nn.Parameter(
                torch.eye(self.n_heads, self.n_heads)
            )
        # Q6 Per-head learnable RoPE base. Stored as raw log-frequency
        # so we can rebase as base * exp(log_h). log_h init 0 → θ_h = base.
        self.use_per_head_rope_base = use_per_head_rope_base
        if use_per_head_rope_base:
            self.per_head_rope_log = nn.Parameter(torch.zeros(self.n_heads))
        # Q7 Partial rotary. 0 < p <= 1 fraction of Q/K dims rotated.
        # p=1 (default) = full RoPE.
        self.partial_rotary_p = float(partial_rotary_p)
        # Q8 Multi-query expansion. Project Q to 2·q_size, run 2 reads, mean.
        # 2nd-query zero-init → step-0 == baseline.
        self.use_q_expansion = use_q_expansion
        if use_q_expansion:
            self.q_expand = nn.Parameter(torch.zeros(q_size, d_model))
        # Q9 Decoupled content/position attention (DeBERTa-style).
        # The shared relative-position module is created once at the
        # MinimalLLM level and threaded into forward() so all layers reuse
        # the same clipped distance table.
        self.use_decoupled_content_pos = use_decoupled_content_pos
        # Q10 Antisymmetric Q·K coupling. add Q^T S K, S skew-init 0.
        # S is stored as a full d_k×d_k; we enforce skew in forward.
        self.use_antisym_qk = use_antisym_qk
        if use_antisym_qk:
            self.antisym_S = nn.Parameter(torch.zeros(self.d_k, self.d_k))
        # Q17 Per-head bias. Q += b_h (per-head×channel) post-RoPE.
        # b_h init 0 → step-0 == baseline.
        self.use_q_per_head_bias = use_q_per_head_bias
        if use_q_per_head_bias:
            self.q_per_head_bias = nn.Parameter(
                torch.zeros(self.n_heads, self.d_k)
            )
        # Q18 Per-channel gain. Q *= g_d (per-channel) post-RoPE.
        # g_d init 1 → step-0 == baseline.
        self.use_q_per_channel_gain = use_q_per_channel_gain
        if use_q_per_channel_gain:
            self.q_per_channel_gain = nn.Parameter(torch.ones(self.d_k))
        # Q19 Head×channel gain. Q *= g_hd post-RoPE.
        self.use_q_hd_gain = use_q_hd_gain
        if use_q_hd_gain:
            self.q_hd_gain = nn.Parameter(
                torch.ones(self.n_heads, self.d_k)
            )
        # Q20 Norm-gate. per-head scalar σ(a_h·‖x‖ + b_h) on Q.
        # a_h init 0, b_h init 0 → σ(0) = 0.5 → step-0 != baseline.
        # We use a_h=0,b=0 with a "shift" so g_h=1 at init: use
        # g_h = σ(a_h·‖x‖ + b_h) + 0.5 (a shift). Or simpler: use
        # a tiny init and an explicit bias-to-1, gated as 1+gate.
        # Implementation: store (a_h, b_h); gate = 1 + (σ(a·‖x‖+b) - 0.5).
        self.use_q_norm_gate = use_q_norm_gate
        if use_q_norm_gate:
            self.q_norm_gate_a = nn.Parameter(torch.zeros(self.n_heads))
            self.q_norm_gate_b = nn.Parameter(torch.zeros(self.n_heads))
        # Q21 Low-rank refine. Q += (W1·x) @ W2, both zero-init.
        self.use_q_lowrank_refine = use_q_lowrank_refine
        if use_q_lowrank_refine:
            r = max(1, int(q_lowrank_refine_rank))
            self.q_refine_W1 = nn.Parameter(torch.zeros(r, d_model))
            self.q_refine_W2 = nn.Parameter(torch.zeros(q_size, r))
        # Q22 LayerScale on Q. Q *= (1 + ls_d) per-channel post-RoPE.
        self.use_q_layerscale = use_q_layerscale
        if use_q_layerscale:
            self.q_layerscale = nn.Parameter(torch.zeros(self.d_k))
        # Q23 Softplus gain. Q *= softplus(g_h) per-head — always ≥ 0.
        # g_h init 0 → softplus(0) = ln(2) ≈ 0.693 → step-0 != baseline.
        # Use a "shift to 1" form: Q *= 1 + softplus(g_h) - ln(2). Or
        # accept the 0.693× init (clearly different from identity) and
        # measure whether it learns back. We pick the latter — identity
        # at step 0 is *not* required (the score has just been rescaled
        # by a constant; the optimizer will see a useful gradient).
        # Per plan.md, "softplus gain" doesn't claim identity-init.
        self.use_q_softplus_gain = use_q_softplus_gain
        if use_q_softplus_gain:
            self.q_softplus_g = nn.Parameter(torch.zeros(self.n_heads))
        # Q24 Head-mix. Q ← Q + Q @ M (M=I init) pre-attention.
        # We store M − I so init gives M_eff=I. This is cleaner
        # than storing M and skipping the residual at init.
        self.use_q_head_mix = use_q_head_mix
        if use_q_head_mix:
            self.q_head_mix = nn.Parameter(
                torch.zeros(self.n_heads, self.n_heads)
            )
        # Q25 Time-conv. 1D conv k=3 over position axis, zero-init.
        # Depthwise: each d_k channel has its own 1-in/1-out conv.
        self.use_q_time_conv = use_q_time_conv
        if use_q_time_conv:
            # weight shape [out_channels, in_channels/groups, kernel]
            # = [D, 1, 3] with groups=D below.
            self.q_time_conv_w = nn.Parameter(
                torch.zeros(self.d_k, 1, 3)
            )
            self.q_time_conv_b = nn.Parameter(torch.zeros(self.d_k))
        # Q26 EMA-smooth over position. Q ← α·Q + (1−α)·Q_prev.
        # We store q_ema_alpha as a free parameter, sigmoid-mapped so
        # α ∈ (0,1). q_ema_alpha=0 → α=0.5 (mid EMA). The plan
        # calls for "α=1" (no smoothing) at step-0. We use a
        # learnable scalar but pick init so α ≈ 1: init = log(100)
        # so sigmoid(4.6) ≈ 0.99. Stored as raw scalar; mapped in forward.
        self.use_q_ema_smooth = use_q_ema_smooth
        if use_q_ema_smooth:
            self.q_ema_logit = nn.Parameter(
                torch.tensor(4.6)  # sigmoid(4.6) ≈ 0.99
            )
        # Q27 Feature-map attention. phi is a small learnable MLP.
        # NOT identity-init — needs its own control (see plan.md).
        self.use_q_feature_map = use_q_feature_map
        if use_q_feature_map:
            h = max(8, int(q_feature_map_hidden))
            self.q_fm_phi = nn.Sequential(
                nn.Linear(self.d_k, h, bias=False),
                nn.GELU(),
                nn.Linear(h, self.d_k, bias=False),
            )
        # Q28 Per-token RoPE. Each token's θ via a small MLP on x.
        self.use_q_per_token_rope = use_q_per_token_rope
        if use_q_per_token_rope:
            h = max(8, int(q_per_token_rope_hidden))
            # output: log-frequency scale per head (init 0 → θ=base).
            self.q_ptr_mlp = nn.Linear(d_model, self.n_heads, bias=False)
            nn.init.zeros_(self.q_ptr_mlp.weight)
        # Q29 Noise reg. Add N(0, σ²) to Q during training only.
        # σ stored as softplus parameter (always ≥ 0). init -10 →
        # softplus(-10) ≈ 4.5e-5, basically zero noise.
        self.use_q_noise_reg = use_q_noise_reg
        if use_q_noise_reg:
            self.q_noise_log = nn.Parameter(torch.tensor(-10.0))
        # O1 TalkingHeadsOut (docs/research/attention_output/plan.md,
        # Batch 1). Cross-head mix on the *post-softmax* attention
        # output (sibling of Q5 talking_heads_q, but on the output
        # side of softmax). M init I → einsum is a no-op at step 0.
        self.use_talking_heads_out = use_talking_heads_out
        if use_talking_heads_out:
            self.talking_heads_out_M = nn.Parameter(
                torch.eye(self.n_heads, self.n_heads)
            )
        self._init_output_op(out_op)

    def _init_output_op(self, out_op: str):
        """Build params for the selected post-softmax output lever (O-family).
        attn_out is [B, H, T, D]. Every lever is identity-init unless its name
        is in the 'own control' set (saturating/normalizing ops). See
        docs/research/attention_output/plan.md."""
        self.out_op = out_op or ""
        H, D = self.n_heads, self.d_k
        if out_op == "head_gate":            # O2: per-head gain, init 1
            self.o_head_gate = nn.Parameter(torch.ones(H))
        elif out_op == "head_gate_reparam":  # O2': *(1+g), g=0
            self.o_head_gate = nn.Parameter(torch.zeros(H))
        elif out_op == "head_gate_sigmoid":  # bounded gate 2σ(s), s=0→1
            self.o_head_gate = nn.Parameter(torch.zeros(H))
        elif out_op == "head_gate_softplus": # g=softplus(w)≈1
            w0 = float(torch.log(torch.expm1(torch.tensor(1.0))))
            self.o_head_gate = nn.Parameter(torch.full((H,), w0))
        elif out_op == "head_gate_clamp":    # g=clamp(g,0,2), init 1
            self.o_head_gate = nn.Parameter(torch.ones(H))
        elif out_op == "head_temp":          # O3': divide by τ_h, init 1
            self.o_head_temp = nn.Parameter(torch.ones(H))
        elif out_op == "per_hd_gain":        # gain over (H,D), init 1
            self.o_hd_gain = nn.Parameter(torch.ones(H, D))
        elif out_op == "out_layerscale":     # per-channel D gain, init 1
            self.o_chan_gain = nn.Parameter(torch.ones(D))
        elif out_op == "out_scale":          # single global scalar, init 1
            self.o_scale = nn.Parameter(torch.ones(1))
        elif out_op == "out_bias":           # O6: per-head bias, init 0
            self.o_head_bias = nn.Parameter(torch.zeros(H, 1))
        elif out_op == "out_bias_channel":   # per-(H,D) bias, init 0
            self.o_hd_bias = nn.Parameter(torch.zeros(H, D))
        elif out_op == "head_affine":        # a_h·out + b_h, a=1,b=0
            self.o_head_gate = nn.Parameter(torch.ones(H))
            self.o_head_bias = nn.Parameter(torch.zeros(H, 1))
        elif out_op == "per_hd_affine":      # G(H,D)·out + b(H,D)
            self.o_hd_gain = nn.Parameter(torch.ones(H, D))
            self.o_hd_bias = nn.Parameter(torch.zeros(H, D))
        elif out_op == "headmix_reparam":    # M=I+Δ cross-head, Δ=0
            self.o_headmix = nn.Parameter(torch.zeros(H, H))
        elif out_op == "headmix_lowrank1":   # M=I+u vᵀ, u,v=0
            self.o_mix_u = nn.Parameter(torch.zeros(H))
            self.o_mix_v = nn.Parameter(torch.zeros(H))
        elif out_op == "headmix_lowrank2":   # M=I+U Vᵀ rank 2, U,V=0
            self.o_mix_U = nn.Parameter(torch.zeros(H, 2))
            self.o_mix_V = nn.Parameter(torch.zeros(H, 2))
        elif out_op == "out_tanh":           # tanh(α·out), α=1 (own control)
            self.o_alpha = nn.Parameter(torch.ones(1))
        elif out_op == "out_rms":            # rms over D + gain(H,D) (own control)
            self.o_hd_gain = nn.Parameter(torch.ones(H, D))
        elif out_op == "out_l2norm":         # unit per head + gain(H) (own control)
            self.o_head_gate = nn.Parameter(torch.ones(H))
        # parameter-free own-control ops need no params:
        #   out_softplus, out_gelu, out_swish, out_signed_sqrt,
        #   out_softcap30, out_clamp10, out_center,
        #   out_dropout10/20, head_dropout10

    def _apply_output_op(self, attn_output):
        """Apply the selected O-family op to attn_output [B, H, T, D]."""
        op = getattr(self, "out_op", "")
        if not op:
            return attn_output
        a = attn_output
        if op == "head_gate":
            return a * self.o_head_gate.view(1, -1, 1, 1)
        if op == "head_gate_reparam":
            return a * (1.0 + self.o_head_gate.view(1, -1, 1, 1))
        if op == "head_gate_sigmoid":
            return a * (2.0 * torch.sigmoid(self.o_head_gate)).view(1, -1, 1, 1)
        if op == "head_gate_softplus":
            return a * F.softplus(self.o_head_gate).view(1, -1, 1, 1)
        if op == "head_gate_clamp":
            return a * self.o_head_gate.clamp(0.0, 2.0).view(1, -1, 1, 1)
        if op == "head_temp":
            return a / self.o_head_temp.view(1, -1, 1, 1)
        if op == "per_hd_gain":
            return a * self.o_hd_gain.view(1, self.n_heads, 1, self.d_k)
        if op == "out_layerscale":
            return a * self.o_chan_gain.view(1, 1, 1, -1)
        if op == "out_scale":
            return a * self.o_scale
        if op == "out_bias":
            return a + self.o_head_bias.view(1, self.n_heads, 1, 1)
        if op == "out_bias_channel":
            return a + self.o_hd_bias.view(1, self.n_heads, 1, self.d_k)
        if op == "head_affine":
            return a * self.o_head_gate.view(1, -1, 1, 1) + self.o_head_bias.view(1, self.n_heads, 1, 1)
        if op == "per_hd_affine":
            g = self.o_hd_gain.view(1, self.n_heads, 1, self.d_k)
            b = self.o_hd_bias.view(1, self.n_heads, 1, self.d_k)
            return a * g + b
        if op == "headmix_reparam":
            M = torch.eye(self.n_heads, device=a.device, dtype=a.dtype) + self.o_headmix
            return torch.einsum("bhtd,hH->bHtd", a, M)
        if op == "headmix_lowrank1":
            M = torch.eye(self.n_heads, device=a.device, dtype=a.dtype) + torch.outer(self.o_mix_u, self.o_mix_v)
            return torch.einsum("bhtd,hH->bHtd", a, M)
        if op == "headmix_lowrank2":
            M = torch.eye(self.n_heads, device=a.device, dtype=a.dtype) + self.o_mix_U @ self.o_mix_V.t()
            return torch.einsum("bhtd,hH->bHtd", a, M)
        if op == "out_tanh":
            return torch.tanh(self.o_alpha * a)
        if op == "out_softplus":
            return F.softplus(a)
        if op == "out_gelu":
            return F.gelu(a)
        if op == "out_swish":
            return F.silu(a)
        if op == "out_signed_sqrt":
            return torch.sign(a) * torch.sqrt(a.abs() + 1e-6)
        if op == "out_softcap30":
            return 30.0 * torch.tanh(a / 30.0)
        if op == "out_clamp10":
            return a.clamp(-10.0, 10.0)
        if op == "out_center":
            return a - a.mean(dim=-1, keepdim=True)
        if op == "out_rms":
            rms = torch.sqrt(a.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            return (a / rms) * self.o_hd_gain.view(1, self.n_heads, 1, self.d_k)
        if op == "out_l2norm":
            n = a.norm(dim=-1, keepdim=True) + 1e-6
            return (a / n) * self.o_head_gate.view(1, -1, 1, 1)
        if op == "out_dropout10":
            return F.dropout(a, p=0.1, training=self.training)
        if op == "out_dropout20":
            return F.dropout(a, p=0.2, training=self.training)
        if op == "head_dropout10":
            if not self.training:
                return a
            keep = (torch.rand(a.size(0), self.n_heads, 1, 1, device=a.device) > 0.1).to(a.dtype)
            return a * keep / 0.9
        return a

    def _manual_rope(self, Q, K, seq_len):
        """Apply RoPE with per-head / per-token / partial support.

        Used by Q6 (per-head base), Q7 (partial rotary), and Q28
        (per-token rope). Q, K are in [B, H, T, D] layout (the
        layout used in the manual attention branch). Returns the
        same [B, H, T, D] layout.
        """
        device = Q.device
        d_k = self.d_k
        # Build frequency spectrum for the GLOBAL rope_base.
        idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.rope_base ** (idx / d_k))  # [d_k/2]
        # Per-head frequency multiplier.
        if self.use_per_head_rope_base:
            head_scale = torch.exp(self.per_head_rope_log)  # [H]
        else:
            head_scale = torch.ones(self.n_heads, device=device)
        # Per-token frequency multiplier (Q28). Computed earlier in
        # [B, T, H]; transpose to [B, H, T] for our layout.
        if self._per_token_rope_log is not None:
            tok_scale = torch.exp(self._per_token_rope_log).permute(0, 2, 1)
        else:
            tok_scale = None
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Build freqs in [B, H, T, d_k/2].
        # outer: position t × inv_freq → [T, d_k/2]
        # multiply by head_scale[h] per head → [H, T, d_k/2]
        # then expand to B.
        freqs = t[None, None, :, None] * inv_freq[None, None, None, :]  # [1, 1, T, d_k/2]
        freqs = freqs * head_scale[None, :, None, None]  # [1, H, T, d_k/2]
        freqs = freqs.expand(Q.size(0), -1, -1, -1).contiguous()
        if tok_scale is not None:
            # tok_scale: [B, H, T]. Multiply along T.
            freqs = freqs * tok_scale.unsqueeze(-1)
        # cos/sin shape [B, H, T, d_k/2] → interleave to [B, H, T, d_k].
        cos_h = freqs.cos()
        sin_h = freqs.sin()
        cos = torch.repeat_interleave(cos_h, 2, dim=-1)
        sin = torch.repeat_interleave(sin_h, 2, dim=-1)
        # Partial rotary: zero out the unrotated dims.
        if self.partial_rotary_p < 1.0:
            keep = int(self.partial_rotary_p * d_k)
            mask = torch.zeros(d_k, device=device)
            mask[:keep] = 1.0
            cos = cos * mask
            sin = sin * mask
        # Apply RoPE: x_rot = x * cos + rotate_half(x) * sin
        def rotate_half(x):
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            return torch.stack([-x2, x1], dim=-1).reshape(*x.shape)
        Qr = Q * cos + rotate_half(Q) * sin
        Kr = K * cos + rotate_half(K) * sin
        return Qr, Kr

    def forward(self, x, ve=None, gate_x=None, v_residual=None, deberta_relpos=None, shared_kv=None):
        batch_size, seq_len = x.size(0), x.size(1)
        # 013 — CoPE replaces RoPE, so the post-RoPE norm has no rotary
        # to post-norm. Reject the misconfiguration loudly so the
        # runner doesn't accidentally launch it.
        assert not (self.use_cope and self.use_qk_norm_post_rope), (
            "use_cope=True is mutually exclusive with use_qk_norm_post_rope=True "
            "(CoPE replaces RoPE; the post-RoPE norm has nothing to act on)."
        )
        # 129 — YOCO: when the flag is on, the MHA must be given a
        # shared_kv tuple. Reject the misconfiguration loudly so the
        # runner doesn't accidentally launch it without plumbing.
        if self.use_shared_kv:
            assert shared_kv is not None and len(shared_kv) == 2, (
                "use_shared_kv=True requires shared_kv=(K_g, V_g) kwarg "
                "passed by YOCOLlamaBlock.forward"
            )

        # ============ MERGED QKV PROJECTION ============
        # Single matmul instead of 3 separate projections
        # 129 — YOCO upper half: when `use_shared_kv=True`, we only need
        # the Q slice of the merged qkvo_proj. The K, V projections are
        # SKIPPED — shared K_g, V_g are supplied via `shared_kv` and
        # used directly below. Q still goes through q_norm + RoPE per
        # the standard path.
        if self.use_shared_kv:
            Q = F.linear(x, self.qkvo_proj[:self.q_size])
            K, V = shared_kv
        else:
            qkv = F.linear(x, self.qkvo_proj[:self.qkv_size])

            # Split the result into Q, K, V
            # #72 Tied QK (PaLM): Q and K share the same W matrix. Use a
            # separate qk_proj parameter; the Q/K slices of qkvo_proj are
            # unused in this mode. V is still from its qkvo_proj slice.
            # #73 MLA: K, V come from a low-rank latent. The latent is
            # computed once per layer (down-project input), then
            # up-projected per head to K, V.
            if self.use_tied_qk:
                qk = F.linear(x, self.qk_proj)
                Q, K = qk.split([self.q_size, self.kv_size], dim=-1)
                V = F.linear(x, self.qkvo_proj[self.qkv_size - self.kv_size:self.qkv_size])
            elif self.use_mla:
                latent = F.linear(x, self.mla_dkv)  # [B, T, mla_latent_dim]
                K = F.linear(latent, self.mla_uk)    # [B, T, kv_size]
                V = F.linear(latent, self.mla_uv)    # [B, T, kv_size]
                Q = F.linear(x, self.qkvo_proj[:self.q_size])
            else:
                Q, K, V = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # ================================================

        # 134 — Mega EMA on V (Ma et al. 2022, arXiv:2209.10655).
        # The V stream is concatenated with `V_ema = W_V @ u` where
        # `u_t = β·u_{t-1} + (1-β)·x_t` is a causal EMA over the input
        # residual stream `x`. β ∈ [0, 1] per-channel (parametrized via
        # sigmoid so it stays bounded during training); raw init 0 ⇒
        # β=0.5 at step 0 (midpoint between paper's β=0 and β=1
        # extremes). The EMA convolution is implemented as a depthwise
        # causal `conv1d` over the T axis with kernel `(1-β)·β^k` —
        # O(T²) flops per layer (≈4M flops at tiny1m3m's T=2048, ≤1%
        # of the d_model²·T FFN cost). The concat doubles the V stream
        # from `[B, T, kv_size]` to `[B, T, 2·kv_size]`. The standard
        # head reshape then sees 2·n_kv_heads "heads" of width d_k
        # (asserted at construction to equal n_heads at tiny1m3m), so
        # SDPA runs unchanged and the O projection reads `[B, T, 2·kv_size
        # = d_model]` → `[B, T, d_model]`. NOT byte-identical to baseline
        # at step 0 — the EMA is non-trivially smoothed at β=0.5 and
        # the concat doubles V. The lever is explicitly NOT an identity
        # trick; per the idea, the design is "β=0 collapse to gated
        # attention (closed 024)" and "β=1 collapse to constant EMA";
        # β=0.5 is the natural midpoint.
        if self.use_mega:
            # β ∈ [0, 1] per-channel, bounded.
            beta = torch.sigmoid(self.mega_beta_raw)  # [d_model]
            # EMA source: either the input residual `x` (paper form)
            # or the projected V_raw. Both are d_model-shaped (V_raw
            # is reshaped as `[..., 2·kv_size]`; flatten via padding is
            # not free, so we use x as the EMA source by default).
            ema_src = x if self.mega_use_input else V
            # Kernel: kernel[d, k] = (1-β[d]) · β[d]^k for k ≥ 0.
            arange = torch.arange(seq_len, device=x.device, dtype=x.dtype)
            # [d_model, 1, T]
            kernel = (1.0 - beta).view(-1, 1, 1) * beta.view(-1, 1, 1).pow(
                arange.view(1, 1, -1)
            )
            # Causal depthwise conv1d: pad T-1 on the left so kernel
            # index k aligns with source position (T-k) at output 0.
            src_perm = ema_src.transpose(1, 2)  # [B, d_model, T]
            padded = F.pad(src_perm, (seq_len - 1, 0))
            u_perm = F.conv1d(padded, kernel, groups=self.d_model)  # [B, d_model, T]
            u = u_perm.transpose(1, 2)  # [B, T, d_model]
            # Project u through the V slice of qkvo_proj (no new params).
            W_V = self.qkvo_proj[self.qkv_size - self.kv_size:self.qkv_size]
            V_ema = F.linear(u, W_V)  # [B, T, kv_size]
            # Concatenate V_raw and V_ema to get V_mega of shape
            # `[B, T, 2·kv_size]`. The head reshape below treats this
            # as 2·n_kv_heads heads (asserted == n_heads at construction).
            V = torch.cat([V, V_ema], dim=-1)

        # ============================================================================
        # Query-tweaks (Batch 1-3, 5-6): x-dependent Q modifications.
        # Applied to the FLAT Q (shape [B, T, q_size]) before the per-head
        # reshape. Each one has a flag — defaults are all identity/zero-init
        # so step-0 == baseline (with the documented exceptions).
        # ============================================================================
        # Q21 Low-rank refine. Q += (W1·x) @ W2; both zero-init.
        if self.use_q_lowrank_refine:
            Q = Q + F.linear(F.linear(x, self.q_refine_W1), self.q_refine_W2)
        # Q8 Q-expansion: project Q to 2·q_size, mean of two reads. The
        # 2nd read is via the learned q_expand projection (zero-init) so
        # step-0 == baseline. The first read uses the existing Q.
        if self.use_q_expansion:
            q2 = F.linear(x, self.q_expand)  # [B, T, q_size], zero-init
            # We can't "mean two reads" without changing the head shape.
            # Practical implementation: 2nd head slot per head, mean over
            # the slot dim after reshape. Simpler: just add the 2nd
            # projection back into the q_size (zero-init => step 0 baseline).
            Q = Q + q2  # zero-init means baseline at step 0
        # Q2 Token-conditioned per-head temperature. Q is flat here;
        # we need per-head projection x·w_h. Defer to per-head block
        # below (after reshape) — see Q2 post-reshape.
        # Q25 Time-conv and Q29 noise reg also need the per-head reshape;
        # deferred to the post-reshape block.

        # #29 value embeddings: add the projected token embedding to the values
        # (before head reshape). Zero-inited projection => exact baseline at step 0.
        if self.use_value_embed and ve is not None:
            V = V + F.linear(ve, self.value_embed_proj)
        # #45 deep value embeddings: 2-layer non-linear V projection.
        # V += GELU(ve @ W1) @ W2. Both W1 and W2 are zero-init so step 0
        # is exact baseline. The GELU has a dead-zone at 0 so the
        # gradient flows through W2 first (Muon), then W1.
        if self.use_deep_value_embed and ve is not None:
            v_hidden = F.gelu(F.linear(ve, self.deep_value_embed_W1))
            V = V + F.linear(v_hidden, self.deep_value_embed_W2)
        # #30 query embeddings: same trick, on Q.
        if self.use_query_embed and ve is not None:
            Q = Q + F.linear(ve, self.query_embed_proj)
        # #31 key embeddings: same trick, on K. (K then goes through RoPE
        # downstream, so this term is positionally rotated — different
        # operating point from V.)
        if self.use_key_embed and ve is not None:
            K = K + F.linear(ve, self.key_embed_proj)

        # Reshape to multi-head format
        # 134 — Mega: V_mega has shape `[B, T, 2·kv_size]` (concat of
        # V_raw + V_ema). We treat the doubled V stream as 2·n_kv_heads
        # heads; the construction-time assert guarantees this equals
        # n_heads at tiny1m3m. K is unchanged (no concat on the K side
        # — the Mega idea's primary lever is the V-side smoothing, and
        # adding K doubles the head count to 3·n_kv_heads which breaks
        # the GQA bookkeeping). Q is also unchanged.
        V_n_kv_heads = (2 * self.n_kv_heads) if self.use_mega else self.n_kv_heads
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, V_n_kv_heads, self.d_k)
        
        # Apply RoPE
        # #49 QK-norm-post-RoPE: by default we apply RMSNorm to Q,K BEFORE
        # RoPE (the pre-RoPE norm). The modded-nanogpt variant applies the
        # norm AFTER RoPE. The two are mathematically different — post-RoPE
        # norm constrains the post-RoPE Q,K magnitudes per head, which can
        # help with attention score stability at scale.
        # #53 NoPE: when use_nope is set, skip the rotary call entirely.
        # RMSNorm still runs (it's a Q/K magnitude stabilizer, separate
        # from position), but the rotation is bypassed.
        # 013 — CoPE: when use_cope is set, also skip the rotary call
        # (CoPE replaces RoPE). The Q/K RMSNorm still runs (same
        # magnitude-stabilizer role). The CoPE bias is added to the
        # attention scores in the manual branch below.
        if self.use_nope or self.use_cope:
            # RMSNorm still runs (it's a Q/K magnitude stabilizer,
            # separate concern from position), but the rotation is
            # bypassed.
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        elif self.use_qk_norm_post_rope:
            Q = self.q_norm(self.rotary(Q))
            K = self.k_norm(self.rotary(K))
        else:
            Q = self.rotary(self.q_norm(Q))
            K = self.rotary(self.k_norm(K))
        # #37 per-head Q-gain: multiply Q by (1 + q_gain) per head after
        # RoPE. Zero-init, so step 0 == baseline.
        if self.use_q_gain:
            Q = Q * (1.0 + self.q_gain.view(1, 1, self.n_heads, 1))
        # #42 per-head K-gain: symmetric to Q-gain. Multiplies K after
        # RoPE. Zero-init baseline. Applied AFTER repeat_interleave so
        # the per-head scalar matches the final head count (n_heads).
        # Repeat K/V for GQA if needed
        # 134 — Mega: when on, V_n_kv_heads = 2·n_kv_heads == n_heads
        # (asserted at construction), so no repeat_interleave on V.
        # K is unchanged (still n_kv_heads), so K still gets the GQA
        # repeat to expand to n_heads. After both repeats, K and V
        # are both [B, n_heads, T, d_k] for SDPA.
        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            if not self.use_mega:
                V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)
        if self.use_k_gain:
            K = K * (1.0 + self.k_gain.view(1, 1, self.n_heads, 1))

        # ============================================================================
        # Query-tweaks post-RoPE Q modifications. Each is gated by its
        # own flag; all are identity/zero-init at step-0 unless noted.
        # Q is in [B, T, H, D] layout here. We apply the cheap vector
        # mods in-place; score-side mods (Q1, Q3, Q4, Q5, Q9, Q10)
        # are handled in the manual-attention branch below.
        # ============================================================================
        # Q2 Token-conditioned per-head temperature. Q *= (1 + tanh(x·w_h)).
        # x has shape [B, T, d_model]; w_h has shape [H, d_model].
        # gate = (x @ w_h.T) has shape [B, T, H]. Reshape to broadcast.
        if self.use_q_temp_token:
            gate = torch.tanh(torch.einsum("btd,hd->bth", x, self.q_temp_w))
            Q = Q * (1.0 + gate.unsqueeze(-1))
        # Q17 Per-head bias. Q += b_h (per-head×channel).
        if self.use_q_per_head_bias:
            Q = Q + self.q_per_head_bias.view(1, 1, self.n_heads, self.d_k)
        # Q18 Per-channel gain. Q *= g_d.
        if self.use_q_per_channel_gain:
            Q = Q * self.q_per_channel_gain.view(1, 1, 1, self.d_k)
        # Q19 Head×channel gain. Q *= g_hd.
        if self.use_q_hd_gain:
            Q = Q * self.q_hd_gain.view(1, 1, self.n_heads, self.d_k)
        # Q22 LayerScale on Q. Q *= (1 + ls_d).
        if self.use_q_layerscale:
            Q = Q * (1.0 + self.q_layerscale.view(1, 1, 1, self.d_k))
        # Q23 Softplus gain. Q *= softplus(g_h) per head. Note: NOT
        # identity at step-0 (softplus(0) = ln 2 ≈ 0.693). The score
        # function starts scaled by 0.693 and learns back.
        if self.use_q_softplus_gain:
            Q = Q * torch.nn.functional.softplus(
                self.q_softplus_g
            ).view(1, 1, self.n_heads, 1)
        # Q20 Norm-gate. g_h = 1 + (σ(a·‖x‖+b) - 0.5) per head. The
        # `+ 0.5` would make step-0 = 1.0; with a_h=0,b_h=0 we get
        # 1 + (0.5 - 0.5) = 1.0 at step 0. Identity-init.
        if self.use_q_norm_gate:
            x_norm = x.norm(dim=-1)  # [B, T]
            gate_arg = (
                torch.einsum("bth,bth->bth",
                             x_norm.unsqueeze(-1).expand(-1, -1, self.n_heads),
                             self.q_norm_gate_a.view(1, 1, -1))
                + self.q_norm_gate_b.view(1, 1, -1)
            )
            gate = 1.0 + (torch.sigmoid(gate_arg) - 0.5)
            Q = Q * gate.unsqueeze(-1)
        # Q24 Head-mix. Q ← Q + Q @ M (M stored as M−I; init 0).
        # Operates on heads, broadcast over [T, d_k].
        if self.use_q_head_mix:
            # Q: [B, T, H, D]. M: [H, H]. Result: [B, T, H, D] = Q @ M.
            Q = Q + torch.einsum("bthd,he->bted", Q, self.q_head_mix)
        # Q25 Time-conv. 1D conv k=3 over position axis, zero-init.
        # Depthwise over (B*H) batches, d_k groups.
        if self.use_q_time_conv:
            # Q: [B, T, H, D] -> [B*H, D, T] for conv1d over T.
            qc = Q.permute(0, 2, 3, 1).reshape(-1, self.d_k, seq_len)
            qc = qc + F.conv1d(
                qc, self.q_time_conv_w, self.q_time_conv_b,
                padding=1, groups=self.d_k,
            )
            Q = qc.reshape(batch_size, self.n_heads, self.d_k, seq_len).permute(0, 3, 1, 2)
        # Q26 EMA-smooth over position. Q ← α·Q + (1−α)·Q_{t-1}.
        if self.use_q_ema_smooth:
            alpha = torch.sigmoid(self.q_ema_logit)  # ~0.99 at init
            shifted = torch.cat([Q[:, :1], Q[:, :-1]], dim=1)
            Q = alpha * Q + (1.0 - alpha) * shifted
        # Q28 Per-token RoPE. Per-head θ_t via small MLP on x; init 0
        # so θ_t = base at step 0. This is a SCALE on the rotary freq
        # per head, per token. We use the per-head RoPE base = base *
        # exp(log_scale_t). Implementation: re-compute rotary for the
        # whole sequence with per-token freqs. (This breaks the SDPA
        # fast path; we mark this as a "manual path required" lever
        # and apply RoPE manually below.)
        if self.use_q_per_token_rope:
            self._per_token_rope_log = self.q_ptr_mlp(x)  # [B, T, H]
        else:
            self._per_token_rope_log = None
        # Q29 Noise reg. Q += N(0, σ²) training-only. softplus(-10) ≈ 4.5e-5.
        if self.use_q_noise_reg and self.training:
            sigma = torch.nn.functional.softplus(self.q_noise_log)
            Q = Q + torch.randn_like(Q) * sigma

        # Transpose for attention
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # 147 — DropKey (Xu et al. 2022, arXiv:2207.01058). Per-head,
        # per-token Bernoulli mask on K applied AFTER the [B, T, H, D]
        # → [B, H, T, D] transpose (so K and the mask shape align) and
        # AFTER GQA repeat_interleave (so the mask broadcasts cleanly
        # across the GQA-replicated K). The mask `M ~ Bernoulli(1-p)`
        # has shape `[B, n_heads, T, 1]` and is applied as
        # `K ← K * M / (1-p)` (inverted-dropout rescale). At eval
        # (`self.training == False`) the mask is identity, so the
        # forward graph is bit-identical to the no-DropKey baseline.
        # When `use_drop_key=False` (default), the branch is never
        # taken. With `drop_key_rate=0.0`, the mask is all-ones and
        # `K = K * 1 / 1 = K` — also bit-identical. See
        # `autoresearch/ideas/147-dropkey/idea.md`.
        if self.use_drop_key and self.training and self.drop_key_rate > 0.0:
            p = self.drop_key_rate
            keep_prob = 1.0 - p
            # Per-(batch, head, token) coin; broadcast across d_k.
            key_mask = torch.empty(
                batch_size, self.n_heads, seq_len, 1,
                device=K.device, dtype=K.dtype,
            ).bernoulli_(keep_prob)
            K = K * key_mask / keep_prob

        # 021 — Value Residual: stash post-transpose V on layer 0
        # (`v_residual is None`); blend `(1-λ)·V + λ·V_1` on layer
        # l > 0. λ=0 init ⇒ `V_l = V_l` bit-identical to baseline at
        # step 0 (within fp32 rounding noise of one extra multiply-
        # add). `.detach()` so gradients don't flow back into layer-0
        # W_V from the layer-l blend. Both V and `v_residual` are
        # shape `[B, n_heads, T, d_k]` at this site (post-GQA
        # repeat_interleave + post-transpose), invariant to
        # `n_kv_heads`. Site is pre-v_norm (idea.md:16-27 spec).
        if self.use_value_residual:
            if v_residual is None:
                self._v_residual = V.detach()
            else:
                V = (1.0 - self.lambda_v) * V + self.lambda_v * v_residual

        # #92 Robust V-norm: normalize the value vectors per head before they
        # are mixed by attention (last dim = d_k).
        if self.use_v_norm:
            V = self.v_norm(V)
        if self.use_value_channel_gate:
            V = V * (1.0 + self.value_channel_gate.view(1, self.n_heads, 1, self.d_k))
        # 109 — KDA channel gate: per-(head, channel) bounded `2·σ(g)`
        # gain on V, applied before the AV product. Sits at the same V
        # site as the closed `use_value_channel_gate` (which uses the
        # unbounded `1+g` form), but the bounded parametrization
        # `(0, 2)` is the difference. Zero-init ⇒ `2·σ(0) = 1.0`
        # exactly ⇒ V unchanged at step 0. Composes with v_norm and
        # with the closed unbounded V-gate (multiplicative on the
        # same tensor); both can be on simultaneously without
        # interference. KDA's per-channel diagonal decay, ported to
        # the softmax-attention V stream.
        if self.use_kda_channel_gate:
            V = V * (2.0 * torch.sigmoid(self.kda_channel_gate)).view(1, self.n_heads, 1, self.d_k)

        # Compute attention
        # #51 sliding-window: when enabled, use a [T, T] causal-local
        # boolean mask instead of SDPA's `is_causal=True` fast path.
        # The mask is broadcast across batch and head dims.
        # ---- Query-tweaks: manual-attention branch for score-side mods ----
        # When any of {alibi, cosine, qk_bilinear, talking_heads_q,
        # decoupled_content_pos, antisym_qk, q_feature_map} is on, we
        # need a manual path (Q@K.T + bias → mask → softmax → @V).
        # We also use this path for {per_head_rope_base, partial_rotary,
        # per_token_rope} since the standard Rotary module doesn't support
        # per-head / per-token frequencies.
        # The O1 talking_heads_out lever (docs/research/attention_output/)
        # lives on the OUTPUT side of softmax (post-matmul(attn_w, V)),
        # so it also forces the manual path.
        if self.use_fire_pe:
            # 009 FIRE PE — drop-in for RoPE. Manual path: scores
            # = Q K^T / √d_k + FIRE bias, then mask + softmax + @V.
            # x is the original input [B, T, d_model] (still in scope).
            scale = 1.0 / (float(self.d_k) ** 0.5)
            scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
            fire_bias = self.fire_bias(x)  # [B, H, T, T]
            scores = scores + fire_bias
            # 013 — CoPE stacks on top of FIRE: add the content-aware
            # positional bias to scores (additive, just like FIRE).
            if self.use_cope:
                scores = scores + self.cope(x)
            # 025 — SSMax: per-head length-dependent temperature on
            # logits. `n = i+1` is the per-query causal key count, so
            # `log_n[t] = log(t+1)` for the t-th query. We multiply
            # AFTER the FIRE/CoPE additive bias and BEFORE the mask so
            # the temperature has its natural interpretation (it
            # rescales the per-position logit scale). The mask is
            # applied later; `log_n[0] = log(1) = 0` so the first
            # query's scores are unchanged (it attends only to itself,
            # and softmax over a single element is independent of
            # temperature).
            if self.use_ssmax:
                log_n = torch.log(
                    torch.arange(1, seq_len + 1, device=Q.device, dtype=torch.float32)
                )
                scores = scores * (
                    self.ssmax_s.view(1, self.n_heads, 1, 1)
                    * log_n.view(1, 1, seq_len, 1)
                )
            # 020 — FoX: per-head learnable causal log-decay added to
            # logits BEFORE the mask + softmax. Mathematically equivalent
            # to the multiply-after-softmax + row-renorm form sketched in
            # the paper (softmax(s) ⊙ exp(log_D) / row_sum = softmax(s +
            # log_D)) but numerically stable: softmax's max-subtraction
            # absorbs arbitrarily negative log_D, whereas the post-softmax
            # multiply underflowed once the gate trained off its identity
            # init (step ~400 NaN — see r1 evidence.md). Strictly
            # orthogonal to FIRE (additive on logits) by construction:
            # both are logit-add levers, but FIRE is position-only while
            # FoX is per-head, per-token, content-conditional.
            if self.use_fox:
                scores = scores + self.fox(x).to(scores.dtype)
            if self.use_sliding_window:
                window = self._sliding_window_mask[:seq_len, :seq_len]
            else:
                ar = torch.arange(seq_len, device=Q.device)
                window = ar[None, :] <= ar[:, None]
            scores = scores.masked_fill(
                ~window.view(1, 1, seq_len, seq_len), -1e9
            )
            # 022 — Softpick: drop-in for `torch.softmax` in the FIRE
            # branch only. The same `window` tensor is reused as the
            # mask argument so masked positions contribute zero to
            # both numerator and denominator (the bug class at
            # `idea.md:32-45`). Default off → softmax.
            if self.use_softpick:
                attn_w = softpick(scores, window.view(1, 1, seq_len, seq_len))
            else:
                attn_w = torch.softmax(scores, dim=-1)
            attn_w = F.dropout(attn_w, p=self.dropout if self.training else 0.0)
            attn_output = torch.matmul(attn_w, V)
        elif (
            self.use_alibi_bias or self.use_cosine_attn
            or self.use_qk_bilinear or self.use_talking_heads_q
            or self.use_decoupled_content_pos or self.use_antisym_qk
            or self.use_q_feature_map or self.use_per_head_rope_base
            or self.use_cope  # 013 — CoPE forces the manual attention path.
            or self.use_fox  # 020 — FoX: post-softmax multiply can't go through SDPA.
            or self.use_softpick  # 022 — Softpick: defensive fallback (swap site is the FIRE branch above).
            or self.use_ssmax  # 025 — SSMax: score-side multiply, can't go through SDPA's flash kernel.
            or self.partial_rotary_p < 1.0
            or (self._per_token_rope_log is not None)
            or self.use_talking_heads_out
        ):
            # Q, K, V are already in [B, H, T, D] from the earlier
            # transpose at line 951 — no further transpose needed.
            # Build the causal mask (and combine with SWA if needed).
            ar = torch.arange(seq_len, device=Q.device)
            causal = ar[None, :] <= ar[:, None]  # [T, T]
            if self.use_sliding_window:
                window = self._sliding_window_mask[:seq_len, :seq_len]
            else:
                window = causal
            # ---- Manual RoPE (per-head base / partial / per-token) ----
            if (
                self.use_per_head_rope_base
                or self.partial_rotary_p < 1.0
                or (self._per_token_rope_log is not None)
            ):
                Q, K = self._manual_rope(Q, K, seq_len)
            # ---- Q3 Cosine attention: L2-normalize Q, K ----
            if self.use_cosine_attn:
                Qn = Q / (Q.norm(dim=-1, keepdim=True) + 1e-6)
                Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-6)
            else:
                Qn, Kn = Q, K
            # ---- Base scores ----
            # Standard QK^T / sqrt(d_k). SDPA-style "scale" math.
            scale = 1.0 / (float(self.d_k) ** 0.5)
            scores = torch.matmul(Qn, Kn.transpose(-1, -2)) * scale
            # ---- Q3 per-head τ: multiply by τ_h (Q3 only — different
            # from Q1 alibi which is a constant prior) ----
            if self.use_cosine_attn:
                tau = self.cosine_tau.view(1, self.n_heads, 1, 1)
                scores = scores * tau
            # ---- Q4 Per-channel relevance: score = Q^T diag(d_h) K ----
            if self.use_qk_bilinear:
                # d_h has shape [H, D]. Broadcast over [B, T].
                d_h = self.qk_bilinear_d.view(1, self.n_heads, 1, self.d_k)
                # Qn * d_h: [B,H,T,D] * [1,H,1,D] → [B,H,T,D]
                Qn_d = Qn * d_h
                scores = torch.matmul(Qn_d, Kn.transpose(-1, -2)) * scale
            # ---- Q1 ALiBi bias: scores += -m_h · (i - j) per head ----
            if self.use_alibi_bias:
                diff = ar[None, :].float() - ar[:, None].float()  # [T, T]
                m = self.alibi_slope.view(1, self.n_heads, 1, 1)
                scores = scores - m * diff.view(1, 1, seq_len, seq_len)
            # ---- Q10 Antisymmetric Q·K coupling: +Q^T S K, S skew ----
            if self.use_antisym_qk:
                # Enforce skew: S = (raw - raw.T) / 2.
                S = 0.5 * (self.antisym_S - self.antisym_S.t())
                # Q, K: [B, H, T, D]. Scores: [B, H, T, T].
                # antisym: Q[b,h,t,:] @ S @ K[b,h,s,:]
                # = (Q @ S)[b,h,t,:] · K[b,h,s,:]
                QS = torch.matmul(Q, S)
                extra = torch.matmul(QS, K.transpose(-1, -2))
                scores = scores + extra
            # ---- Q9 Decoupled content + position (DeBERTa-style) ----
            if self.use_decoupled_content_pos:
                if deberta_relpos is None:
                    raise RuntimeError(
                        "use_decoupled_content_pos=True requires a shared "
                        "DeBERTaRelativePositionBias module"
                    )
                scores = scores + deberta_relpos(Qn)
            # ---- Q27 Feature-map attention: phi(Q) phi(K)^T ----
            if self.use_q_feature_map:
                Qp = self.q_fm_phi(Q)
                Kp = self.q_fm_phi(K)
                scores = torch.matmul(Qp, Kp.transpose(-1, -2)) * scale
            # ---- 013 CoPE: content-aware positional bias added to scores ----
            if self.use_cope:
                scores = scores + self.cope(x)
            # ---- 025 SSMax: per-head length-dependent temperature on
            # logits. Same recipe as the FIRE branch (after the
            # additive bias additions, before the mask). ----
            if self.use_ssmax:
                log_n = torch.log(
                    torch.arange(1, seq_len + 1, device=Q.device, dtype=torch.float32)
                )
                scores = scores * (
                    self.ssmax_s.view(1, self.n_heads, 1, 1)
                    * log_n.view(1, 1, seq_len, 1)
                )
            # 020 — FoX: per-head learnable causal log-decay added to
            # logits BEFORE the mask + softmax (paper's logit-add form,
            # numerically stable). See the FIRE-branch comment above and
            # `models/fox.py` for the identity-init derivation. Strictly
            # orthogonal to the score-side tweaks above (alibi / cosine
            # / qk_bilinear / cope / ssmax) — they modify content logits;
            # FoX adds a per-head, per-token, causal cumulative log-decay.
            if self.use_fox:
                scores = scores + self.fox(x).to(scores.dtype)
            # ---- Mask (causal / SWA) ----
            scores = scores.masked_fill(~window.view(1, 1, seq_len, seq_len), -1e9)
            # ---- Q5 Talking-heads: logit-mix across heads pre-softmax ----
            if self.use_talking_heads_q:
                # scores: [B, H, T, T]. M: [H, H]. Mix over H only.
                # out[b, h_new, t, s] = sum_h M[h_new, h] * scores[b, h, t, s]
                scores = torch.einsum(
                    "bhst,hH->bHst", scores, self.talking_heads_M
                )
            # Softmax
            attn_w = torch.softmax(scores, dim=-1)
            attn_w = F.dropout(attn_w, p=self.dropout if self.training else 0.0)
            attn_output = torch.matmul(attn_w, V)
            # O1 TalkingHeadsOut: cross-head mix on the *post-softmax*
            # output. Sibling of Q5 (which mixes *scores* pre-softmax).
            # attn_output: [B, H, T, D]. M: [H, H]. M init I → no-op
            # at step 0. See docs/research/attention_output/plan.md.
            if self.use_talking_heads_out:
                attn_output = torch.einsum(
                    "bhtd,hH->bHtd", attn_output, self.talking_heads_out_M
                )
        elif self.use_diff_attn:
            # #87 Differential Attention: two sub-attentions on the split
            # head_dim, combined as a1 - lambda * a2 (SDPA is linear in V).
            d_half = self.d_k // 2
            Q1, Q2 = Q[..., :d_half], Q[..., d_half:]
            K1, K2 = K[..., :d_half], K[..., d_half:]
            drop = self.dropout if self.training else 0.0
            if self.use_sliding_window:
                m = self._sliding_window_mask[:seq_len, :seq_len]
                a1 = F.scaled_dot_product_attention(Q1, K1, V, attn_mask=m, dropout_p=drop)
                a2 = F.scaled_dot_product_attention(Q2, K2, V, attn_mask=m, dropout_p=drop)
            else:
                a1 = F.scaled_dot_product_attention(Q1, K1, V, is_causal=True, dropout_p=drop)
                a2 = F.scaled_dot_product_attention(Q2, K2, V, is_causal=True, dropout_p=drop)
            lam = self.diff_lambda.view(1, self.n_heads, 1, 1)
            attn_output = a1 - lam * a2
            attn_output = self.diff_norm(attn_output) * (1.0 - self._diff_lambda_init)
        elif self.use_hybrid_heads:
            # #89 Hybrid heads: per-head local/global mask, one SDPA call.
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=self._hybrid_head_mask[:, :seq_len, :seq_len],
                dropout_p=self.dropout if self.training else 0.0,
            )
        elif self.use_multiscale_heads:
            # #97 Multi-scale heads: per-head graded-window mask, one SDPA call.
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=self._multiscale_mask[:, :seq_len, :seq_len],
                dropout_p=self.dropout if self.training else 0.0,
            )
        elif self.use_attn_sink:
            # #99 Attention sink: append a zero K/V slot every query may attend
            # to (softmax-off-by-one). Standard causal/SWA mask over the real
            # tokens, plus an always-on sink column.
            sink = torch.zeros(Q.size(0), Q.size(1), 1, Q.size(3), dtype=Q.dtype, device=Q.device)
            Kp = torch.cat([K, sink], dim=2)
            Vp = torch.cat([V, sink], dim=2)
            if self.use_sliding_window:
                base = self._sliding_window_mask[:seq_len, :seq_len]
            else:
                ar = torch.arange(seq_len, device=Q.device)
                base = ar[:, None] >= ar[None, :]
            sink_col = torch.ones(seq_len, 1, dtype=torch.bool, device=Q.device)
            mask = torch.cat([base, sink_col], dim=1)
            attn_output = F.scaled_dot_product_attention(
                Q, Kp, Vp, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        elif self.use_nsa_global:
            # #88 NSA-style: local window + block-compressed global branch.
            drop = self.dropout if self.training else 0.0
            if self.use_sliding_window:
                local = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=self._sliding_window_mask[:seq_len, :seq_len],
                    dropout_p=drop,
                )
            else:
                local = F.scaled_dot_product_attention(
                    Q, K, V, is_causal=True, dropout_p=drop,
                )
            Bsz = self.nsa_block
            n_blk = (seq_len + Bsz - 1) // Bsz
            pad = n_blk * Bsz - seq_len
            Kp = F.pad(K, (0, 0, 0, pad)) if pad else K
            Vp = F.pad(V, (0, 0, 0, pad)) if pad else V
            Kb = Kp.reshape(batch_size, self.n_heads, n_blk, Bsz, self.d_k).mean(dim=3)
            Vb = Vp.reshape(batch_size, self.n_heads, n_blk, Bsz, self.d_k).mean(dim=3)
            scores = torch.matmul(Q, Kb.transpose(-1, -2)) / (float(self.d_k) ** 0.5)
            pos = torch.arange(seq_len, device=Q.device)
            blk_end = (torch.arange(n_blk, device=Q.device) + 1) * Bsz
            allow = blk_end[None, :] <= (pos[:, None] + 1)  # [T, n_blk], strictly-past blocks
            scores = scores.masked_fill(~allow[None, None], -1e9)
            w = torch.softmax(scores, dim=-1)
            any_blk = allow.any(dim=-1).view(1, 1, seq_len, 1).to(w.dtype)
            glob = torch.matmul(w, Vb) * any_blk
            gate = self.nsa_gate.view(1, self.n_heads, 1, 1)
            attn_output = local + gate * glob
        elif self.use_linear_attn:
            q_phi = (F.elu(Q) + 1.0).float()
            k_phi = (F.elu(K) + 1.0).float()
            v_float = V.float()

            if self.use_sliding_window and self.attention_dilation != 1:
                scores = torch.einsum("bhtd,bhsd->bhts", q_phi, k_phi)
                mask = self._sliding_window_mask[:seq_len, :seq_len]
                scores = scores.masked_fill(~mask, 0.0)
                denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                weights = scores / denom
                attn_output = torch.einsum("bhts,bhsd->bhtd", weights, v_float)
            else:
                window = self.sliding_window_size if self.use_sliding_window else seq_len
                kv = k_phi.unsqueeze(-1) * v_float.unsqueeze(-2)
                prefix_kv = torch.cat(
                    [torch.zeros_like(kv[:, :, :1]), kv.cumsum(dim=2)],
                    dim=2,
                )
                prefix_k = torch.cat(
                    [torch.zeros_like(k_phi[:, :, :1]), k_phi.cumsum(dim=2)],
                    dim=2,
                )
                end_idx = torch.arange(1, seq_len + 1, device=Q.device)
                start_idx = (end_idx - window).clamp_min(0)
                kv_sum = prefix_kv[:, :, end_idx] - prefix_kv[:, :, start_idx]
                k_sum = prefix_k[:, :, end_idx] - prefix_k[:, :, start_idx]
                numerator = torch.einsum("bhtd,bhtde->bhte", q_phi, kv_sum)
                denom = torch.einsum("bhtd,bhtd->bht", q_phi, k_sum).clamp_min(1e-6)
                attn_output = numerator / denom.unsqueeze(-1)

            attn_output = attn_output.to(V.dtype)
        elif self.use_sliding_window:
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=self._sliding_window_mask[:seq_len, :seq_len],
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        # #107 Exclusive self-attn: remove the component of each head's
        # output that lies along the current token's value vector.
        # V is already in [B, H, T, D] after the GQA repeat_interleave
        # step above. Zero-init coefficient keeps step 0 identical.
        if self.use_exclusive_self_attn:
            v_norm_sq = V.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            v_proj = (attn_output * V).sum(dim=-1, keepdim=True) / v_norm_sq
            attn_output = attn_output - self.exclusive_self_attn.view(1, self.n_heads, 1, 1) * v_proj * V
        if self.use_attn_output_gate:
            gate = 1.0 + self.attn_output_gate.view(1, self.n_heads, 1, 1)
            attn_output = attn_output * gate
        if self.use_attn_output_channel_gate:
            attn_output = attn_output * (1.0 + self.attn_output_channel_gate.view(1, self.n_heads, 1, self.d_k))
        # 024 — Gated Attention: input-conditional per-head scalar
        # sigmoid gate on `o_h = A_h V_h` (Qiu et al. 2025). Applied
        # post-AV, pre-merge. Composes cleanly with `use_attn_output_gate`
        # (ReZero gain) since the two multiply through. Gate input is
        # the sublayer input residual (pre-LN) — see
        # `TransformerBlock.forward` for the `gate_x` plumbing. When
        # `gate_x` is None, fall back to the MHA's primary input `x`
        # (post-norm or call sites that don't plumb the raw residual).
        # W=0, b=0 init → 2·σ(0) = 1.0 exactly at step 0 → bit-identical
        # to the no-gate path at step 0.
        if self.use_gated_attn:
            _gx = gate_x if gate_x is not None else x
            g = 2.0 * torch.sigmoid(self.gated_attn_proj(_gx))  # [B, T, H]
            attn_output = attn_output * g.view(batch_size, seq_len, self.n_heads, 1).transpose(1, 2)
        # O-family: cheap op on the post-softmax output [B,H,T,D] (pre-merge),
        # applied for every attention branch. No-op when out_op == "".
        attn_output = self._apply_output_op(attn_output)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        
        # ============ MERGED O PROJECTION ============
        # Use the last part of qkvo_proj for output projection
        output = F.linear(attn_output, self.qkvo_proj[self.qkv_size:])
        # #33 output embeddings: add the projected token embedding to the
        # attention OUTPUT (post-O). Different operating point from V/Q/K
        # (which inject into attention inputs). ve is the raw token
        # embedding [B, T, emb_rank], projection is zero-init so step 0
        # matches the baseline.
        if self.use_output_embed and ve is not None:
            output = output + F.linear(ve, self.output_embed_proj)
        return output


class TransformerBlock(nn.Module):
    """Standard transformer block with dense feed-forward"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
        ffn_variant: str = "squared_relu",
        use_embed_residual: bool = False,
        use_attn_output_gate: bool = False,
        use_value_channel_gate: bool = False,
        use_attn_output_channel_gate: bool = False,
        use_exclusive_self_attn: bool = False,
        use_kda_channel_gate: bool = False,
        # 147 — DropKey (Xu et al. 2022, arXiv:2207.01058). Per-head,
        # per-token Bernoulli mask on K during training. Pass-through
        # to the inner MHA. See `autoresearch/ideas/147-dropkey/idea.md`.
        use_drop_key: bool = False,
        drop_key_rate: float = 0.1,
        use_talking_heads_out: bool = False,
        out_op: str = "",
        use_layerscale: bool = False,
        # 142 — LayerScale (Touvron et al. 2021, arXiv:2103.17239). Per-channel
        # learnable diagonal scale `gamma ∈ R^{d_model}` on each sublayer's
        # residual branch. Direct form `x = x + gamma * sub_block(x)` (NOT
        # the reparam `(1+γ)` form used by `use_layerscale` above). Init
        # `gamma = layer_scale_init * ones(d_model)` (default 1e-4) → at
        # step 0 the residual contribution is `1e-4 × sub_block(x)`, four
        # orders of magnitude smaller than the residual stream magnitude,
        # so the val loss at step 0 is within fp32 noise of baseline. The
        # per-channel selectivity is qualitatively different from scalar
        # ReZero (130) and whole-residual Sub-LN (017). Default off →
        # baseline path bit-identical. See
        # `autoresearch/ideas/142-layerscale/idea.md`.
        use_layer_scale: bool = False,
        layer_scale_init: float = 1e-4,
        use_value_embed: bool = False,
        value_embed_rank: int | None = None,
        use_query_embed: bool = False,
        use_key_embed: bool = False,
        use_output_embed: bool = False,
        use_q_gain: bool = False,
        use_k_gain: bool = False,
        use_deep_value_embed: bool = False,
        deep_value_embed_hidden: int | None = None,
        use_ffn_embed: bool = False,
        use_qk_norm_post_rope: bool = False,
        use_sliding_window: bool = False,
        sliding_window_size: int = 512,
        use_nope: bool = False,
        rope_base: int = 10000,
        use_layernorm: bool = False,
        use_tied_qk: bool = False,
        use_mla: bool = False,
        mla_latent_dim: int | None = None,
        attention_dilation: int = 1,
        use_post_norm: bool = False,
        use_linear_attn: bool = False,
        use_diff_attn: bool = False,
        use_nsa_global: bool = False,
        nsa_block: int = 64,
        use_hybrid_heads: bool = False,
        norm_type: str = "rmsnorm",
        qk_norm_type: str = "rmsnorm",
        v_norm_type: str = "",
        # #16 QK-Norm (Dehghani et al. 2023, ViT-22B, arXiv:2302.05442):
        # forward to MHA — see MultiHeadAttention.use_qk_layernorm for the
        # mechanism description. Default off → Q/K stay on RMSNorm,
        # baseline path is bit-identical.
        use_qk_layernorm: bool = False,
        # 029 — V-Norm (Wortsman et al. 2023, arXiv:2309.14322):
        # forward to MHA — see MultiHeadAttention.use_v_layernorm for the
        # mechanism description. Default off → V stays unnormalized,
        # baseline path is bit-identical.
        use_v_layernorm: bool = False,
        use_multiscale_heads: bool = False,
        use_parallel_block: bool = False,
        use_attn_sink: bool = False,
        # 017 — Sub-LN / Sandwich block (Wang et al. 2022, DeepNet §3.1):
        # wrap each sublayer output with a fresh `nn.LayerNorm(d_model)`
        # (γ=1, β=0 init → identity at step 0). When off, the pre-norm
        # baseline path stays bit-identical. Pre-LN stays as whatever
        # `norm_type`/`use_layernorm` selected; the post-LN is always a
        # plain `nn.LayerNorm` (residual-stream re-bounding role).
        use_sub_ln: bool = False,
        # R1 ReZero: per-sublayer scalar gate x = x + α·f(x), α=0 init.
        # Replaces the baseline residual add; lever is the *learning* of α.
        # 2 params/block (one for attention, one for FFN).
        use_re_zero: bool = False,
        # Residual-stream lever family (docs/research/residual_stream/plan.md):
        # one cheap scalar/vector knob on the residual add `x = a·x + g·f(x)`.
        # One string selects the lever; params are built in _init_resid and
        # applied via _resid_add in the pre-norm forward. n_layers feeds the
        # DeepNorm-style fixed-scale levers.
        resid_mode: str = "",
        n_layers: int = 1,
        # Query-tweaks (29 experiments). Pass-through to MHA.
        q_norm_type: str = "rmsnorm",
        use_alibi_bias: bool = False,
        use_q_temp_token: bool = False,
        use_cosine_attn: bool = False,
        use_qk_bilinear: bool = False,
        use_talking_heads_q: bool = False,
        use_per_head_rope_base: bool = False,
        partial_rotary_p: float = 1.0,
        use_q_expansion: bool = False,
        use_decoupled_content_pos: bool = False,
        use_antisym_qk: bool = False,
        use_q_per_head_bias: bool = False,
        use_q_per_channel_gain: bool = False,
        use_q_hd_gain: bool = False,
        use_q_norm_gate: bool = False,
        use_q_lowrank_refine: bool = False,
        q_lowrank_refine_rank: int = 8,
        use_q_layerscale: bool = False,
        use_q_softplus_gain: bool = False,
        use_q_head_mix: bool = False,
        use_q_time_conv: bool = False,
        use_q_ema_smooth: bool = False,
        q_ema_alpha: float = 0.0,
        use_q_feature_map: bool = False,
        q_feature_map_hidden: int = 64,
        use_q_per_token_rope: bool = False,
        q_per_token_rope_hidden: int = 32,
        use_q_noise_reg: bool = False,
        use_fire_pe: bool = False,
        fire_pe_d_phi: int = 4,
        # 024 — Gated Attention (Qiu et al. 2025, arXiv:2505.06708):
        # per-head *scalar* input-conditional sigmoid gate on the head
        # output `o_h = A_h V_h`, post-AV, pre-merge. Pass-through to MHA
        # (see the MHA `use_gated_attn` kwarg for the mechanism). Default
        # off → baseline path bit-identical. See
        # `autoresearch/ideas/024-gated-attention/plan.md`.
        use_gated_attn: bool = False,
        # 013 — CoPE: passed through to MultiHeadAttention (see note
        # at the MHA `use_cope` kwarg). Default off → baseline path
        # bit-identical. Mutually exclusive with use_qk_norm_post_rope
        # (enforced by the assert in the MHA's forward).
        use_cope: bool = False,
        # 020 — Forgetting Transformer (FoX). Passed through to
        # MultiHeadAttention (see note at the MHA `use_fox` kwarg).
        # Default off → baseline path bit-identical. Identity-init
        # clean (W_f=0, b_f=+10 → D ≈ 1 within 9% over T=2048).
        use_fox: bool = False,
        # 022 — Softpick. Passed through to MultiHeadAttention
        # (see note at the MHA `use_softpick` kwarg). Default off →
        # baseline path bit-identical.
        use_softpick: bool = False,
        # 025 — SSMax. Passed through to MultiHeadAttention (see note
        # at the MHA `use_ssmax` kwarg). Default off → baseline path
        # bit-identical.
        use_ssmax: bool = False,
        # 023 — Canon conv (Griffin/Mamba local-mixing half). One
        # causal depthwise Conv1d (kernel=3) per block on the residual
        # stream, immediately before the attention sublayer's pre-LN.
        # Single scalar output gate `g` per block init 0 → step-0
        # ≡ no-conv baseline. Pre-LN read. Default off → baseline
        # path bit-identical (the conv module is never built, the
        # forward branch is never taken). See
        # `autoresearch/ideas/023-canon-conv/plan.md`.
        use_canon_conv: bool = False,
        # 143 — ShortConv (Hyena ShortConv variant, Poli/Massaroli
        # et al. 2023, arXiv:2302.10866): one causal depthwise Conv1d
        # per block on the residual stream, immediately before the
        # attention sublayer's pre-LN (same placement as CanonConv
        # 023). Identity-init weights (center tap = 1, rest = 0) and
        # a per-block scalar output gate `g` init 0 → step-0 ≡
        # no-conv baseline. Different from CanonConv by (a) the
        # identity-init weights (vs Kaiming-uniform) and (b) the
        # parameterizable kernel size `short_conv_kernel` (3 or 4).
        # Pre-LN read. Default off → baseline path bit-identical
        # (the conv module is never built, the forward branch is
        # never taken). See
        # `autoresearch/ideas/143-shortconv/idea.md`.
        use_short_conv: bool = False,
        short_conv_kernel: int = 3,
        # 021 — Value Residual Learning. Passed through to
        # MultiHeadAttention (see the MHA `use_value_residual` kwarg
        # for the mechanism). Default off → baseline path bit-identical.
        use_value_residual: bool = False,
        # 111 — DropPath / Stochastic Depth (Huang et al. 2016,
        # arXiv:1603.09382). Per-block Bernoulli gate during training:
        # with probability `1 - p_l` skip the whole block (residual
        # update `x ← x`); with probability `p_l` keep and rescale the
        # block's residual contribution by `1/p_l`. `p_l` is a function
        # of the layer's position in the stack (passed in `forward` as
        # `layer_index`) and `drop_path_max`. The first block has
        # p_l = 1.0 (never dropped); the last block has p_l = 1 -
        # drop_path_max. The coin is shared across the batch (one
        # flip per block per step). Eval: no stochasticity, no rescale.
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/111-drop-path/idea.md`.
        use_drop_path: bool = False,
        drop_path_max: float = 0.1,
        # 131 — LayerDrop (Fan, Grave, Joulin 2019, arXiv:1904.09728,
        # ICLR 2020). Block-level stochastic depth: per-block
        # Bernoulli gate shared across the batch. With probability
        # `1 - p_l` skip the entire block (`x ← x`); with probability
        # `p_l` keep and rescale by `1/p_l` so the expected residual
        # matches baseline. `p_l` is computed from `layer_index`,
        # `n_layers`, and `layerdrop_p`/`layerdrop_schedule`. Eval has
        # no stochasticity. Distinct from DropPath (111): DropPath
        # uses a linear schedule `p_l = 1 - drop_path_max·l/(L-1)` with
        # p_0 = 1 (never drop the first block); LayerDrop is the
        # `constant` paper default `p_l = p` for all `l`, optionally
        # with a `linear`/`stochastic_depth` schedule. The
        # flag-on lever is *not* byte-identical to baseline at step 0
        # (the kept-block rescale `1/p_l` magnifies the residual); the
        # flag-off path is bit-identical. See
        # `autoresearch/ideas/131-layer-drop/idea.md`.
        use_layerdrop: bool = False,
        layerdrop_p: float = 0.2,
        layerdrop_schedule: str = "constant",
        # 117 — Soft MoE FFN replacement (Puigcerver et al. 2024).
        # When `use_soft_moe=True`, swap the standard dense FFN for
        # `SoftMoEFFN` (E parallel narrower FFNs + softmax dispatch/
        # combine). Each expert has width `d_ff / soft_moe_n_experts`
        # so total FFN params stay at the baseline budget. Default
        # off → baseline FFN path bit-identical (the `SoftMoEFFN`
        # module is never built). See `models/soft_moe.py` +
        # `autoresearch/ideas/117-soft-moe/idea.md`.
        use_soft_moe: bool = False,
        soft_moe_n_experts: int = 4,
        soft_moe_n_slots: int = 4,
        # 118 — Mixture-of-Depths (Raposo et al. 2024, arXiv:2404.02258).
        # When `use_mod=True`, wrap the block's residual update with a
        # per-token top-k router. Default off → `MoDRouter` is never
        # built, baseline forward graph is bit-identical. See
        # `models/mod_router.py` +
        # `autoresearch/ideas/118-mixture-of-depths/idea.md`.
        use_mod: bool = False,
        mod_capacity: float = 0.5,
        mod_router_hidden: int = 64,
        # 146 — Switch FFN (Fedus, Zoph, Shazeer 2022, arXiv:2101.03961):
        # when `use_switch_ffn=True`, swap the standard dense FFN for
        # `SwitchFFN` (E parallel full-width FFNs + top-1 router).
        # Default off → baseline FFN path bit-identical (the
        # `SwitchFFN` module is never built). See
        # `models/switch_ffn.py` +
        # `autoresearch/ideas/146-sparse-ffn/idea.md`.
        use_switch_ffn: bool = False,
        n_ffn_experts: int = 4,
        expert_capacity_factor: float = 1.25,
        # 145 — Expert-Choice MoE (Zhou, Lei, et al. 2022,
        # arXiv:2202.09368). INVERTED routing direction vs Switch
        # FFN: each expert picks its own top-k tokens (k = ceil(N/E))
        # instead of each token picking its top-1 expert. Load
        # balance is by construction — every expert processes
        # exactly k tokens — so NO auxiliary load-balancing loss is
        # needed (the auxiliary-loss knob is the structural
        # difference from 117-soft-moe, which uses *soft* slot
        # assignment). When `use_expert_choice_moe=True`, swap the
        # standard dense FFN for `ExpertChoiceMoE` (E parallel
        # full-width FFNs + a `nn.Linear(d_model, n_experts)` zero-
        # init router). Default off → baseline FFN path bit-
        # identical (the module is never built). See
        # `models/expert_choice_moe.py` and
        # `autoresearch/ideas/145-expert-choice/idea.md`.
        use_expert_choice_moe: bool = False,
        n_moe_experts: int = 4,
        # 129 — YOCO shared KV (Sun et al. 2024, arXiv:2405.05254).
        # When set, the inner MHA is built with `use_shared_kv=True`,
        # which makes the MHA skip its W_K, W_V slices of the merged
        # qkvo_proj and read K, V from the `shared_kv` kwarg passed
        # on `forward`. Used by `models/yoco.py:YOCOLlamaBlock` for
        # the upper-half block stack. Default off → standard path.
        # See `autoresearch/ideas/129-yoco/idea.md`.
        use_shared_kv: bool = False,
        # 134 — Mega EMA on V (Ma et al. 2022, arXiv:2209.10655).
        # Pass-through to the inner MHA. See
        # `MultiHeadAttention.use_mega` for the mechanism. Default
        # off → baseline path bit-identical. Tiny1M3M satisfies the
        # construction-time assert `2·n_kv_heads == n_heads`
        # (n_kv_heads=2, n_heads=4). See
        # `autoresearch/ideas/134-mega-ema/idea.md`.
        use_mega: bool = False,
        mega_beta: float = 0.9,
        mega_use_input: bool = True,
        # 148 — Focal Modulation Networks (Yang et al. 2022,
        # arXiv:2203.11926, NeurIPS 2022). Replaces the attention
        # sub-block with a focal modulation block (no softmax, no
        # QKᵀ). The MHA is still built (cheap) but is never called
        # when `use_focal_mod=True`. Default off → baseline path
        # bit-identical (the focal module is never built, MHA path
        # is unchanged). At step 0 the focal block's `gather` and
        # `h_proj` linears are zero-init so the modulation
        # contribution is exactly 0 ⇒ output `= x` ≡ no-op. See
        # `autoresearch/ideas/148-focal-mod/idea.md`.
        use_focal_mod: bool = False,
        focal_mod_kernels: tuple = (3, 5, 7),
    ):
        super().__init__()
        # #75 Post-norm: when set, the norm is applied AFTER the
        # residual addition instead of before. Implementation:
        # compute (norm, residual) inside the function but apply
        # the norm to (x + sublayer_out) before returning.
        self.use_post_norm = use_post_norm
        # #98 Parallel block (PaLM / GPT-J): attention and FFN both read the
        # SAME normed input and their outputs are summed into the residual,
        # instead of running sequentially. Halves the per-layer serial depth.
        self.use_parallel_block = use_parallel_block

        self.attention = MultiHeadAttention(
            d_model,
            n_heads,
            max_seq_len,
            dropout,
            n_kv_heads,
            use_attn_output_gate=use_attn_output_gate,
            use_value_channel_gate=use_value_channel_gate,
            use_attn_output_channel_gate=use_attn_output_channel_gate,
            use_exclusive_self_attn=use_exclusive_self_attn,
            use_kda_channel_gate=use_kda_channel_gate,
            # 147 — DropKey: per-head Bernoulli gate on K during training.
            use_drop_key=use_drop_key,
            drop_key_rate=drop_key_rate,
            use_talking_heads_out=use_talking_heads_out,
            out_op=out_op,
            use_value_embed=use_value_embed,
            use_query_embed=use_query_embed,
            use_key_embed=use_key_embed,
            use_output_embed=use_output_embed,
            use_q_gain=use_q_gain,
            use_k_gain=use_k_gain,
            use_deep_value_embed=use_deep_value_embed,
            deep_value_embed_hidden=deep_value_embed_hidden,
            use_qk_norm_post_rope=use_qk_norm_post_rope,
            use_sliding_window=use_sliding_window,
            sliding_window_size=sliding_window_size,
            use_nope=use_nope,
            rope_base=rope_base,
            use_fire_pe=use_fire_pe,
            fire_pe_d_phi=fire_pe_d_phi,
            use_gated_attn=use_gated_attn,
            use_cope=use_cope,
            use_fox=use_fox,
            use_softpick=use_softpick,
            use_ssmax=use_ssmax,
            use_value_residual=use_value_residual,
            # 129 — YOCO shared KV pass-through to the MHA. Default
            # off → standard path. See `autoresearch/ideas/129-yoco/idea.md`.
            use_shared_kv=use_shared_kv,
            # 134 — Mega EMA on V pass-through to the MHA. Default
            # off → baseline path bit-identical.
            use_mega=use_mega,
            mega_beta=mega_beta,
            mega_use_input=mega_use_input,
            use_tied_qk=use_tied_qk,
            use_mla=use_mla,
            mla_latent_dim=mla_latent_dim,
            attention_dilation=attention_dilation,
            use_layernorm=use_layernorm,
            use_linear_attn=use_linear_attn,
            use_diff_attn=use_diff_attn,
            use_nsa_global=use_nsa_global,
            nsa_block=nsa_block,
            use_hybrid_heads=use_hybrid_heads,
            use_multiscale_heads=use_multiscale_heads,
            use_attn_sink=use_attn_sink,
            qk_norm_type=qk_norm_type,
            v_norm_type=v_norm_type,
            # #16 QK-Norm pass-through: when set, MHA's Q/K norms are
            # LayerNorm (not RMSNorm) — see MultiHeadAttention.use_qk_layernorm.
            use_qk_layernorm=use_qk_layernorm,
            # 029 — V-Norm pass-through: when set, MHA builds a per-head
            # `nn.LayerNorm(d_k)` on V — see MultiHeadAttention.use_v_layernorm.
            use_v_layernorm=use_v_layernorm,
            # Query-tweaks pass-through.
            q_norm_type=q_norm_type,
            use_alibi_bias=use_alibi_bias,
            use_q_temp_token=use_q_temp_token,
            use_cosine_attn=use_cosine_attn,
            use_qk_bilinear=use_qk_bilinear,
            use_talking_heads_q=use_talking_heads_q,
            use_per_head_rope_base=use_per_head_rope_base,
            partial_rotary_p=partial_rotary_p,
            use_q_expansion=use_q_expansion,
            use_decoupled_content_pos=use_decoupled_content_pos,
            use_antisym_qk=use_antisym_qk,
            use_q_per_head_bias=use_q_per_head_bias,
            use_q_per_channel_gain=use_q_per_channel_gain,
            use_q_hd_gain=use_q_hd_gain,
            use_q_norm_gate=use_q_norm_gate,
            use_q_lowrank_refine=use_q_lowrank_refine,
            q_lowrank_refine_rank=q_lowrank_refine_rank,
            use_q_layerscale=use_q_layerscale,
            use_q_softplus_gain=use_q_softplus_gain,
            use_q_head_mix=use_q_head_mix,
            use_q_time_conv=use_q_time_conv,
            use_q_ema_smooth=use_q_ema_smooth,
            q_ema_alpha=q_ema_alpha,
            use_q_feature_map=use_q_feature_map,
            q_feature_map_hidden=q_feature_map_hidden,
            use_q_per_token_rope=use_q_per_token_rope,
            q_per_token_rope_hidden=q_per_token_rope_hidden,
            use_q_noise_reg=use_q_noise_reg,
            value_embed_rank=value_embed_rank,
        )
        if use_soft_moe:
            # 117 — Soft MoE: E parallel narrower FFNs + softmax
            # dispatch/combine. See `models/soft_moe.py` for the
            # mechanism. Each expert uses the same `ffn_variant` as
            # the baseline would have used (squared_relu by default).
            self.feed_forward = SoftMoEFFN(
                d_model, d_ff,
                n_experts=soft_moe_n_experts,
                n_slots=soft_moe_n_slots,
                dropout=dropout,
                ffn_variant=ffn_variant,
            )
        elif use_switch_ffn:
            # 146 — Switch FFN: E parallel full-width FFNs + top-1
            # router. See `models/switch_ffn.py` for the mechanism.
            # Each expert uses the same `ffn_variant` as the baseline
            # would have used. Router is zero-init ⇒ all tokens
            # route to expert 0 at step 0 ⇒ output = a standard FFN.
            self.feed_forward = SwitchFFN(
                d_model, d_ff,
                n_experts=n_ffn_experts,
                capacity_factor=expert_capacity_factor,
                dropout=dropout,
                ffn_variant=ffn_variant,
            )
        elif use_expert_choice_moe:
            # 145 — Expert-Choice MoE: E parallel full-width FFNs +
            # top-k-per-expert router. See
            # `models/expert_choice_moe.py` for the mechanism. Each
            # expert uses the same `ffn_variant` as the baseline.
            # Router is zero-init ⇒ all expert-token scores are 0 ⇒
            # every expert processes the same set of k tokens with
            # uniform softmax weights ⇒ output ≈ uniform mean of
            # E identically-init'd FFNs (NOT byte-identical to a
            # single FFN at step 0 — documented caveat mirroring
            # 117-soft-moe).
            self.feed_forward = ExpertChoiceMoE(
                d_model, d_ff,
                n_experts=n_moe_experts,
                dropout=dropout,
                ffn_variant=ffn_variant,
            )
        elif ffn_variant == "squared_relu":
            self.feed_forward = SquaredReLUFeedForward(d_model, d_ff, dropout)
        elif ffn_variant == "swiglu":
            self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        elif ffn_variant == "gelu":
            self.feed_forward = GELUFeedForward(d_model, d_ff, dropout)
        elif ffn_variant == "satrelu":
            self.feed_forward = SaturatingReLUFeedForward(d_model, d_ff, dropout)
        else:
            raise ValueError(f"Unknown ffn_variant: {ffn_variant}")

        # Normalization layers
        self.norm1 = make_norm(d_model, norm_type, use_layernorm)
        self.norm2 = make_norm(d_model, norm_type, use_layernorm)
        self.dropout = nn.Dropout(dropout)

        # #20 embedding residual: per-dim mix with the original token embedding x0,
        # init [m0=1, m1=0] so it starts exactly at baseline.
        self.use_embed_residual = use_embed_residual
        if use_embed_residual:
            self.resid_m0 = nn.Parameter(torch.ones(d_model))
            self.resid_m1 = nn.Parameter(torch.zeros(d_model))
        self.use_layerscale = use_layerscale
        if self.use_layerscale:
            self.attn_layerscale = nn.Parameter(torch.zeros(d_model))
            self.ffn_layerscale = nn.Parameter(torch.zeros(d_model))
        # 142 — LayerScale (Touvron et al. 2021, arXiv:2103.17239). Per-channel
        # learnable diagonal scale `gamma ∈ R^{d_model}` on each sublayer's
        # residual branch. Direct form `x = x + gamma * sub_block(x)` (NOT
        # the reparam `(1+γ)` form used by `use_layerscale` above). Init
        # `gamma = layer_scale_init * ones(d_model)` (default 1e-4) → at
        # step 0 the residual contribution is `1e-4 × sub_block(x)`, four
        # orders of magnitude smaller than the residual stream magnitude,
        # so the val loss at step 0 is within fp32 noise of baseline (the
        # "soft warmup" the paper specifies). Per-channel vs scalar
        # (ReZero, 130) is the headline architectural novelty. Default
        # off → baseline path bit-identical. See
        # `autoresearch/ideas/142-layerscale/idea.md`.
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = float(layer_scale_init)
        if self.use_layer_scale:
            self.attn_gamma = nn.Parameter(
                torch.full((d_model,), self.layer_scale_init)
            )
            self.ffn_gamma = nn.Parameter(
                torch.full((d_model,), self.layer_scale_init)
            )
        # 017 — Sub-LN / Sandwich block: one fresh `nn.LayerNorm(d_model)`
        # per sublayer (γ=1, β=0 init → identity at step 0). When off,
        # the modules are not constructed and the pre-norm baseline is
        # bit-identical (no parameter overhead, no FLOPs overhead).
        # Cost when on: 2 × d_model² = 2 × 144² = 41,472 params per block
        # (screen10m); 2 × 64² = 8,192 per block (tiny1m3m).
        self.use_sub_ln = use_sub_ln
        if self.use_sub_ln:
            self.sub_ln_attn = nn.LayerNorm(d_model, elementwise_affine=True)
            self.sub_ln_ffn = nn.LayerNorm(d_model, elementwise_affine=True)
        # R1 ReZero: one α per sublayer, zero-init (no-op at step 0).
        # Mirrors the use_layerscale wiring: declared as nn.Parameter
        # scalars so they get routed to AdamW by the optimizer setup.
        self.use_re_zero = use_re_zero
        if self.use_re_zero:
            self.re_zero_alpha_attn = nn.Parameter(torch.zeros(1))
            self.re_zero_alpha_ffn = nn.Parameter(torch.zeros(1))
        self._init_resid(resid_mode, d_model, n_layers)
        # 111 — DropPath reads `self.n_layers` in `forward` to schedule
        # `p_l = 1 - drop_path_max * l / (n_layers - 1)`. The kwarg was
        # only used inside `_init_resid` for the DeepNorm constant, so
        # without this attribute the trt training forward crashes at
        # the first drop-path branch. Default `n_layers=1` keeps
        # single-block callers (the closure tests) safe.
        self.n_layers = int(n_layers)
        # #47 FFN embeddings: add a learned projection of the factorized
        # token embedding to the FFN input. Different position from
        # V-embed (#29, inside attention) and O-embed (#33, post-O).
        # Tests whether the V-embed win is about attention content or
        # about residual content. Zero-init, so step 0 = exact baseline.
        # Cost: 24 × (d_model 144 × emb_rank 48) = 165,888 extra params.
        self.use_ffn_embed = use_ffn_embed
        if self.use_ffn_embed:
            assert value_embed_rank is not None, "value_embed_rank required for FFN-embed"
            self.ffn_embed_proj = nn.Parameter(
                torch.zeros(d_model, value_embed_rank)
            )
        # 023 — Canon conv: gated depthwise causal Conv1d on the
        # residual stream. Constructed lazily; never called when
        # `use_canon_conv=False` so the baseline path is bit-
        # identical. See `models/canon_conv.py` for the module doc.
        self.use_canon_conv = use_canon_conv
        if self.use_canon_conv:
            self.canon_conv = CanonConv(d_model)
        # 143 — ShortConv: identity-init depthwise causal Conv1d on
        # the residual stream, with a per-block scalar output gate
        # `g=0` at init. Constructed lazily; never called when
        # `use_short_conv=False` so the baseline path is bit-
        # identical. See `models/short_conv.py` for the module doc.
        self.use_short_conv = use_short_conv
        self.short_conv_kernel = int(short_conv_kernel)
        if self.use_short_conv:
            assert self.short_conv_kernel in (3, 4), (
                f"short_conv_kernel={self.short_conv_kernel} must be 3 or 4"
            )
            self.short_conv = ShortConv1D(d_model, kernel_size=self.short_conv_kernel)
            # Per-block scalar gate `g=0` → step-0 ≡ no-conv baseline.
            # The conv has identity init internally, so g=1 would give
            # `x = x + x = 2x` at step 0 (NOT identity). The gate
            # scales the conv contribution to 0 at init, matching the
            # canon_conv gating pattern.
            self.short_conv_gate = nn.Parameter(torch.zeros(1))
        # 111 — DropPath / Stochastic Depth (Huang et al. 2016). The
        # `p_l` is computed in `forward` from `drop_path_max`,
        # `n_layers` (already a block attr, used by DeepNorm), and
        # the per-call `layer_index` kwarg. Flag is off by default so
        # the entire branch is skipped → baseline path bit-identical.
        self.use_drop_path = use_drop_path
        self.drop_path_max = float(drop_path_max)
        # 131 — LayerDrop. Same per-step coin structure as DropPath
        # but the schedule is independent (`layerdrop_schedule`):
        # "constant" → p_l = p (paper default); "linear" → p_l ramps
        # from 0 to p over L; "stochastic_depth" → p_l = p·l/(L-1).
        # Default off → no gate computed → baseline path bit-identical.
        self.use_layerdrop = use_layerdrop
        self.layerdrop_p = float(layerdrop_p)
        self.layerdrop_schedule = str(layerdrop_schedule or "constant")
        # 118 — Mixture-of-Depths: per-block `MoDRouter` that scores
        # every token, picks the top-k, and gates the block's residual
        # update to that subset. Default off → no router built, baseline
        # forward graph is bit-identical. See `models/mod_router.py`.
        self.use_mod = use_mod
        self.mod_capacity = float(mod_capacity)
        self.mod_router_hidden = int(mod_router_hidden)
        if self.use_mod:
            self.mod_router = MoDRouter(d_model, hidden=self.mod_router_hidden)
        else:
            self.mod_router = None

        # 148 — Focal Modulation. Built lazily; never called when
        # `use_focal_mod=False` so the baseline MHA path is
        # bit-identical. See `FocalModulationBlock` above.
        self.use_focal_mod = use_focal_mod
        self.focal_mod_kernels = tuple(int(k) for k in (focal_mod_kernels or (3, 5, 7)))
        if self.use_focal_mod:
            self.focal_mod = FocalModulationBlock(
                d_model,
                kernels=self.focal_mod_kernels,
                dropout=dropout,
            )
        else:
            self.focal_mod = None

    def _init_resid(self, resid_mode: str, d_model: int, n_layers: int):
        """Build params for the selected residual-add lever. Each sublayer
        ('attn'/'ffn') gets its own params unless the mode is shared. Identity
        at init unless the mode name implies a fixed/const scale (own control).
        See docs/research/residual_stream/plan.md."""
        self.resid_mode = resid_mode or ""
        if not self.resid_mode:
            return
        m = self.resid_mode
        sub = ["attn", "ffn"]

        def P(name, tensor):
            setattr(self, f"_rs_{name}", nn.Parameter(tensor))

        if m == "rezero_shared":            # one α=0, shared by both adds
            P("alpha", torch.zeros(1))
        elif m in ("fixed_half", "fixed_sqrt2", "fixed_deepnorm",
                   "input_scale_half", "ema_resid", "convex_half",
                   "branch_dropout05", "branch_dropout10",
                   "stoch_depth10", "stoch_depth20"):
            pass  # parameter-free (fixed scales / stochastic)
        else:
            for s in sub:
                if m == "rezero_vec":
                    P(f"alpha_{s}", torch.zeros(d_model))
                elif m == "rezero_init_one":
                    P(f"alpha_{s}", torch.ones(1))
                elif m == "rezero_init_half":
                    P(f"alpha_{s}", torch.full((1,), 0.5))
                elif m == "branch_gain":
                    P(f"g_{s}", torch.ones(1))
                elif m == "branch_gain_vec":
                    P(f"g_{s}", torch.ones(d_model))
                elif m == "input_scale":
                    P(f"a_{s}", torch.ones(1))
                elif m == "input_scale_vec":
                    P(f"a_{s}", torch.ones(d_model))
                elif m == "resid_mix":
                    P(f"a_{s}", torch.ones(1)); P(f"g_{s}", torch.ones(1))
                elif m == "resid_mix_vec":
                    P(f"a_{s}", torch.ones(d_model)); P(f"g_{s}", torch.ones(d_model))
                elif m == "highway_sigmoid":
                    P(f"s_{s}", torch.zeros(1))          # 2σ(0)=1
                elif m == "tanh_gate":
                    P(f"s_{s}", torch.zeros(1))          # 1+tanh(0)=1
                elif m == "softplus_gate":
                    w0 = float(torch.log(torch.expm1(torch.tensor(1.0))))
                    P(f"s_{s}", torch.full((1,), w0))    # softplus≈1
                elif m == "clamp_gate":
                    P(f"g_{s}", torch.ones(1))
                elif m == "double_gate":
                    P(f"g1_{s}", torch.ones(1)); P(f"g2_{s}", torch.ones(1))
                elif m == "attn_only_rezero" and s == "attn":
                    P("alpha_attn", torch.zeros(1))
                elif m == "ffn_only_rezero" and s == "ffn":
                    P("alpha_ffn", torch.zeros(1))
                elif m == "attn_only_gain" and s == "attn":
                    P("g_attn", torch.ones(d_model))
                elif m == "ffn_only_gain" and s == "ffn":
                    P("g_ffn", torch.ones(d_model))
        # fixed DeepNorm constant (same for every layer): 1/√(2·n_layers)
        self._resid_deepnorm = (2.0 * max(1, n_layers)) ** -0.5

    def _resid_add(self, x, f, which):
        """x = a·x + g·f for the selected lever (`which` ∈ {attn, ffn})."""
        m = self.resid_mode
        g = lambda n: getattr(self, f"_rs_{n}_{which}", None)
        if m == "rezero_shared":
            return x + self._rs_alpha * f
        if m in ("rezero_vec", "rezero_init_one", "rezero_init_half"):
            a = g("alpha")
            return x + (a if a is not None else 0.0) * f
        if m in ("attn_only_rezero", "ffn_only_rezero"):
            a = getattr(self, f"_rs_alpha_{which}", None)
            return x + f if a is None else x + a * f
        if m == "branch_gain":
            return x + g("g") * f
        if m == "branch_gain_vec":
            return x + g("g") * f
        if m in ("attn_only_gain", "ffn_only_gain"):
            gg = getattr(self, f"_rs_g_{which}", None)
            return x + f if gg is None else x + gg * f
        if m in ("input_scale", "input_scale_vec"):
            return g("a") * x + f
        if m in ("resid_mix", "resid_mix_vec"):
            return g("a") * x + g("g") * f
        if m == "highway_sigmoid":
            return x + (2.0 * torch.sigmoid(g("s"))) * f
        if m == "tanh_gate":
            return x + (1.0 + torch.tanh(g("s"))) * f
        if m == "softplus_gate":
            return x + F.softplus(g("s")) * f
        if m == "clamp_gate":
            return x + g("g").clamp(0.0, 2.0) * f
        if m == "double_gate":
            return x + g("g2") * (g("g1") * f)
        # ---- fixed / const (own control) ----
        if m == "fixed_half":
            return x + 0.5 * f
        if m == "fixed_sqrt2":
            return (2.0 ** -0.5) * x + (2.0 ** -0.5) * f
        if m == "fixed_deepnorm":
            return x + self._resid_deepnorm * f
        if m == "input_scale_half":
            return 0.5 * x + f
        if m == "ema_resid":
            return 0.9 * x + 0.1 * f
        if m == "convex_half":
            return 0.5 * x + 0.5 * f
        # ---- stochastic regularizers (identity at eval) ----
        if m == "branch_dropout05":
            return x + F.dropout(f, p=0.05, training=self.training)
        if m == "branch_dropout10":
            return x + F.dropout(f, p=0.10, training=self.training)
        if m in ("stoch_depth10", "stoch_depth20"):
            p = 0.1 if m == "stoch_depth10" else 0.2
            if not self.training:
                return x + f
            if torch.rand(()) < p:
                return x
            return x + f / (1.0 - p)
        return x + f

    def forward(self, x, x0=None, ve=None, v_residual=None, layer_index=None, shared_kv=None):
        # 111 — DropPath / Stochastic Depth. Linear schedule
        # `p_l = 1 - drop_path_max * l / (n_layers - 1)` where `l` is
        # the 0-indexed layer position (l=0 → p_l=1.0; l=n_layers-1 →
        # p_l = 1 - drop_path_max). One coin flip per block per step,
        # shared across the batch (one 0-dim sample). Eval has no
        # stochasticity. When the block is kept, the residual
        # contribution `(out - x_orig)` is rescaled by `1/p_l` so the
        # expected residual magnitude matches the baseline. The rescale
        # is exact for any p_l ∈ (0, 1]; for n_layers==1 we set p_l=1.0
        # (no drop) to avoid the divide-by-zero.
        drop_path_scale = 1.0
        if self.use_drop_path and self.training and layer_index is not None:
            if self.n_layers > 1:
                p_l = 1.0 - self.drop_path_max * float(layer_index) / float(self.n_layers - 1)
            else:
                p_l = 1.0
            # Clamp to (0, 1] for safety (drop_path_max could be > 1
            # in pathological configs); torch.rand(()) is a 0-dim
            # tensor on the default device.
            p_l = float(p_l)
            if p_l < 1.0:
                if torch.rand(()) >= p_l:
                    return x  # block skipped entirely
                drop_path_scale = 1.0 / p_l
        x_orig = x if drop_path_scale != 1.0 else None
        # 131 — LayerDrop (block-level skip, independent schedule).
        # Schedules (selected via `self.layerdrop_schedule`):
        #   "constant"         → p_l = layerdrop_p (paper default, p=0.2).
        #   "linear"           → p_l = layerdrop_p · l/(L-1) (paper
        #                        stable-training variant — more drops
        #                        at later layers).
        #   "stochastic_depth" → p_l = layerdrop_p · l/(L-1) too (the
        #                        paper's "stochastic depth" schedule
        #                        starts at 0 and ramps up — same math
        #                        as `linear` here; the naming follows
        #                        the paper's section 3.1).
        # One coin flip per block per step, shared across the batch.
        # When the block is kept, rescale the residual delta by 1/p_l
        # so expected magnitude matches baseline. Eval has no
        # stochasticity. Distinct from DropPath (111) above: DropPath
        # is a per-sample/per-batch gate on the residual branch inside
        # the block, with a fixed `p_l = 1 - drop_path_max·l/(L-1)`
        # schedule that starts at p_0=1; LayerDrop is a per-batch gate
        # on the WHOLE block, with `constant` paper default. See
        # `autoresearch/ideas/131-layer-drop/idea.md`.
        layerdrop_scale = 1.0
        if self.use_layerdrop and self.training and layer_index is not None:
            if self.n_layers > 1:
                l = float(layer_index) / float(self.n_layers - 1)
            else:
                l = 0.0
            sched = self.layerdrop_schedule
            if sched == "linear" or sched == "stochastic_depth":
                p_l = self.layerdrop_p * l
            else:  # "constant" (paper default)
                p_l = self.layerdrop_p
            p_l = float(max(1e-6, min(1.0, p_l)))
            if torch.rand(()) >= p_l:
                return x  # block skipped entirely (identity pass-through)
            layerdrop_scale = 1.0 / p_l
        layerdrop_orig = x if layerdrop_scale != 1.0 else None
        # 118 — Mixture-of-Depths: capture the block's input so we can
        # gate the residual delta `(x_after - x_in)` by the per-token
        # router at the end of forward. Skipped (no-router) when
        # `use_mod=False` → zero overhead on the baseline path.
        x_in = x if self.use_mod else None
        # Re-inject the original embedding before attention/MLP (#20)
        if self.use_embed_residual:
            x = self.resid_m0 * x + self.resid_m1 * x0

        # 023 — Canon conv: gated depthwise causal Conv1d on the
        # residual stream, immediately before the attention sublayer's
        # pre-LN. Scalar gate `g=0` at init → bit-identical to no-conv
        # baseline at step 0. ONE conv per block (the spec's placement
        # pin: not before FFN, not at sublayer-input, not twice per
        # block). See `autoresearch/ideas/023-canon-conv/plan.md`.
        if self.use_canon_conv:
            x = self.canon_conv(x)

        # 143 — ShortConv: identity-init depthwise causal Conv1d on
        # the residual stream, immediately before the attention
        # sublayer's pre-LN. Per-block scalar gate `g=0` at init →
        # bit-identical to no-conv baseline at step 0. The conv has
        # identity init (center tap = 1, rest = 0) so the conv output
        # equals the input — the gate absorbs this so `x = x + 0·x = x`.
        # ONE conv per block (matches CanonConv's placement pin). See
        # `autoresearch/ideas/143-shortconv/idea.md`.
        if self.use_short_conv:
            x = x + self.short_conv_gate * self.short_conv(x)

        if self.use_parallel_block:
            # #98 Parallel block: attn and FFN both read one shared normed
            # input; their outputs are summed into the residual together.
            n = self.norm1(x)
            # 024 — pass the raw residual `x` to the MHA gate (pre-LN
            # signal). `n = norm1(x)` is the post-LN signal; the spec
            # pins the gate input as the pre-LN residual.
            # 148 — Focal Modulation: replace the MHA call with the
            # focal block. The focal block takes only the normed input
            # (no `ve`, `gate_x`, `v_residual`, `shared_kv` args).
            if self.use_focal_mod:
                attn_out = self.focal_mod(n)
            else:
                attn_out = self.attention(n, ve, gate_x=x, v_residual=v_residual, shared_kv=shared_kv)
            if self.use_layerscale:
                attn_out = attn_out * (1.0 + self.attn_layerscale)
            if self.use_layer_scale:
                attn_out = attn_out * self.attn_gamma
            ffn_in = n
            if self.use_ffn_embed and ve is not None:
                ffn_in = ffn_in + F.linear(ve, self.ffn_embed_proj)
            ff_out = self.feed_forward(ffn_in)
            if self.use_layerscale:
                ff_out = ff_out * (1.0 + self.ffn_layerscale)
            if self.use_layer_scale:
                ff_out = ff_out * self.ffn_gamma
            return x + self.dropout(attn_out) + self.dropout(ff_out)

        if self.use_post_norm:
            # #75 Post-norm: apply norm AFTER the residual addition.
            # x = norm(x + sublayer(x_or_norm_x))
            # We use the un-normalized x as the sublayer input (the
            # original Transformer design — sometimes called "post-norm").
            # 024 — post-norm: sublayer input is already the raw residual,
            # pass it as the gate signal.
            # 148 — Focal Modulation: replace the MHA call with the
            # focal block. Post-norm passes the raw residual (not the
            # normed one) — focal mod is post-LN-style, so this is the
            # correct input.
            if self.use_focal_mod:
                attn_out = self.focal_mod(x)
            else:
                attn_out = self.attention(x, ve, gate_x=x, v_residual=v_residual, shared_kv=shared_kv)
            if self.use_layerscale:
                attn_out = attn_out * (1.0 + self.attn_layerscale)
            if self.use_layer_scale:
                attn_out = attn_out * self.attn_gamma
            # 017 — Sub-LN wrap on the sublayer output (γ=1, β=0 ⇒ identity
            # at step 0, so post-norm baseline path stays bit-identical
            # when use_sub_ln=False). When on, this constrains each
            # sublayer's contribution to the residual stream to unit-RMS.
            if self.use_sub_ln:
                attn_out = self.sub_ln_attn(attn_out)
            x = self.norm1(x + self.dropout(attn_out))

            ffn_in = x
            if self.use_ffn_embed and ve is not None:
                ffn_in = ffn_in + F.linear(ve, self.ffn_embed_proj)
            ff_out = self.feed_forward(ffn_in)
            if self.use_layerscale:
                ff_out = ff_out * (1.0 + self.ffn_layerscale)
            if self.use_layer_scale:
                ff_out = ff_out * self.ffn_gamma
            if self.use_sub_ln:
                ff_out = self.sub_ln_ffn(ff_out)
            x = self.norm2(x + self.dropout(ff_out))
        else:
            # Pre-norm (default): norm before sublayer, residual after.
            # 024 — pass the raw residual `x` (pre-LN signal) to the MHA
            # gate. The MHA's primary input is `norm1(x)` (post-LN); the
            # gate reads the pre-LN residual.
            # 148 — Focal Modulation: replace the MHA call with the
            # focal block. Reads the same normed input as MHA.
            if self.use_focal_mod:
                attn_out = self.focal_mod(self.norm1(x))
            else:
                attn_out = self.attention(self.norm1(x), ve, gate_x=x, v_residual=v_residual, shared_kv=shared_kv)
            if self.use_layerscale:
                attn_out = attn_out * (1.0 + self.attn_layerscale)
            if self.use_layer_scale:
                attn_out = attn_out * self.attn_gamma
            # 017 — Sub-LN wrap on the sublayer output (γ=1, β=0 ⇒ identity
            # at step 0, so pre-norm baseline path stays bit-identical
            # when use_sub_ln=False). When on, this constrains each
            # sublayer's contribution to the residual stream to unit-RMS.
            if self.use_sub_ln:
                attn_out = self.sub_ln_attn(attn_out)
            # R1 ReZero (pre-norm branch): x = x + α·f(x). α=0 init ⇒
            # zero contribution at step 0; lever is α's learning. Replaces
            # the baseline add (x = x + dropout(attn_out)).
            if self.use_re_zero:
                x = x + self.re_zero_alpha_attn * self.dropout(attn_out)
            elif self.resid_mode:
                x = self._resid_add(x, self.dropout(attn_out), "attn")
            else:
                x = x + self.dropout(attn_out)

            ffn_in = self.norm2(x)
            if self.use_ffn_embed and ve is not None:
                ffn_in = ffn_in + F.linear(ve, self.ffn_embed_proj)
            ff_out = self.feed_forward(ffn_in)
            if self.use_layerscale:
                ff_out = ff_out * (1.0 + self.ffn_layerscale)
            if self.use_layer_scale:
                ff_out = ff_out * self.ffn_gamma
            if self.use_sub_ln:
                ff_out = self.sub_ln_ffn(ff_out)
            # R1 ReZero (pre-norm branch, FFN): same gate on the FFN add.
            if self.use_re_zero:
                x = x + self.re_zero_alpha_ffn * self.dropout(ff_out)
            elif self.resid_mode:
                x = self._resid_add(x, self.dropout(ff_out), "ffn")
            else:
                x = x + self.dropout(ff_out)
        # 111 — DropPath rescale: when this block was kept during
        # training, rescale its residual contribution by `1/p_l` so
        # the expected residual magnitude matches the no-drop-path
        # baseline. `drop_path_scale == 1.0` (and thus `x_orig is None`)
        # when the flag is off or eval mode — short-circuited, no
        # extra ops on the baseline path.
        if x_orig is not None:
            x = x_orig + (x - x_orig) * drop_path_scale
        # 131 — LayerDrop rescale: same structure as DropPath but the
        # kept-block rescale is `1/p_l` for the LayerDrop schedule
        # (constant by default). `layerdrop_scale == 1.0` (and thus
        # `layerdrop_orig is None`) when the flag is off or eval mode
        # — short-circuited, no extra ops on the baseline path.
        if layerdrop_orig is not None:
            x = layerdrop_orig + (x - layerdrop_orig) * layerdrop_scale
        # 118 — Mixture-of-Depths: gate the block's residual delta by
        # the per-token router. Routed tokens get `c · delta`, the rest
        # get `0`. `c = k/T` keeps the expected per-token contribution
        # equal to the dense baseline. The router reads from `x_in`
        # (the block input, pre-norm) so the routing decision is a
        # function of the pre-block residual stream. Default off →
        # `x_in is None` → this branch is skipped → baseline path
        # bit-identical.
        if x_in is not None:
            scores = self.mod_router(x_in)            # [B, T] in [0,1]
            B, T = scores.shape
            k = max(1, int(round(self.mod_capacity * T)))
            # Top-k indices per batch row. Deterministic ordering via
            # `largest=True, sorted=False`; tie-breaks don't matter for
            # the lever (the router has no useful signal at step 0).
            _, top_k_idx = torch.topk(scores, k, dim=-1, largest=True, sorted=False)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, top_k_idx, 1.0)          # [B, T]
            c = float(k) / float(T)
            delta = x - x_in
            x = x_in + (mask.unsqueeze(-1) * c) * delta
        return x
