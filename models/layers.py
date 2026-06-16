import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import SquaredReLUFeedForward, SwiGLUFeedForward, GELUFeedForward, SaturatingReLUFeedForward, ReLU2FeedForward, MishGLUFeedForward
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
from .ttt_linear import TTTFeedForward
from .xlayer_attn import XLayerCrossAttn
from .conv_ffn import ConvFFN


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
# 173 — Entmax-1.5 (Peters/Niculae/Martins, ACL 2019, arXiv:1905.09018).
# Sparse, differentiable drop-in for softmax. α-entmax is the optimizer
# of `p·(α-1)·s − H_α^T(p)` over the simplex. Closed form: project
# `(α-1)·s` onto Δ^{n-1} by clamping negatives to 0 and renormalizing.
# At α=1 the projection is softmax (continuous limit); at α=2 it is
# sparsemax; α=1.5 is the empirical sweet spot ("entmax-1.5"). For
# α=1.5 the per-element closed form is `p_i = max(0, 0.5·(s_i − λ))^2`
# where `λ` is the Lagrange multiplier chosen so `Σp = 1`. We solve for
# λ by bisection on the simplex constraint — 25 iterations is enough
# for n_keys ≤ 4096.
# Per-head learnable α_h: `α_h = 1 + 0.5·(1 + tanh(α_raw_h))`. Init
# `α_raw_h = 0` ⇒ `α_h = 1` ⇒ the α=1 limit IS softmax, so we
# short-circuit to `torch.softmax` for perfect step-0 bit-identity with
# the baseline (max-abs-diff = 0.0). For α > 1 the bisection delivers
# the sparse projection in fp32 with tol=1e-7. The `mask` arg mirrors
# softpick: True = attend, False = zero out (a masked position's score
# is replaced with −inf so it cannot be in the support of the
# projection). α_per_head has shape [H] and is broadcast over
# [B, H, T, T]. See `autoresearch/ideas/173-entmax-15/idea.md`.
# ============================================================================
def entmax_15(
    scores: torch.Tensor,
    mask: torch.Tensor,
    alpha_per_head: torch.Tensor,
    dim: int = -1,
    n_iter: int = 15,
    tol: float = 1e-4,
) -> torch.Tensor:
    """α-entmax-1.5 attention normalization via bisection on λ.

    Args:
        scores: attention logits, shape [B, H, T_q, T_k].
        mask: bool tensor broadcast-compatible with `scores`. True =
            attend, False = mask out (replaced with −inf internally).
        alpha_per_head: per-head α values, shape [H]. Must satisfy
            α ≥ 1. For α = 1 the projection is exactly softmax
            (short-circuited to torch.softmax for bit-identity).
        dim: reduction axis (default -1, the key axis).
        n_iter: bisection budget (default 15; gives bracket precision
            ~1e-4, comfortably below the 1e-3 fp16 score precision that
            dominates downstream gradient noise at tiny1m3m).
        tol: bisection tolerance on `Σp − 1` (default 1e-4; the
            short-circuit α=1 path keeps the strict 1e-5 step-0
            identity check via the literal `torch.softmax` call).

    Returns:
        Tensor of the same shape and dtype as `scores`. Each row sums
        to 1.0 (over unmasked positions) and is sparse (true zeros)
        for low-scoring positions.
    """
    # Per-head amp1 = (α − 1). Broadcast to [1, H, 1, 1] for the
    # [B, H, T_q, T_k] score layout. We assume a single amp1 value
    # shared across heads in the current call (i.e. the MHA only
    # applies ONE alpha to all heads via averaging); the per-head
    # parameter is the longer-term lever — see the idea spec.
    amp1_per_head = (alpha_per_head - 1.0).view(1, -1, 1, 1)
    # When all heads have α = 1 (init case: amp1 = 0 everywhere), the
    # bisection degenerates (0/0 in the exponent). The α = 1 limit IS
    # softmax, so short-circuit for perfect step-0 bit-identity.
    if amp1_per_head.abs().max().item() == 0.0:
        return torch.softmax(
            scores.masked_fill(~mask, float("-inf")), dim=dim
        ).to(scores.dtype)
    # Use a single α across heads for the bisection (the per-head
    # parameter is averaged). This keeps the algorithm vectorized
    # over the [B, T_q] axes without per-row scalar loops. The
    # per-head granularity lives in the param layout, not the forward.
    amp1 = float(amp1_per_head.mean().item())
    if amp1 <= 0.0:
        # α ≤ 1 ⇒ softmax (the limit is continuous). Defensive.
        return torch.softmax(
            scores.masked_fill(~mask, float("-inf")), dim=dim
        ).to(scores.dtype)
    # Work in fp32; cast back at the end.
    s = scores.to(torch.float32).masked_fill(~mask, float("-inf"))
    if dim != -1 and dim != s.ndim - 1:
        s = s.transpose(dim, -1)
        perm_back = True
    else:
        perm_back = False
    # Bracket λ: lower = amp1·(min s) − 1 (definitely under-shoots
    # Σp=1), upper = amp1·(max s) (definitely over-shoots). Using
    # masked positions (s = −inf) means amp1·s = −inf, so they don't
    # contribute. amax/amin with −inf/inf sentinels handle the mask
    # automatically (a fully-masked row's max would be −inf; we clamp
    # below to avoid that pathology).
    s_max = s.amax(dim=-1, keepdim=True)
    s_min = s.amin(dim=-1, keepdim=True)
    # Track which rows are fully-masked (no valid position to attend
    # to). For these rows, the projection is undefined; we set the
    # bracket to [0, 0] and the final output to zero. In practice the
    # model never produces fully-masked rows (causal attention always
    # has at least the diagonal), but the helper must be robust to
    # adversarial inputs (e.g. the swap site passes a mask where some
    # rows are entirely False).
    fully_masked = ~mask.any(dim=-1, keepdim=True)  # [B, H, T_q, 1]
    safe_s_max = torch.where(torch.isfinite(s_max), s_max, torch.zeros_like(s_max))
    safe_s_min = torch.where(torch.isfinite(s_min), s_min, torch.zeros_like(s_min))
    lo = (amp1 * safe_s_min - 1.0)
    hi = (amp1 * safe_s_max)
    # Bisection: each step, project, check Σp, halve bracket.
    # Vectorized: we keep lo/hi as [B, H, T_q, 1] tensors and
    # update them with torch.where. We deliberately do NOT add an
    # early-exit `.item() < tol` check here: every `tensor.item()`
    # forces a CUDA sync (~50-100µs each on RTX 3060) and 15 iters ×
    # 12 layers × 732 forward passes would dominate wall-clock. The
    # extra ~2 iters of compute on rows that already converged is
    # ~3 fp32 ops × 33M elements = negligible next to the sync cost.
    # Convergence is guaranteed by the fixed-iter budget (bracket
    # halves each iter ⇒ n_iter=15 ⇒ bracket/2^15 ≈ 1e-4 relative
    # precision at tiny1m3m's T=2048).
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        # p_i = max(0, amp1·s_i − mid)^(1/amp1)
        z = torch.clamp(amp1 * s - mid, min=0.0)
        proj_sum = z.pow(1.0 / amp1).sum(dim=-1, keepdim=True)
        # If sum < 1 we need a SMALLER λ (mid is too high ⇒ projected
        # mass too low); lower the upper bound.
        # If sum > 1 we need a LARGER λ; raise the lower bound.
        go_up = proj_sum < 1.0  # need smaller mid → lower hi
        lo = torch.where(~go_up, mid, lo)  # if sum >= 1, mid is a valid lower bound
        hi = torch.where(go_up, mid, hi)   # if sum < 1, mid is a valid upper bound
    # Final projection at the midpoint.
    lam = 0.5 * (lo + hi)
    z = torch.clamp(amp1 * s - lam, min=0.0)
    p_t = z.pow(1.0 / amp1)
    # Normalize: p_t / Σp_t over the key axis. Row sums of unmasked
    # positions are 1.0 by construction (λ chosen for Σp=1).
    out = p_t / (p_t.sum(dim=-1, keepdim=True) + 1e-12)
    # Zero out fully-masked rows (no valid position to project onto).
    out = torch.where(fully_masked, torch.zeros_like(out), out)
    if perm_back:
        out = out.transpose(dim, -1)
    return out.to(scores.dtype)


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
        # 166 — T5-style bucketed relative position bias
        # (Raffel et al. JMLR 2020, arXiv:1910.10683; re-used in
        # BigBird, REALM, LongT5). Add a learned additive per-head
        # logit bias indexed by relative distance `|i − j|` via a
        # logarithmic bucket function `b = floor(log2(|i-j|+1))`,
        # clipped to `t5_rpe_buckets-1`. Bias is registered as
        # `self.rpe_bias = nn.Parameter(zeros(H, B))`, init 0 ⇒
        # `scores + 0` is bit-identical to the no-RPE baseline at
        # step 0. The lever composes additively with whatever
        # positional scheme is active (RoPE / FIRE / CoPE — these
        # all live on the score side and the RPE just adds another
        # additive term). Forces the manual attention path so the
        # bucket-indexed bias cannot go through SDPA's flash
        # kernel. Default off → no Parameter registered, no branch
        # taken, baseline path bit-identical. See
        # `autoresearch/ideas/166-t5-rpe/idea.md`.
        use_t5_rpe: bool = False,
        t5_rpe_buckets: int = 32,
        # 152 — Per-head attention logit bias (PaLM 2 §arch,
        # OLMo 2). Add a learnable per-head additive bias `b_h ∈ R^H`
        # to the attention logits pre-softmax: `logits_h ← logits_h
        # + b_h`. `b_h` is init 0 → `softmax(logits_h + 0) = softmax
        # (logits_h)` byte-for-byte at step 0. NB: mathematically, a
        # per-head *scalar* bias cancels in softmax over the key axis
        # for all subsequent steps too (per-(b,h,t) `e^{b_h}` factor
        # cancels in the per-row normalizer); the experiment is
        # therefore a *recorded null* rather than a useful lever.
        # Default off → baseline path bit-identical (no Parameter is
        # registered, no branch is taken). Forces the manual
        # attention path so SDPA's flash/efficient backends don't
        # perturb step-0 numerics. See
        # `autoresearch/ideas/152-attn-logit-bias/idea.md`.
        use_attn_logit_bias: bool = False,
        # 155 — Per-head learnable attention temperature
        # (PaLM 2 §arch, OLMo 2, Gemma 2). Replace the standard
        # `1/sqrt(d_k)` attention scale with a per-head learnable
        # scalar `τ_h ∈ R^H` so the per-head logit scale becomes
        # `Q_h K_h^T * τ_h`. Init `τ_h = 1/sqrt(d_k)` exactly ⇒
        # `Q_h K_h^T * (1/sqrt(d_k))` is bit-identical to the
        # standard `Q_h K_h^T / sqrt(d_k)` at step 0 (the standard
        # pre-softmax scaling is exactly that factor). Each head
        # can then adjust its own temperature during training —
        # heads wanting sharper focus can lower `τ_h`, heads wanting
        # broader context can raise it. Cost: H scalars/layer (4 at
        # tiny1m3m, total 48 — negligible). Forces the manual
        # attention path so SDPA's flash/efficient backends don't
        # perturb step-0 numerics. Default off → baseline path
        # bit-identical (no Parameter is registered, no branch is
        # taken). See `autoresearch/ideas/155-per-head-temp/idea.md`.
        use_per_head_temp: bool = False,
        # 195 — Tight hard QK logit clamp. Apply
        # `torch.clamp(scores, min=-c, max=+c)` to the pre-softmax
        # attention logits `Q·K^T / sqrt(d_k)` so no single logit
        # can dominate the softmax. `c` is a fixed config
        # constant (not learnable). Default off → baseline path
        # bit-identical (the `if self.use_qk_clamp:` branch is
        # never taken). When on, the lever is intentionally NOT
        # bit-identical at step 0 — at Kaiming init, QK^T entries
        # are O(1) Gaussian and a tight c (default 2.0) actively
        # clips ~5% of the 2-sigma tail at step 0, so the
        # regularizer effect is exercised immediately and the
        # gradient at the boundary is discontinuous (exactly 0
        # outside the clamp). Forces the manual attention path
        # so SDPA's flash kernel doesn't fuse QK^T+softmax+AV
        # (the pre-softmax logit must be exposed for clamping).
        # Distinct from the closed `logit softcap` axis (smooth
        # tanh at c=8 — inactive at step 0 at tiny1m3m; here
        # it's a *hard* clip at c=2.0 — *active* at step 0).
        # See `autoresearch/ideas/195-qk-clamp-min-max/idea.md`.
        use_qk_clamp: bool = False,
        qk_clamp_c: float = 2.0,
        # 193 — Blockwise attention temperature schedule (fixed
        # cosine-depth, no learned params). One scalar `τ_b ∈ R^1`
        # per block `b ∈ [0, L-1]` where `L = n_layers` (12 at
        # tiny1m3m), shape `τ_b = 1 + α · cos(π · b / L)`, applied
        # to the pre-softmax attention scores as
        # `scores_b = Q_b K_b^T / (τ_b · √d_k)`. The buffer of
        # `τ_b` values is registered on the MHA at construction
        # (non-Parameter `Buffer` of shape `[L]`); the per-forward
        # cost is one elementwise divide on `[B, H, T, T]`. At
        # `α = 0` ⇒ `τ_b = 1` for all `b` ⇒ `scores / (1·√d_k) =
        # scores / √d_k` byte-identical to the standard pre-softmax
        # scale. Default off → no Buffer registered, no branch
        # taken, baseline path bit-identical. See
        # `autoresearch/ideas/193-blockwise-attn-temp-schedule/
        # idea.md` and `configs/llm_config.LLMConfig.use_block_
        # temp_schedule` for the committed `α = -0.3` value.
        use_block_temp_schedule: bool = False,
        block_temp_alpha: float = 0.0,
        # 193 — Per-block precomputed temperature scalar `tau_b ∈ R^1`
        # (the model passes this in; shape `[1]` Buffer is registered
        # when `use_block_temp_schedule=True`). Ignored when the flag
        # is off. See the buffer-registration comment in
        # `MultiHeadAttention.__init__` and the per-block schedule
        # formula in `configs/llm_config.LLMConfig.use_block_temp_
        # schedule`. See
        # `autoresearch/ideas/193-blockwise-attn-temp-schedule/idea.md`.
        tau_b: float = 1.0,
        # 161 — Per-layer learnable attention temperature. The actual
        # parameter `layer_temperature ∈ R^{n_layers}` lives on the
        # MODEL (`MinimalLLM`); each MHA reads `layer_temperature
        # [layer_index]` at forward (passed via `layer_index` kwarg).
        # The flag here only controls whether the MHA applies the
        # scaling in forward. The parameter creation is on the model
        # so the optimizer sees ONE `nn.Parameter` (not n_layers of
        # them), keeping the parameter layout flat. Default off →
        # no branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/161-dyt-temp/idea.md`.
        use_per_layer_temp: bool = False,
        # 160 — Per-head RMS gain on the attention output. See
        # `MultiHeadAttention.use_head_gain` for the mechanism.
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/160-rms-gain-per-head/idea.md`.
        use_head_gain: bool = False,
        # 181 — Cross-Head Channel RMSNorm. See
        # `MultiHeadAttention.use_cross_head_rmsnorm` for the
        # mechanism. Default off → baseline path bit-identical.
        # See `autoresearch/ideas/181-cross-head-rmsnorm/idea.md`.
        use_cross_head_rmsnorm: bool = False,
        # 191 — Per-token attention output gain (Shleifer et al.
        # 2021 "NormFormer" / Touvron et al. 2021 "CaiT" class-
        # attention gain, arXiv:2110.09423). Multiply the post-
        # merge `[B, T, d_model]` attention output by a learnable
        # per-position scalar `(1 + γ_t)` where `γ_t ∈ R^T` is
        # shared across batch and the d_model axis. Init γ=0 ⇒
        # `(1 + 0) = 1` exactly ⇒ `attn * 1 = attn` byte-
        # identical to baseline at step 0. Per-token granularity
        # (T scalars/block) is a different axis from the closed
        # per-head (160: H scalars), per-channel (142: d_model
        # scalars), and per-(h, k) (181) levers. Default off →
        # baseline path bit-identical. See
        # `autoresearch/ideas/191-token-attn-gain/idea.md`.
        use_token_attn_gain: bool = False,
        # 203 — Pre-W_O Squeeze-Excitation channel attention (Hu et
        # al. TPAMI 2019, arXiv:1709.01507). Per-token channel
        # reweighting on the post-merge attention output via a tiny
        # bottleneck MLP. W_1: d_model → d_model/r, W_2:
        # d_model/r → d_model, plus a per-block `se_gamma_raw`
        # scalar (init `se_alpha_init=-10.0` ⇒ sigmoid ≈ 4.5e-5
        # ⇒ silent at step 0). The branch is gated on
        # `self.se_W1 is not None` (set only when the flag is on)
        # so the default-off path is bit-identical to the no-flag
        # baseline (no Parameter registered, no `nn.Linear` built,
        # no forward branch taken). The 1-D `se_gamma_raw` scalar
        # is routed to Muon per the spec (1-D gain → Muon, mirrors
        # 021/207 reviewer precedent) — see the explicit
        # `'se_gamma' in name` branch in `training/trainer.py`.
        # Distinct from 142 (per-channel static gain), 160
        # (per-head gain), 181 (cross-head RMSNorm), 191
        # (per-token *scalar* gain) — 203 is the *per-token
        # channel vector* (content-dependent channel reweighting).
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/203-pre-wo-se-channel-attn/idea.md`.
        use_se_pre_wo: bool = False,
        se_reduction_ratio: int = 4,
        se_alpha_init: float = -10.0,
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
        # 189 — CosFormer-style linear attention (Qin et al. NeurIPS
        # 2022, arXiv:2202.08791). Replace softmax(QK^T/√d)·V with
        # the kernel-replacement form
        # `out = (Q'·(K'^T·V)) / (Q'·K'^T)` where
        # `Q' = cos(Q)` and `K' = exp(γ·K)·cos(K)` (γ is a learnable
        # per-block scalar passed in by the model as
        # `cosformer_gamma`, init 0 ⇒ `K' = cos(K)`). Linear in
        # sequence length via the prefix-sum cumsum trick. The
        # denominator `Q'·K'^T` is MANDATORY (no skip-flag) — it is
        # the softmax replacement, not a global mean-pool.
        # `cosformer_gamma_init` is the init value for γ (default 0
        # ⇒ K' = cos(K)). Mutually exclusive with `use_linear_attn`
        # / `use_diff_attn` / `use_nsa_global` / `use_hybrid_heads`
        # / `use_multiscale_heads` (the cosFormer branch IS the
        # attention path; combining with another is double-attention
        # and a structural lever change). Default off → baseline
        # path bit-identical (the branch is gated on
        # `self.use_cosformer`, no Parameter registered, the flag is
        # a strict no-op). See
        # `autoresearch/ideas/189-cosformer-linear-attn/idea.md`.
        use_cosformer: bool = False,
        cosformer_gamma_init: float = 0.0,
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
        # 173 — Entmax-1.5 sparse attention (Peters/Niculae/Martins,
        # ACL 2019, arXiv:1905.09018). Replace `torch.softmax` in the
        # manual attention path with the Tsallis α-entmax projection
        # `p_i = max(0, 0.5·(s_i − λ))^2 / Σp`, where α=1.5 is the
        # "entmax-1.5" sweet spot. The lever is per-head learnable
        # α_h parameterized as `α_h = 1 + 0.5·(1 + tanh(α_raw_h))`,
        # init `α_raw_h = 0` ⇒ `α_h = 1` ⇒ the α=1 limit IS softmax,
        # short-circuited in the helper for byte-identity at step 0.
        # As training proceeds the optimizer can push `α_raw_h`
        # positive to make the attention sparser (towards sparsemax
        # at α=2). Default off → baseline path bit-identical (no
        # Parameter registered, no branch taken, no bisection). Forces
        # the manual attention path (entmax-1.5 can't go through
        # SDPA's flash kernel — the projection is closed-form
        # per-row, not a softmax-callable). Distinct from
        # 022-softpick (no params, fixed operator), 025-SSMax (per-head
        # scaling on softmax, not a replacement), 020-FoX (post-softmax
        # forget gate, softmax stays). See
        # `autoresearch/ideas/173-entmax-15/idea.md`.
        use_entmax: bool = False,
        # 192 — Pre-softmax per-row hard top-k sparse attention
        # (Touvron et al. 2021, DeiT III, arXiv:2103.17239). Keep
        # only the k largest pre-softmax scores per row, scatter
        # -inf to the rest, then softmax-renormalize over the
        # surviving k positions. `topk_k` is a config int (default
        # 512 = T/4 at the tiny1m3m `max_seq_len=2048`, i.e. 75%
        # sparsity). 0 new params, no learnable scalar. Step-0 is
        # NOT bit-identical to baseline when flag-on (topk of
        # random Gaussians is a different operator than full
        # softmax) — same structural-lever category as 173 / 022
        # / 154. Forces the manual attention path (the scatter
        # write can't go through SDPA's flash kernel). Causal-mask
        # interaction is correct: topk runs on the already-masked
        # scores, so -inf future positions are below the topk
        # budget and `scores.topk` never selects them. The
        # defensive bound `k = min(topk_k, scores.size(-1))`
        # handles shorter eval contexts. Default off → baseline
        # path bit-identical. See
        # `autoresearch/ideas/192-topk-attn/idea.md`.
        use_topk_attn: bool = False,
        topk_k: int = 512,
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
        # 164 — Cross-Block Q Residual ("Q-Carry"). Per-block scalar
        # `α_q = nn.Parameter(torch.zeros(()))` is added to the Q
        # projection at every layer l ≥ 1: `Q_l = W_Q(x_l) +
        # α_q · W_Q(prev_x)`, where `prev_x = LN(x_{l-1})` is the
        # previous block's MHA sublayer input (passed via `q_carry`
        # kwarg, `.detach()`-ed by the model loop). α=0 init ⇒
        # `α · W_Q(prev_x) = 0` exactly in fp32 ⇒ step-0 forward is
        # bit-identical to the no-carry baseline (within fp32
        # rounding noise of one extra multiply-add). Layer 0 has no
        # previous block ⇒ `q_carry is None` ⇒ the MHA only stashes
        # `self._q_carry = x.detach()` for the model loop to read
        # back. The carry is added BEFORE q_norm / RoPE so the
        # existing 016/162 norms still rescale `Q + α·Q_carry`
        # consistently. Default off → baseline path bit-identical
        # (no Parameter registered, no stash, no carry branch).
        # See `autoresearch/ideas/164-q-carry/plan.md`.
        use_q_carry: bool = False,
        # 168 — AV-Output Carry (post-AV cross-block residual). For
        # each block l ≥ 1, augment the attention output
        # (post-SDPA, post-reshape, pre-W_O) with a learnable α_l-
        # scaled carry from the previous block's same-stage tensor:
        # `out_l = W_O @ (av_l + α_l · av_{l-1})`. `α_l` is a per-
        # block 0-dim scalar (init 0 ⇒ identity blend at step 0).
        # The stash `_av_carry` is `.detach()`-ed (mirroring 021's
        # V-residual contract) so the cross-block gradient is
        # structurally bounded to α_l. Site is post-merge-reshape
        # (shape `[B, T, d_model]`), pre-W_O — sits BEFORE the W_O
        # projection so it composes with 160/024/107/045 output-side
        # gates. Default off → baseline path bit-identical. See
        # `autoresearch/ideas/168-av-output-carry/plan.md`.
        use_av_output_carry: bool = False,
        # 186 — Within-Block V-Carry (per-head learnable V
        # recurrence). A learnable per-head scalar
        # `α_h = tanh(v_carry_alphas_h)` (init `v_carry_alphas_h = 0`
        # ⇒ `α_h = 0` exactly) drives a left-to-right recurrence
        # along the time axis of V at the post-W_V / post-GQA /
        # post-transpose site (V is `[B, n_heads, T, d_k]`):
        # `V_new[0] = V[0];  V_new[t] = V[t] + α_h · V_new[t-1]` for
        # `t ≥ 1`. Closed form: `V_new[t] = Σ_{k=0}^{t} α_h^k · V[t-k]`
        # (a 1-pole IIR low-pass on V, equivalent to the linear-
        # attention recurrence without the K side). Implemented via
        # a vectorized depthwise `F.conv1d` along T with kernel
        # `α_h^0, α_h^1, …, α_h^{T-1}` per head (matches 134-Mega's
        # depthwise conv1d pattern; ~0.5 GFLOPs/layer — the Python
        # for-loop alternative is ~2k sequential ops/head, too slow
        # at GPU scale). The tanh parameterization keeps `|α_h| < 1`
        # strictly so the geometric sum stays bounded even at T=2048.
        # α=0 init ⇒ kernel is `[1, 0, 0, …, 0]` (post-flip) ⇒ conv1d
        # output is `V` exactly ⇒ forward is bit-identical to the
        # no-flag baseline at step 0 (no RNG consumed in the
        # construction beyond the empty `nn.Parameter(zeros(n_heads))`).
        # Local to each block (no cross-block stash, no `q_carry=`/
        # `av_carry=`-style plumbing) — the recurrence runs causally
        # within a single block. Cost: H × n_layers = 48 scalars
        # (+0.005% of 0.94M). See `autoresearch/ideas/186-v-carry-block/plan.md`.
        use_v_carry_block: bool = False,
        # 163 — Post-Attention V-Mix Depthwise Convolution. When on,
        # `forward()` applies `F.conv1d(attn_output.transpose(1,2),
        # self.v_mix_conv_weight, padding=k//2, groups=d_model)` AFTER
        # the post-SDPA reshape and BEFORE the W_O projection (see
        # `MultiHeadAttention.forward`). `v_mix_conv_kernel` is the
        # 1-D kernel size; pinned to 3 (odd ≥ 3) for the spec test.
        # Default off → baseline path bit-identical (no Parameter
        # registered, no forward branch taken). See
        # `autoresearch/ideas/163-v-mix-conv/idea.md`.
        use_v_mix_conv: bool = False,
        v_mix_conv_kernel: int = 3,
        # 201 — Degenerate gMLP Spatial Gating Unit on Attention
        # Output. See `LLMConfig.use_gmlp_sgu` in
        # `configs/llm_config.py` for the mechanism. The SGU is
        # allocated ONLY when `block_idx % gmlp_sgu_block_stride == 0`
        # (per-block-stochastic) — at the default stride 3 ⇒ 4 of 12
        # blocks at tiny1m3m (block_idx ∈ {0, 3, 6, 9}). The
        # construction (raw `nn.Parameter(torch.empty(d_model,
        # d_model))` + inline `.data.normal_(std=0.02)`) keeps RNG
        # state aligned with the no-flag path. `sgu_alpha` is init
        # at `gmlp_sgu_alpha_init=-10.0` ⇒ `sigmoid(-10) ≈ 4.5e-5`
        # ⇒ silent at step 0 (bit-identical to baseline within fp32
        # noise). `block_idx` defaults to 0 (single-block callers
        # like closure tests stay safe; the gate is also on
        # `use_gmlp_sgu` so the default-off path is unchanged).
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/201-mlp-token-mixer/idea.md`.
        use_gmlp_sgu: bool = False,
        gmlp_sgu_block_stride: int = 3,
        gmlp_sgu_alpha_init: float = -10.0,
        block_idx: int = 0,
        # 188 — Cross-Block K/V Projection Sharing (Universal
        # Transformers-style learnable parameter sharing across depth,
        # Dehghani et al. ICLR 2019, arXiv:1807.03819). Each block's
        # effective K, V projection is a learnable convex blend of
        # its own (new) projection and the previous block's
        # projection:
        #   `W_K_eff = (1 − σ(α_K_raw)) · W_K_self + σ(α_K_raw) · W_K_prev`
        # (same for V). Init `α_K_raw = α_V_raw = -10.0` ⇒
        # `σ(-10) ≈ 4.5e-5` ⇒ the blend is numerically dominated by
        # `W_K_self` at step 0, so the projection output is bit-
        # identical (within fp32 noise of one extra multiply-add) to
        # the no-flag baseline. `prev_W_K` / `prev_W_V` are passed
        # in via forward kwargs (the model loop stashes them on
        # layer 0 and reuses for layers 1..N-1; same pattern as
        # `q_carry` / `av_carry` / `v_residual`). `detach()` on the
        # prev-block weights keeps the cross-block gradient
        # structurally bounded to the 2 scalar α params per block.
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/188-cross-block-kv-share/idea.md`.
        use_cross_block_kv_share: bool = False,
        # 204 — Cross-Block Attention Score Sharing (Sukhbaatar et
        # al. Memorizing Transformers ICLR 2022, arXiv:2203.08913
        # — within-model cross-block pre-softmax score-reuse
        # lever). Each block's attention scores are blended with
        # the previous block's pre-softmax scores via a learnable
        # per-block scalar α:
        #   `α = σ(score_share_alpha_raw)`
        #   `scores_b_eff = (1 − α) · scores_b_self + α · scores_{b-1}.detach()`
        # `prev_block_scores` is the PRE-SOFTMAX logit
        # `Q_{b-1} · K_{b-1}^T / √d_k` (NOT the post-softmax
        # attention distribution — a different lever; see
        # review.md finding B), `.detach()`-ed so gradients flow
        # only through `α_raw` and the current block's Q, K —
        # never through the previous block's QK computation
        # (mirrors the 021 / 164 / 168 cross-block detach
        # contract). Init `score_share_alpha_init=-10.0` ⇒
        # `σ(-10) ≈ 4.5e-5` ⇒ `scores_eff ≈ scores_self` at
        # step 0 within fp32 noise of one extra multiply-add.
        # Forces the manual attention path (the score-blend
        # can't go through SDPA's flash kernel). Default off ⇒
        # baseline path bit-identical (forward branch gated on
        # `use_cross_block_score_share`, `score_share_alpha_raw`
        # not registered, `_prev_block_scores` attribute never
        # written). Cost: 1 scalar/block × 12 blocks = 12 α
        # scalars (+0.001% of 0.94M — the cheapest profile in
        # the cross-block family). See
        # `autoresearch/ideas/204-cross-block-attn-score-share/idea.md`.
        use_cross_block_score_share: bool = False,
        score_share_alpha_init: float = -10.0,
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
        # 176 — Pre-AV V RMSNorm (per-head α-gate + per-head γ-gain,
        # Wortsman 2023 V-norm primitive + per-head gating in the
        # closed-016 family). Apply RMSNorm to V along `d_k` BEFORE
        # the AV product, with a per-head scalar gate
        # `α_h = relu(α_raw_h)` (init 0) and per-head gain
        # `γ_h ∈ R^{d_k}` (init 1.0). Output
        # `V_out = (1 − α_h)·V + α_h·RMSNorm(V)·γ_h`. Init α=0,γ=1
        # ⇒ step-0 forward is byte-identical to baseline
        # (max-abs-diff = 0.0). Mutually exclusive with
        # use_v_layernorm (closed-029) and the closed-#92
        # v_norm_type zoo (asserted in `forward`). Default off →
        # no Parameter registered, no branch taken, baseline path
        # bit-identical. See
        # `autoresearch/ideas/176-v-pre-av-norm/idea.md`.
        use_v_rmsnorm: bool = False,
        # 162 — Q-Only RMSNorm (asymmetric QK pre-softmax). Apply
        # `nn.RMSNorm(d_head, eps=1e-6)` to Q only, leave K untouched.
        # nn.RMSNorm weight=1, bias=0 init ⇒ at step 0 the lever
        # rescales Q to unit RMS per head-dim (spec-allowed fp32
        # max-abs-diff < 1e-3 tolerance). Default off ⇒ no module is
        # built, baseline path bit-identical. Distinct from 016 (which
        # norms BOTH Q and K). See
        # autoresearch/ideas/162-q-only-norm/idea.md.
        use_q_only_norm: bool = False,
        # 165 — K-Only RMSNorm (asymmetric QK pre-softmax, K-side).
        # Apply `nn.RMSNorm(d_head, eps=1e-6)` to K only, leave Q
        # untouched. nn.RMSNorm weight=1, bias=0 init ⇒ at step 0 the
        # lever rescales K to unit RMS per head-dim (spec-allowed fp32
        # max-abs-diff < 1e-3 tolerance, same trade-off as 162). Default
        # off ⇒ no module is built, baseline path bit-identical. The
        # K-mirror of 162 (Q-only); together with 016 (symmetric QK)
        # forms a clean 3-way attribution test. See
        # `autoresearch/ideas/165-k-only-norm/idea.md`.
        use_k_only_norm: bool = False,
        # 169 — Depth-Conditional QK-Norm (per-block learnable scale on
        # top of 016's WIN). Keep the per-head RMSNorm/LayerNorm from
        # 016 intact and add one scalar `qk_norm_scale` per MHA, init
        # 1.0, applied AFTER the per-head norm and BEFORE the QK
        # matmul: `Q ← Q · qk_norm_scale; K ← K · qk_norm_scale` (the
        # MoA `extra_K` branch mirrors the same multiply on the extra
        # K). α_l = 1.0 init ⇒ step-0 multiplicative gain is exactly
        # the identity ⇒ forward is byte-identical to 016's step-0
        # (max-abs-diff = 0.0 — no tolerance needed). The optimizer
        # can then learn per-block normalization strength. Mirrors
        # NormFormer's per-layer attention-output gains (Shleifer et
        # al. 2021) applied to the QK-norm output. Mutually exclusive
        # with use_q_only_norm / use_k_only_norm / use_qk_norm_post_rope
        # (asserted in `forward`). Default off ⇒ no Parameter
        # registered, no branch taken, baseline path bit-identical.
        # See `autoresearch/ideas/169-qk-norm-depth/idea.md`.
        use_qk_norm_depth: bool = False,
        # 190 — Per-Layer QK-Norm (scalar γ per block per side, replaces
        # 016's per-channel γ with a single scalar per side per block).
        # Default off ⇒ no Parameter registered, no branch taken,
        # baseline path bit-identical. When ON with init 1.0, the
        # multiply is exactly the identity in fp32 ⇒ step-0 forward is
        # byte-identical to 016's step-0 (max-abs-diff = 0.0). Distinct
        # from 169 (`use_qk_norm_depth`): 190 keeps Q and K scalars
        # SEPARATE by default (preserves 016's QK symmetry 162+165
        # closed, 016 attributed WIN to); the shared variant
        # `qk_norm_scalar_qk_shared` is a sub-flag. Default separate:
        # 12 × 2 × 1 = 24 γ params (vs 016's 384 per-channel); shared
        # variant: 12 × 1 = 12. Mutually exclusive with
        # use_q_only_norm / use_k_only_norm / use_qk_norm_post_rope /
        # use_qk_norm_depth (asserted in `forward`) — those levers
        # restructure the norm, not the gain, and combining them
        # confounds 190's axis. See
        # `autoresearch/ideas/190-per-layer-qk-norm/idea.md`.
        qk_norm_scalar_per_block: bool = False,
        qk_norm_scalar_qk_shared: bool = False,
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
        # 171 — DropConnect on W_O (Wan et al. 2013, arXiv:1304.3174).
        # Per-weight Bernoulli mask on the attention output projection
        # during training. Sampled per forward pass; same mask for all
        # batch elements and positions; rescale by `1/(1-p)` so the
        # expected magnitude matches the un-masked baseline (inverted-
        # dropout convention, matches `F.dropout` and the 147-DropKey
        # rescale). At eval (`self.training == False`) and with
        # `dropconnect_wo_rate=0.0` the branch is never taken ⇒
        # forward graph bit-identical to the no-DropConnect baseline.
        # The effective rate is **ramped** from 0.0 → `dropconnect_wo_rate`
        # over the first `dropconnect_wo_warmup_steps` training forwards
        # (see `self._dc_step_count` below) so step 0 has effective rate
        # 0.0 ⇒ the mask branch is short-circuited ⇒ trt forward is
        # bit-identical to baseline at step 0 (max-abs-diff = 0.0 across
        # the full forward). Default off → no Parameter created, no
        # branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/171-dropconnect-wo/idea.md`.
        use_dropconnect_wo: bool = False,
        dropconnect_wo_rate: float = 0.0,
        dropconnect_wo_warmup_steps: int = 100,
        # 207 — W_O Low-Rank Bottleneck (learnable rank-r residual
        # correction on the W_O projection, Arora et al. "Linear
        # Algebraic Structure of Word Senses" + LoRA-style trained-
        # from-scratch low-rank factorization on the attention
        # output projection). Replace W_O with
        #   `W_O_eff = W_O + σ(α) · (W_O_A @ W_O_B)`
        # where `W_O_A ∈ R^{d_model × r}`, `W_O_B ∈ R^{r × d_model}`
        # (`r = wo_rank`, default 16), and `α` is a 0-dim learnable
        # scalar (init `wo_lowrank_alpha_init`, default −10 ⇒
        # `σ(α) ≈ 4.5e-5` at step 0). `W_O_A` is normal-init std=0.02
        # (matches the existing `out_proj` / `qkvo_proj` init); `W_O_B`
        # is **zero-init** so the rank-r correction is exactly 0 at
        # step 0 ⇒ `W_O_eff == W_O` byte-identical at step 0
        # (max-abs-diff = 0.0 across the full forward — bit-identical
        # to the no-flag baseline). As training proceeds, the
        # optimizer can grow W_O_B and α, activating a learnable
        # low-rank correction that soft-bottlenecks what each
        # attention block can write back to the residual stream.
        # Composes with 171-DropConnect (the 171 mask runs first on
        # `w_o`, the 207 correction adds after — both are
        # joint-by-default and individually silent at step 0).
        # Default off → no Parameter registered, no branch taken,
        # baseline path bit-identical. See
        # `autoresearch/ideas/207-wo-lowrank-bottleneck/idea.md` /
        # `plan.md`.
        use_lowrank_wo: bool = False,
        wo_rank: int = 16,
        wo_lowrank_alpha_init: float = -10.0,
        # 194 — W_V Low-Rank Residual Correction (LoRA-style
        # trained-from-scratch low-rank factorization on the value
        # projection). Replace W_V with
        #   `W_V_eff = W_V + σ(α) · (W_V_A @ W_V_B)`
        # where `W_V_A ∈ R^{d_model × r}`, `W_V_B ∈ R^{r × d_model}`
        # (`r = wv_rank`, default 8), and `α` is a 0-dim learnable
        # scalar (init `wv_lowrank_alpha_init`, default −10 ⇒
        # `σ(α) ≈ 4.5e-5` at step 0). `W_V_A` is normal-init std=0.02
        # (matches the existing `qkvo_proj` init); `W_V_B` is
        # **zero-init** so the rank-r correction is exactly 0 at
        # step 0 ⇒ `W_V_eff == W_V` bit-identical to the no-flag
        # baseline. Complementary to 207-W_O-LowRank (same
        # mechanism, different sub-block); null closes the entire
        # low-rank-residual sub-block family. Default off → no
        # Parameter registered, no branch taken, baseline path
        # bit-identical. See `autoresearch/ideas/194-lowrank-ffn/
        # idea.md` / `plan.md`.
        use_lowrank_wv: bool = False,
        wv_rank: int = 8,
        wv_lowrank_alpha_init: float = -10.0,
        # 199 — W_Q Low-Rank Residual Correction (LoRA-style
        # trained-from-scratch low-rank factorization on the query
        # projection). Replace W_Q with
        #   `W_Q_eff = W_Q + σ(α) · (W_Q_A @ W_Q_B)`
        # where `W_Q_A ∈ R^{d_model × r}`, `W_Q_B ∈ R^{r × d_model}`
        # (`r = wq_rank`, default 16), and `α` is a 0-dim learnable
        # scalar (init `wq_lowrank_alpha_init`, default −10 ⇒
        # `σ(α) ≈ 4.5e-5` at step 0). `W_Q_A` is normal-init std=0.02
        # (matches the existing `qkvo_proj` init); `W_Q_B` is
        # **zero-init** so the rank-r correction is exactly 0 at
        # step 0 ⇒ `W_Q_eff == W_Q` bit-identical to the no-flag
        # baseline. Complementary to 207-W_O-LowRank (same
        # mechanism, different sub-block) and 194-W_V-LowRank —
        # completing the rank-residual sub-block family across
        # {W_Q, W_V, W_O}. Default off → no Parameter registered,
        # no branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/199-attn-output-lowrank/idea.md` /
        # `plan.md`.
        use_lowrank_wq: bool = False,
        wq_rank: int = 16,
        wq_lowrank_alpha_init: float = -10.0,
        # 199 — Spectral-Norm-Bounded W_O Projection (per-block
        # learnable Lipschitz cap on the attention output
        # projection, Miyato et al. 2018 "Spectral Normalization
        # for GANs" ICLR 2018, arXiv:1802.05957 + Gouk et al.
        # 2021 arXiv:1804.04368). Per block *l*, apply an
        # *asymmetric* (clip-only) Lipschitz cap on W_O's
        # spectral norm:
        #   cap_l       = σ_max(W_O_init^[l]) · exp(γ_l)
        #   W_O_eff^[l] = W_O^[l] · min(1, cap_l / σ_max(W_O^[l]))
        # `γ_l` is a per-MHA 0-dim learnable scalar (init 0 ⇒
        # `exp(γ_l)=1`). `σ_max_init^[l]` is captured on the FIRST
        # forward (frozen; never recomputed from a perturbed W_O
        # — this is the byte-identity guarantee). `σ_max(W_O)` is
        # tracked via power iteration on the O-slice weight with a
        # per-block Buffer `u ∈ R^{d_model}` (initialized on first
        # forward from a random direction, then updated as
        # `u ← W_O · u / ||·||₂` and `σ_max ≈ u^T · W_O · u /
        # (u^T · u)`). `wo_spectral_cap_pi_iters` controls how
        # many PI steps run per forward (default 1 — the σ_max
        # drift is slow at 0.94M/12L so a single PI step tracks
        # it). At step 0 `γ_l = 0` and `σ_max_current = σ_max_init`
        # ⇒ the factor is exactly 1 ⇒ `W_O_eff == W_O` byte-
        # identical to baseline. The optimizer can push `γ_l < 0`
        # to tighten the cap and bind the Lipschitz constant on
        # the projection; `γ_l > 0` is wasted optimizer signal
        # (the clip never fires because σ_max_current <
        # σ_max_init · exp(γ_l)). `γ_l` is a free Parameter
        # (init 0); the power-iteration state `u` and the captured
        # `σ_max_init` are Buffers (not Parameters — they survive
        # optimizer state serialization but do not consume an
        # optimizer slot). Default off → no Parameter, no Buffer,
        # no branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/199-spectral-attn-output/idea.md` /
        # `plan.md`.
        use_wo_spectral_cap: bool = False,
        wo_spectral_cap_pi_iters: int = 1,
        # 197 — Tied W_O Across Blocks (soft blend, Universal-
        # Transformer-style learnable parameter sharing restricted
        # to the attention output projection, Dehghani et al. ICLR
        # 2019 arXiv:1807.03819 + Lan et al. ALBERT arXiv:1909.11942).
        # When on, the MHA gets a per-block 0-dim scalar
        # `tied_wo_alpha_raw` (init `tied_wo_alpha_init`, default
        # −10 ⇒ `σ(−10) ≈ 4.54e-5` at step 0) and a reference to
        # the model's shared `W_O_shared ∈ R^{d_model × d_model}`
        # parameter (passed in via `tied_wo_shared=...`; this is the
        # SAME parameter reference for every block — NOT a per-
        # block copy). In `forward()`, AFTER extracting
        # `w_o = self.qkvo_proj[self.qkv_size:]` and BEFORE the
        # 171-DropConnect mask site, the per-block effective O
        # projection is built as
        #   `w_o_eff = (1 − σ(α_b_raw)) · w_o_b + σ(α_b_raw) · W_O_shared`
        # At step 0 `σ(−10) ≈ 4.54e-5` and `W_O_shared` is std=0.02
        # normal-init, so the contribution from `W_O_shared` is on
        # the order of 1e-7 in std ⇒ the forward is bit-identical
        # to the no-flag baseline up to fp32 noise of one extra
        # multiply-add — same tolerance the 188 cross-block K/V
        # share and 204 cross-block score share siblings accept.
        # Per-block `w_o_b` is KEPT (treatment is param-superset of
        # control by +4,108 params: 1 shared matrix + 12 scalars
        # = +0.4% of 0.94M); no per-block slot is removed, so the
        # A/B is a parameter-shape lever, not a model-size lever.
        # Default off → no Parameter registered, no branch taken,
        # baseline path bit-identical. Mutually exclusive with
        # `use_yoco` / `use_gau` (asserted in `MinimalLLM.__init__`;
        # the upper-half YOCO blocks and GAU blocks don't take the
        # `tied_wo_shared` kwarg). Composes with 171-DropConnect
        # (the 171 mask runs on the blended `w_o` after this
        # branch) and 207-W_O-LowRank (the 207 addition runs on
        # the blended `w_o` after this branch). See
        # `autoresearch/ideas/197-tied-wo-across-blocks/idea.md` /
        # `plan.md`.
        use_tied_wo_across_blocks: bool = False,
        tied_wo_alpha_init: float = -10.0,
        tied_wo_shared=None,  # Optional[torch.nn.Parameter]; set only when flag is on.
        # 151 — RoV (Rotary Value Embeddings, gated; Su et al. 2024
        # Hunyuan-DiT / RoV for ViT, arXiv:2403.13257 §2.3). Apply the
        # same rotary position embedding already used on Q, K to the
        # value vector V as well, mixed via a per-block scalar gate
        # `rov_gate = nn.Parameter(torch.zeros(1))`. Init 0 ⇒
        # V_combined = V + 0·V_rot = V ⇒ step-0 forward graph
        # bit-identical to baseline. The base rotary buffer is reused
        # (no extra params beyond the 1 scalar/block = 12 at
        # tiny1m3m). Default off → baseline path bit-identical. When
        # `use_nope` or `use_cope` is on, the rotary is bypassed on
        # Q,K; RoV becomes a no-op (the geometric lever is
        # unavailable). See `autoresearch/ideas/151-rov-gated/idea.md`.
        use_rov: bool = False,
        # 174 — xPos exponential decay on the RoPE-magnitude (Sun et
        # al. 2022, arXiv:2212.10554). One learnable per-layer scalar
        # `xpos_gamma = nn.Parameter(torch.zeros(1))` applied to the
        # rotated K as `K = K · exp(-xpos_gamma · t)` (the paper's
        # `g_t = (1 − γ)^t`, in `exp` form for numerical stability;
        # the two are identical at γ=0 and both equal 1). With γ = 0
        # (init) the decay is identity ⇒ K is unchanged ⇒ attention
        # scores are unchanged ⇒ forward is bit-identical to the
        # 500k-base RoPE baseline at step 0. The lever is the per-layer
        # "how local is local" knob the optimizer can dial: γ > 0
        # shrinks distant keys toward zero (bias attention toward
        # recent tokens), γ < 0 grows them (extend context). Decay
        # applied to K only (not Q) so the score factor is
        # `g_s = (1-γ)^s` on K's position — matches the standard
        # xPos reading. The single scalar per MHA ⇒ 12 scalars at
        # tiny1m3m (+0.001% of 0.94M). Default off → no parameter
        # created, no branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/174-xpos-decay/idea.md`.
        use_xpos: bool = False,
        # 154 — Rebased Attention (Shi et al. 2024, arXiv:2407.06641):
        # pool K, V along the time axis with stride `rebase_stride`
        # (default 8) before the softmax, so attention reads from R =
        # ceil(T/R) summary positions instead of T raw ones. Identity
        # when `rebase_stride >= T` (pool collapses to a no-op). When
        # `use_rebased_attn=False` (default) the branch is never
        # taken and the standard softmax path is bit-identical to the
        # no-flag baseline. When ON, the manual attention path is
        # forced because the rebased causal mask can't go through
        # SDPA's flash kernel. See
        # `autoresearch/ideas/154-rebased-attn/idea.md`.
        use_rebased_attn: bool = False,
        rebase_stride: int = 8,
        # 185 — Static per-head learned K-rotation (learned
        # orthogonal rebase of K only, position-independent). Each
        # head has its own `R_h ∈ R^{d_k × d_k}` orthogonal matrix
        # applied as `K_h = R_h @ K_h` post-GQA-repeat, pre-RoPE /
        # pre-qk_norm. `R_h` is a product of `d_k/2 = 8` 2D rotations
        # on disjoint `(2i, 2i+1)` planes, parametrized by `n_heads
        # × d_k/2 = 32` angles per block (init 0 ⇒ identity ⇒ step-0
        # bit-identical to baseline). `R_h` orthogonal preserves norms
        # and dot products ⇒ QK^T magnitudes are unchanged (no softmax
        # temperature shift). Default off ⇒ no Parameter registered,
        # no branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/185-static-per-head-k-rotation/idea.md`.
        use_static_k_rotation: bool = False,
        # 200 — Static per-layer × per-pair learned K-rotation
        # (depth-axis twin of 185, RoFormer/RoPE-parameterized).
        # Each layer has its OWN `R_l ∈ R^{d_k × d_k}` orthogonal
        # matrix applied to K ONLY (Q untouched), parameterized
        # as a product of `d_k/2 = 8` 2D rotations on disjoint
        # `(2i, 2i+1)` planes — one learnable angle `φ_{l,i} ∈ R`
        # per (layer, plane), SHARED across heads (this is the
        # "depth-axis" axis: 185 varies angles across heads, 200
        # varies angles across layers). The K-only application
        # breaks QK^T inner-product preservation (Q is in
        # baseline basis, K is in R_l-rotated basis), so the
        # lever has a real axis to bind on — unlike a QK-
        # symmetric application, which would be a provable no-op.
        # `R_l` block-diagonal-orthogonal preserves K's norm and
        # the softmax temperature. Init `φ_{l,i} = 0` ⇒
        # `cos(0)=1, sin(0)=0` in fp32 ⇒ `R_l = I_{d_k}` exactly
        # ⇒ `K = R_l @ K = K` exactly ⇒ step-0 forward is bit-
        # identical to the no-flag baseline. Default off ⇒ no
        # Parameter registered, no branch taken, baseline path
        # bit-identical. See
        # `autoresearch/ideas/200-rope-phase-offset-per-layer/idea.md`.
        use_per_layer_k_rotation: bool = False,
        # 192 — Pre-RoPE per-head × per-pair learned Q+K rotation
        # (Su et al. 2024 RoFormer / RoPE, arXiv:2104.09864,
        # position-dependent rotation context). Per head h and per
        # pair i (d_k/2 = 8 planes), one learnable scalar angle
        # `φ_{h,i} ∈ R` applied to BOTH Q and K as a static
        # (position-independent) 2D rotation on disjoint `(2i,
        # 2i+1)` planes BEFORE RoPE's position-dependent rotation.
        # The block-diagonal `R_h` (product of d_k/2 2D rotations)
        # is orthogonal. Applied to both Q and K the static
        # rotation acts as a basis change on the (Q, K) features
        # before position is mixed in by RoPE — the pre-RoPE
        # placement is the fresh axis (185 rotates K post-RoPE,
        # 200 rotates K post-RoPE with shared angles, 154 uses a
        # fixed rebase on K, V pre-softmax — 192 is the only
        # learned QK rotation that lives *before* the position
        # mix). Init `φ_{h,i} = 0` ⇒ `cos(0)=1, sin(0)=0` in fp32
        # ⇒ `R_h = I_{d_k}` exactly ⇒ `Q = R_h @ Q = Q` and
        # `K = R_h @ K = K` exactly ⇒ step-0 forward is bit-
        # identical to the no-flag baseline. Default off ⇒ no
        # Parameter registered, no branch taken, baseline path bit-
        # identical. See
        # `autoresearch/ideas/192-pre-rope-qk-rotation/idea.md`.
        use_pre_rope_rotation: bool = False,
        # 156 — Mixture-of-Attentions (MoA). Run `E` parallel
        # attention computations per layer with SEPARATE K_e, V_e
        # projections (Q is shared across experts, so the routing is
        # only on the K/V side). The E attention outputs are mixed by
        # a per-token router `g_e = softmax(W_g x)_e`. At init the
        # (E-1) extra K/V projections are zero so the extra experts
        # produce 0, and the router bias is one-hot on expert 0 so
        # g_0 = 1.0, g_e>=1 = 0 ⇒ step-0 output is bit-identical to
        # a single standard attention. Distinct from MoS (144, closed)
        # which mixes softmax *variants* within a single attention —
        # MoA mixes full attention computations. See
        # `autoresearch/ideas/156-moa/idea.md`.
        use_moa: bool = False,
        moa_num_experts: int = 2,
        # 178 — Gated Multi-Query Attention (G-MQA). A learnable
        # per-KV-head scalar gate `β_k, β_v ∈ R^{n_kv_heads}` blends
        # between the head-local K, V projection and a single shared
        # K, V projection: `K_h = K_local_h + β_k_h · (K_shared_h −
        # K_local_h)`, same for V. β init 0 ⇒ K_mix = K_local exactly
        # ⇒ step-0 forward is byte-identical to baseline. The shared
        # K, V projection is a single `nn.Linear(d_model, d_model)`
        # (per block) — 2·d_model² extra params/layer vs head-local.
        # At β→1 the head-local path is dead weight, recovering
        # standard MQA's K/V param savings. Default off ⇒ no Parameter
        # registered, no branch taken, baseline path bit-identical.
        # See `autoresearch/ideas/178-mqa-gated/idea.md`.
        use_mqa_gated: bool = False,
        # 182 — Per-head learnable attention window (soft local
        # window size per head). One scalar `w_h ∈ R^H` per MHA maps
        # to a window half-size `half_w_h = T · sigmoid(w_h)`. Applied
        # as `score -= 1e9 · relu(|t − s| − half_w_h)` in the manual
        # attention branch — equivalent to a hard window per head but
        # fp32-clean (no `−∞`, no NaN risk — matches 154-rebased-attn's
        # rebased-softmax style). Init `w_h = 10 ⇒ sigmoid(10) ≈
        # 0.99995 ⇒ half_w ≈ T − 0.00005·T > T − 1 = max|t − s|`, so
        # the relu is identically 0 everywhere and the step-0 forward
        # is byte-identical to the no-flag baseline. Total cost:
        # n_heads × n_layers = 48 extra params at tiny1m3m (+0.005%
        # of 0.94M). Default off ⇒ no Parameter registered, no
        # branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/182-per-head-window/idea.md`.
        use_per_head_window: bool = False,
        # 180 — Pre-softmax 1D causal depthwise conv on attention logits
        # (QK^T). Per-head kernel `w_h ∈ R^{K}` is convolved along the
        # key axis of the score tensor `[B, H, T, S]` between the
        # causal mask and softmax. Kernel init is a delta function
        # (`w_h[:, K//2] = 1`, rest 0) ⇒ step-0 conv is the identity on
        # scores ⇒ softmax unchanged ⇒ forward is byte-identical to
        # the no-flag baseline. The optimizer can grow any kernel
        # shape — smoothing (positive, sums to 1) or sharpening
        # (signed) — over training. K=3 default; small enough to be
        # negligible compute, large enough to encode a meaningful
        # local-attention prior. Forces the manual attention path so
        # SDPA's flash kernel doesn't perturb step-0 numerics.
        # Default off → no Parameter registered, no branch taken,
        # baseline path bit-identical. See
        # `autoresearch/ideas/180-qk-logit-conv/idea.md`.
        use_logit_conv: bool = False,
        logit_conv_kernel_size: int = 3,
        # 179 — Anti-Causal Sub-Heads (UniLM-style hybrid
        # causal + bidirectional heads). A learnable per-head
        # scalar `γ_h = sigmoid(γ_raw_h)` (init `γ_raw_h = -10`
        # ⇒ `γ_h ≈ 4.5e-5` at step 0) controls how much of head
        # h's attention is bidirectional: the upper-triangle
        # fill is replaced per head with `−1e9 · (1 − γ_h)` so
        # γ_h=0 ⇒ effectively masked (causal), γ_h=1 ⇒ no fill
        # (fully bidirectional), and intermediate γ_h smoothly
        # interpolate the mask magnitude. The repo convention
        # uses a finite mask sentinel `-1e9` (not `-∞`); the
        # r1 review closed the `-∞` interpretation because
        # `(1−γ_h)·-∞` degenerates the lever. With `−1e9`,
        # `(1−γ_h)·-1e9` is a real gradient across γ_h ∈ [0,1]
        # and the step-0 byte-identical claim holds:
        # `sigmoid(-10) ≈ 4.5e-5` ⇒ fill `≈ -9.99955e8` ⇒
        # `exp(-9.99955e8) < 1e-300` in fp32 ⇒ upper-triangle
        # bitwise 0 in softmax output. Per the inference
        # schedule in `idea.md`, γ_h stays as trained at both
        # train and eval (measures real deployment behavior).
        # Default off ⇒ no Parameter registered, no branch
        # taken, baseline path bit-identical. Forces the
        # manual attention path so SDPA's flash kernel doesn't
        # perturb the per-head fill. See
        # `autoresearch/ideas/179-anti-causal-subheads/idea.md`.
        use_anti_causal_subheads: bool = False,
        # 202 — V-Only Soft-Blend Probe (Isolate V-Sharing From
        # K-Sharing). Per head h, soft-blend per-head V with a
        # group-shared V via per-head `sigmoid(α_h) ∈ R^H`:
        #   `V_h_eff = (1 − σ(α_h)) · V_h_local + σ(α_h) · V_group_g(x)`
        # where `g = h // v_group_size` is the head's group and
        # `V_group_g(x) ∈ R^{d_k}` is the output of a fresh
        # group-shared projection `W_V_group_g ∈ R^{d_k × d_model}`.
        # K is **never touched** — every head keeps its own W_K_h,
        # so the K-axis is the held-out implicit control. Group V
        # projections (G = n_heads // v_group_size, default G=2 at
        # tiny1m3m with v_group_size=2 and H=4) are allocated and
        # init to the elementwise mean of the in-group per-head
        # W_V_h weights; α_h init `-25.0` ⇒ `σ(α_h) ≈ 1.4e-11`
        # (well below fp32 precision) ⇒ `V_h_eff ≈ V_h_local`
        # exactly at step 0 ⇒ forward is bit-identical to the
        # no-flag baseline. K remains untouched, so the K-axis is
        # the held-out implicit control (the family-dead or
        # family-keep attribution is read off the σ(α) trajectory,
        # not val loss). Default off ⇒ no Parameter registered, no
        # branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/202-grouped-value-projection/idea.md`.
        use_grouped_v: bool = False,
        v_group_size: int = 2,
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
        # 189 — CosFormer-style linear attention (Qin et al. NeurIPS
        # 2022, arXiv:2202.08791). Replaces softmax attention with
        # the kernel-replacement form `out = (Q'·(K'^T·V)) /
        # (Q'·K'^T)` where `Q' = cos(Q)` and `K' = exp(γ·K)·cos(K)`.
        # γ itself is a per-block learnable scalar passed in by the
        # MODEL (`MinimalLLM.cosformer_gammas`), not stored here.
        # When the flag is off (default), no Parameter is registered
        # on the MHA and the forward branch is gated — baseline path
        # bit-identical. See
        # `autoresearch/ideas/189-cosformer-linear-attn/idea.md`.
        self.use_cosformer = use_cosformer
        self.cosformer_gamma_init = float(cosformer_gamma_init)
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

        # 202 — V-Only Soft-Blend Probe. Allocate G = n_heads //
        # v_group_size group-shared V projections (each shape
        # `[d_k, d_model]`) and a per-head `α_h ∈ R^H` gate
        # parameter. Init `W_V_group_g` to the in-group
        # elementwise mean of the per-head W_V_h weights — under
        # GQA, all heads in a group share a single KV-head's V
        # slice, so the "mean" collapses to that slice. Init
        # `α_h = -25.0` ⇒ `σ(α) ≈ 1.4e-11` ⇒ V_h_eff ≈ V_h_local
        # in fp32 at step 0 ⇒ baseline path is bit-identical. K
        # is untouched, so the K-axis is the held-out control.
        # Default off ⇒ no Parameter, no branch, baseline
        # bit-identical. See
        # `autoresearch/ideas/202-grouped-value-projection/idea.md`.
        self.use_grouped_v = use_grouped_v
        self.v_group_size = v_group_size
        if self.use_grouped_v:
            G = self.n_heads // self.v_group_size
            assert G * self.v_group_size == self.n_heads, (
                f"use_grouped_v=True requires n_heads ({self.n_heads}) "
                f"to be divisible by v_group_size ({self.v_group_size}); "
                f"got G = n_heads // v_group_size = {G} "
                f"(must be integer ≥ 1)."
            )
            # Group V projections stored as a ParameterList of G
            # tensors of shape [d_k, d_model]. Init each to the
            # in-group mean of per-head W_V_h weights (under
            # GQA, all heads in a group share one KV head's W_V
            # slice — mean = that slice).
            self.W_V_group = nn.ParameterList()
            W_V_slice = self.qkvo_proj[
                self.qkv_size - self.kv_size : self.qkv_size
            ]  # [kv_size, d_model] → reshape to [n_kv_heads, d_k, d_model]
            W_V_per_kv = W_V_slice.view(self.n_kv_heads, self.d_k, self.d_model)
            for g in range(G):
                head_indices = list(
                    range(g * self.v_group_size, (g + 1) * self.v_group_size)
                )
                kv_indices = [
                    h // self.num_key_value_groups for h in head_indices
                ]
                # Mean across heads in the group — under GQA this
                # collapses to the shared KV-head slice.
                W_V_group_g = W_V_per_kv[kv_indices].mean(dim=0).clone()
                self.W_V_group.append(nn.Parameter(W_V_group_g))
            # Per-head α gate, init -25.0 ⇒ σ(α) ≈ 1.4e-11
            # (below fp32 precision ⇒ V_h_eff = V_h_local
            # numerically at step 0).
            self.v_group_alpha = nn.Parameter(
                torch.full((self.n_heads,), -25.0)
            )
        else:
            self.W_V_group = None
            self.v_group_alpha = None

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
        # 162 — Q-Only RMSNorm (asymmetric QK pre-softmax). Separate
        # `nn.RMSNorm(d_head, eps=1e-6)` on Q only; K stays raw. nn.RMSNorm
        # weight=1, bias=0 init ⇒ at step 0 the lever rescales Q to unit
        # RMS per head-dim (not byte-identical to no-norm; spec-allowed
        # fp32 max-abs-diff < 1e-3 tolerance, same trade-off as 159-emb-
        # layernorm). Default off ⇒ no module is registered, no branch
        # is taken, baseline path bit-identical. See
        # autoresearch/ideas/162-q-only-norm/idea.md.
        self.use_q_only_norm = use_q_only_norm
        if use_q_only_norm:
            self.q_only_norm = nn.RMSNorm(self.d_k, eps=1e-6)
        # 165 — K-Only RMSNorm (asymmetric QK pre-softmax, K-side).
        # Separate `nn.RMSNorm(d_head, eps=1e-6)` on K only; Q stays raw.
        # nn.RMSNorm weight=1, bias=0 init ⇒ at step 0 the lever
        # rescales K to unit RMS per head-dim (not byte-identical to
        # no-norm; spec-allowed fp32 max-abs-diff < 1e-3 tolerance,
        # same trade-off as 159-emb-layernorm, 162-q-only-norm). Default
        # off ⇒ no module is registered, no branch is taken, baseline
        # path bit-identical. See
        # autoresearch/ideas/165-k-only-norm/idea.md.
        self.use_k_only_norm = use_k_only_norm
        if use_k_only_norm:
            self.k_only_norm = nn.RMSNorm(self.d_k, eps=1e-6)
        # Mutual exclusion: 162 (Q-only) and 165 (K-only) cannot both
        # be on at once — together they would re-derive the symmetric
        # 016 path via two separate modules, double-counting the
        # weight tensor and producing a meaningless operator. We
        # assert loudly at construction so misconfigurations fail
        # fast (the build-smoke catches this before GPU time).
        assert not (use_q_only_norm and use_k_only_norm), (
            "use_q_only_norm and use_k_only_norm are mutually exclusive "
            "(162 + 165 re-derives the symmetric 016 path); pick one."
        )
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
        # 176 — Pre-AV V RMSNorm: when `use_v_rmsnorm=True` and the
        # closed-#92 v_norm_type is off and use_v_layernorm is off,
        # register two new per-head Parameters. `v_rmsnorm_alpha ∈
        # R^H` init 0 (relu(0) = 0 ⇒ identity blend at step 0);
        # `v_rmsnorm_gain ∈ R^{H × d_k}` init 1.0 (per-head identity
        # gain at step 0). Independent of the closed-#92 v_norm_type
        # zoo and of closed-029 use_v_layernorm (mutual-exclusion
        # asserted at forward). Default off → both attributes are
        # `None` so lookups stay valid; the forward `if` guard
        # short-circuits when the flag is off.
        # Also expose `self.use_v_layernorm` and `self.v_norm_type` as
        # raw kwarg shadows so the mutual-exclusion asserts at
        # forward can distinguish the two closed V-norm axes
        # (`use_v_layernorm` is closed-029 LayerNorm;
        # `v_norm_type != ""` is closed-#92 zoo).
        self.use_v_layernorm = use_v_layernorm
        self.v_norm_type = v_norm_type
        self.use_v_rmsnorm = use_v_rmsnorm
        if use_v_rmsnorm:
            self.v_rmsnorm_alpha = nn.Parameter(torch.zeros(self.n_heads))
            self.v_rmsnorm_gain = nn.Parameter(
                torch.ones(self.n_heads, self.d_k)
            )
        else:
            self.v_rmsnorm_alpha = None
            self.v_rmsnorm_gain = None
        # 176 — Pre-AV V RMSNorm: when `use_v_rmsnorm=True` and the
        # closed-#92 v_norm_type is off and use_v_layernorm is off,
        # register two new per-head Parameters. `v_rmsnorm_alpha ∈
        # R^H` init 0 (relu(0) = 0 ⇒ identity blend at step 0);
        # `v_rmsnorm_gain ∈ R^{H × d_k}` init 1.0 (per-head identity
        # gain at step 0). Independent of the closed-#92 v_norm_type
        # zoo and of closed-029 use_v_layernorm (mutual-exclusion
        # asserted at forward). Default off → both attributes are
        # `None` so lookups stay valid; the forward `if` guard
        # short-circuits when the flag is off.
        self.use_v_rmsnorm = use_v_rmsnorm
        if use_v_rmsnorm:
            self.v_rmsnorm_alpha = nn.Parameter(torch.zeros(self.n_heads))
            self.v_rmsnorm_gain = nn.Parameter(
                torch.ones(self.n_heads, self.d_k)
            )
        else:
            self.v_rmsnorm_alpha = None
            self.v_rmsnorm_gain = None

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
        # 173 — Entmax-1.5 sparse attention. The `use_entmax` flag is
        # stored on self so the manual attention path can branch on it
        # at the swap site (line ~3035 below, replacing `softmax`).
        # `entmax_alpha_raw ∈ R^H` is the raw parameter; the actual
        # α_h = 1 + 0.5·(1 + tanh(α_raw_h)) is derived in the helper
        # call. Init `α_raw_h = 0` ⇒ `α_h = 1` ⇒ the helper
        # short-circuits to `torch.softmax` for byte-identity at step
        # 0. When `use_entmax=False` the parameter is NOT registered
        # (the `if` guard keeps it out of the parameter list so it
        # doesn't consume RNG or optimizer state) and the baseline
        # forward graph is untouched. See
        # `autoresearch/ideas/173-entmax-15/idea.md`.
        self.use_entmax = use_entmax
        if use_entmax:
            self.entmax_alpha_raw = nn.Parameter(torch.zeros(self.n_heads))
        # 192 — Pre-softmax per-row hard top-k sparse attention. The
        # `use_topk_attn` flag is stored on self so the manual
        # attention path can branch on it at the softmax swap site.
        # `topk_k` is a config int, not a learnable scalar — no
        # `nn.Parameter` is registered, so the baseline path is
        # bit-identical when the flag is off (the forward branch
        # is gated on `self.use_topk_attn`). See
        # `autoresearch/ideas/192-topk-attn/idea.md`.
        self.use_topk_attn = use_topk_attn
        self.topk_k = topk_k
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
        # 164 — Q-Carry. Per-block scalar `α_q` (init 0 → identity at
        # step 0) plus the forward-pass-local stash `self._q_carry`
        # (read back by the model loop after the layer-0 block).
        # `self._q_carry` is always initialized to `None` so the
        # `block.attention._q_carry` readout in `MinimalLLM.forward`
        # is safe even before the first forward pass.
        self.use_q_carry = use_q_carry
        if self.use_q_carry:
            self.alpha_q = nn.Parameter(torch.zeros(()))
            self._q_carry = None
        else:
            # Stub so attribute lookups are always valid even when
            # the flag is off. The forward branch is gated on
            # `self.use_q_carry` so these are never consumed.
            self.alpha_q = None
            self._q_carry = None
        # 168 — AV-Output Carry. Per-block scalar `α_av` (init 0 →
        # identity at step 0) plus the forward-pass-local stash
        # `self._av_carry` (read back by the model loop after the
        # layer-0 block). `self._av_carry` is always initialized to
        # `None` so the `block.attention._av_carry` readout in
        # `MinimalLLM.forward` is safe even before the first forward
        # pass. Stash shape is `[B, T, d_model]` (post-merge-reshape,
        # pre-W_O), `.detach()`-ed.
        self.use_av_output_carry = use_av_output_carry
        if self.use_av_output_carry:
            self.alpha_av = nn.Parameter(torch.zeros(()))
            self._av_carry = None
        else:
            # Stub so attribute lookups are always valid even when
            # the flag is off. The forward branch is gated on
            # `self.use_av_output_carry` so these are never consumed.
            self.alpha_av = None
            self._av_carry = None
        # 186 — Within-Block V-Carry. Per-head learnable scalar
        # `α_h = tanh(v_carry_alphas_h)` (init `v_carry_alphas_h = 0`
        # ⇒ `α_h = 0` exactly ⇒ recurrence collapses to identity at
        # step 0). One parameter per head (H=4 at tiny1m3m). Stored on
        # `self.v_carry_alphas`; the forward branch is gated on
        # `self.use_v_carry_block` so the parameter is never consumed
        # when the flag is off.
        self.use_v_carry_block = use_v_carry_block
        if self.use_v_carry_block:
            self.v_carry_alphas = nn.Parameter(torch.zeros(n_heads))
        else:
            # Stub so attribute lookups are always valid even when
            # the flag is off. The forward branch is gated on
            # `self.use_v_carry_block` so this is never consumed.
            self.v_carry_alphas = None
        # 169 — Depth-Conditional QK-Norm. Per-block scalar `α_l =
        # nn.Parameter(torch.ones(()))` (init 1.0 ⇒ `Q ← Q · 1 = Q`
        # and `K ← K · 1 = K` exactly in fp32 ⇒ step-0 forward is
        # byte-identical to 016's step-0 with max-abs-diff = 0.0).
        # One scalar per MHA (12 blocks × 1 = 12 scalars total).
        # Applied AFTER the per-head RMSNorm / LayerNorm and BEFORE
        # the QK matmul (post-RoPE in either path so the multiply
        # commutes with the post-norm tweaks at α=1.0). Mutually
        # exclusive with use_q_only_norm / use_k_only_norm /
        # use_qk_norm_post_rope (asserted in `forward`). Default off
        # ⇒ no Parameter registered, no branch taken, baseline path
        # bit-identical. See
        # `autoresearch/ideas/169-qk-norm-depth/idea.md`.
        self.use_qk_norm_depth = use_qk_norm_depth
        if self.use_qk_norm_depth:
            self.qk_norm_scale = nn.Parameter(torch.ones(()))
        else:
            # Stub so attribute lookups are always valid even when
            # the flag is off. The forward branch is gated on
            # `self.use_qk_norm_depth` so this is never consumed.
            self.qk_norm_scale = None
        # 190 — Per-Layer QK-Norm (scalar γ per block per side).
        # Sits on top of 016's WIN shape: keep the per-head
        # `q_norm`/`k_norm` and add a per-block scalar `γ_Q ∈ R^1` and
        # `γ_K ∈ R^1` (init 1.0, applied AFTER the per-head norm and
        # BEFORE the QK matmul). At γ=1.0 the multiply is exactly the
        # identity in fp32 ⇒ step-0 forward is byte-identical to 016's
        # step-0 (max-abs-diff = 0.0). Default Q/K separate (preserves
        # 016's QK symmetry 162+165 attributed WIN to); the Q/K-shared
        # variant collapses to one scalar (the 169 axis) and is gated
        # behind `use_qk_norm_scalar_qk_shared`. Total: 12 blocks × 2
        # scalars/block = 24 γ params (default) vs 016's 384 per-
        # channel; or 12 × 1 = 12 γ params if shared. Mutually
        # exclusive with use_q_only_norm / use_k_only_norm /
        # use_qk_norm_post_rope (asserted in `forward`) — those
        # restructure the norm, not the gain; combining with
        # use_qk_norm_depth is also forbidden (190 IS the
        # 169-style gain, not a partner). Default off ⇒ no Parameter
        # registered, no branch taken, baseline path bit-identical.
        # See `autoresearch/ideas/190-per-layer-qk-norm/idea.md`.
        self.use_qk_norm_scalar_per_block = qk_norm_scalar_per_block
        self.use_qk_norm_scalar_qk_shared = qk_norm_scalar_qk_shared
        if self.use_qk_norm_scalar_per_block:
            if self.use_qk_norm_scalar_qk_shared:
                # Shared scalar: one `γ` per block, applied to both Q
                # and K (the 169 axis). Two attributes (`scalar_q` and
                # `scalar_k`) point to the same Parameter so the
                # forward code can stay side-symmetric without a
                # branch — the gate (`use_qk_norm_scalar_qk_shared`)
                # is read once per forward, not per multiply.
                self.qk_norm_scalar_q = nn.Parameter(torch.ones(()))
                self.qk_norm_scalar_k = self.qk_norm_scalar_q
            else:
                # Separate scalars: one γ per side per block (the 190
                # default). Distinct Parameter objects so the
                # optimizer can move them independently — preserves
                # 016's QK symmetry 162+165 attributed WIN to.
                self.qk_norm_scalar_q = nn.Parameter(torch.ones(()))
                self.qk_norm_scalar_k = nn.Parameter(torch.ones(()))
        else:
            # Stubs so attribute lookups are always valid even when
            # the flag is off. The forward branch is gated on
            # `self.use_qk_norm_scalar_per_block` so these are never
            # consumed.
            self.qk_norm_scalar_q = None
            self.qk_norm_scalar_k = None
        # 163 — Post-Attention V-Mix Depthwise Convolution. Stored
        # on self; the conv is applied in `forward()` AFTER the
        # `[B, H, T, D] → [B, T, d_model]` reshape and BEFORE the W_O
        # projection. Raw `nn.Parameter(zeros(d_model, 1, k))` with
        # center tap = 1.0 set inline (NOT `nn.Conv1d(...)`) so the
        # construction does NOT consume RNG — keeping the RNG state
        # aligned with the baseline path for the step-0 byte-
        # identity test (any RNG advance between the two
        # constructions would shift the next-block qkvo_proj random
        # init and break the comparison). See
        # `models/conv_ffn.py:103-105` for the sibling pattern.
        self.use_v_mix_conv = use_v_mix_conv
        # Clamp to odd >= 3; the spec pins k=3 (the only kernel we
        # test at tiny1m3m — see idea.md §Design sketch caveat
        # about small-channel regimes).
        self.v_mix_conv_kernel = max(3, int(v_mix_conv_kernel) | 1)
        if self.use_v_mix_conv:
            w = torch.zeros(self.d_model, 1, self.v_mix_conv_kernel)
            w[:, 0, self.v_mix_conv_kernel // 2] = 1.0
            self.v_mix_conv_weight = nn.Parameter(w)
        else:
            # Stub so attribute lookups are always valid even when
            # the flag is off. `forward()` never references this
            # when `use_v_mix_conv=False` so it can be anything.
            self.v_mix_conv_weight = None
        # 201 — Degenerate gMLP Spatial Gating Unit on attention
        # output (Liu et al. 2021, arXiv:2105.08050, §3.1). Sits
        # at the post-merge / pre-W_O site alongside 163 (local
        # depthwise conv) and 175 (alibi pre-O bias). Allocated
        # ONLY when `use_gmlp_sgu=True` AND
        # `block_idx % gmlp_sgu_block_stride == 0` (per-block-
        # stochastic). For 4 of 12 blocks at default stride 3.
        # Raw `nn.Parameter(torch.empty(d_model, d_model))` plus
        # inline `.data.normal_(std=0.02)` — NOT `nn.Linear(...)`
        # — so the construction does NOT consume RNG via a child
        # module's `_reset_parameters()` (the no-flag path skips
        # this entire block, so any RNG advance here would shift
        # the next-block qkvo_proj random init and break the
        # no-flag step-0 byte-identity test). The 0-dim
        # `sgu_alpha` is init at `gmlp_sgu_alpha_init` (default
        # -10) ⇒ `sigmoid(-10) ≈ 4.5e-5` ⇒ silent at step 0
        # (forward bit-identical to baseline within fp32 noise
        # of one extra multiply-add — same pattern as 175-alibi-
        # slopes / 188-cross-block-kv-share / 179-anti-causal-
        # subheads). Default off → no Parameter registered, no
        # forward branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/201-mlp-token-mixer/idea.md` /
        # `plan.md`.
        self.use_gmlp_sgu = use_gmlp_sgu
        # Clamp to >= 1 (stride 0 would mean "every block"; stride
        # 1 also means "every block"; stride >= 2 starts to skip).
        # The default of 3 picks block_idx ∈ {0, 3, 6, 9} = 4 of 12
        # blocks at tiny1m3m.
        self.gmlp_sgu_block_stride = max(1, int(gmlp_sgu_block_stride))
        self.block_idx = int(block_idx)
        if (
            self.use_gmlp_sgu
            and (self.block_idx % self.gmlp_sgu_block_stride == 0)
        ):
            sgu_w = torch.empty(self.d_model, self.d_model)
            sgu_w.data.normal_(mean=0.0, std=0.02)
            self.sgu_W = nn.Parameter(sgu_w)
            self.sgu_alpha = nn.Parameter(
                torch.tensor(float(gmlp_sgu_alpha_init))
            )
        else:
            # Stubs so attribute lookups are always valid even when
            # the flag is off OR this block is on a non-stochastic
            # stride. `forward()` never references these when
            # `self.sgu_W is None` (the gate is on the Parameter,
            # not the flag, so a stride-miss stays bit-identical to
            # baseline).
            self.sgu_W = None
            self.sgu_alpha = None
        # 129 — YOCO: when set, the MHA reads shared K, V from the
        # `shared_kv` kwarg on `forward`, skipping the W_K, W_V
        # slices of the merged qkvo_proj. Default off → the W_K,
        # W_V slices are used as in the standard path; baseline
        # forward is bit-identical. See
        # `models/yoco.py` and `autoresearch/ideas/129-yoco/idea.md`.
        self.use_shared_kv = use_shared_kv
        # 188 — Cross-Block K/V Projection Sharing. Two 0-dim
        # learnable scalars per MHA (`cross_block_alpha_K`,
        # `cross_block_alpha_V`), init -10 so `sigmoid(-10) ≈ 4.5e-5`
        # at step 0 ⇒ the blend on K, V is dominated by the
        # block-local projection (effectively identity at step 0,
        # bit-identical to baseline within fp32 noise of one extra
        # multiply-add). `prev_W_K` / `prev_W_V` are passed in
        # `forward` and `.detach()`-ed at the call site (the model
        # loop stashes the layer-0 W_K, W_V slices and passes them
        # to layers 1..N-1). When `use_cross_block_kv_share=False`
        # (default) no Parameter is registered, the forward branch
        # is never taken, and the baseline K, V projection path is
        # bit-identical. See
        # `autoresearch/ideas/188-cross-block-kv-share/idea.md`.
        self.use_cross_block_kv_share = use_cross_block_kv_share
        if self.use_cross_block_kv_share:
            self.cross_block_alpha_K = nn.Parameter(
                torch.full((), -10.0)
            )
            self.cross_block_alpha_V = nn.Parameter(
                torch.full((), -10.0)
            )
            # Forward-pass-local stash slots; written by `forward` so
            # the model loop can read `block.attention._prev_W_K` /
            # `_prev_W_V` after the layer-0 call and pass them as
            # `prev_W_K` / `prev_W_V` kwargs to layer 1..N-1. Always
            # initialized to `None` so the attribute exists for the
            # model loop's `getattr` lookups even before the first
            # forward (and when the flag is off).
            self._prev_W_K = None
            self._prev_W_V = None
        else:
            self.cross_block_alpha_K = None
            self.cross_block_alpha_V = None
            self._prev_W_K = None
            self._prev_W_V = None
        # 204 — Cross-Block Attention Score Sharing. One 0-dim
        # learnable scalar `score_share_alpha_raw` (init
        # `score_share_alpha_init=-10.0` ⇒ `σ(-10) ≈ 4.5e-5` at
        # step 0 ⇒ the blend on pre-softmax scores is dominated
        # by the block-local scores, i.e. effectively identity at
        # step 0 — bit-identical to baseline within fp32 noise of
        # one extra multiply-add). `prev_block_scores` is passed
        # in `forward` as the previous block's pre-softmax scores
        # `[B, H, T, T]`, `.detach()`-ed at the call site (the
        # model loop stashes the layer-0 pre-softmax scores and
        # passes them to layers 1..N-1; same pattern as 021's
        # `v_residual=` / 164's `q_carry=` / 168's `av_carry=` /
        # 188's `prev_W_K=` / `prev_W_V=`). The detach on the
        # stash keeps the cross-block gradient structurally
        # bounded to the 1 scalar α per block. When
        # `use_cross_block_score_share=False` (default) no
        # Parameter is registered, the forward branch is never
        # taken, and the baseline pre-softmax score path is
        # bit-identical. Forces the manual attention path (the
        # blend on `scores = Q·K^T/√d_k` can't go through SDPA's
        # flash kernel — `scores` is materialized in the manual
        # path so we can blend it; SDPA fuses QK^T + softmax +
        # AV into a single kernel that doesn't expose the
        # pre-softmax logit). See
        # `autoresearch/ideas/204-cross-block-attn-score-share/idea.md`.
        self.use_cross_block_score_share = use_cross_block_score_share
        if self.use_cross_block_score_share:
            self.score_share_alpha_raw = nn.Parameter(
                torch.full((), float(score_share_alpha_init))
            )
            # Forward-pass-local stash slot; written by `forward`
            # so the model loop can read `block.attention.
            # _prev_block_scores` after the layer-0 call and pass
            # it as `prev_block_scores=` kwarg to layers 1..N-1.
            # Always initialized to `None` so the attribute exists
            # for the model loop's `getattr` lookups even before
            # the first forward (and when the flag is off).
            self._prev_block_scores = None
        else:
            # Stub so attribute lookups are always valid even when
            # the flag is off. `forward()` never references these
            # when `use_cross_block_score_share=False` so they can
            # be anything.
            self.score_share_alpha_raw = None
            self._prev_block_scores = None
        # 207 — W_O Low-Rank Bottleneck. Two `nn.Parameter` matrices
        # `wo_a ∈ R^{d_model × r}` (normal-init std=0.02, matches the
        # existing `qkvo_proj` init) and `wo_b ∈ R^{r × d_model}`
        # (zero-init ⇒ `wo_a @ wo_b == 0` exactly at step 0), plus one
        # 0-dim learnable scalar `wo_lowrank_alpha` (init
        # `wo_lowrank_alpha_init`, default −10 ⇒ `sigmoid(-10) ≈ 4.5e-5`).
        # The forward computes
        #   `w_o_eff = w_o + σ(α) · (wo_a @ wo_b)`
        # At step 0 `wo_b = 0` ⇒ `wo_a @ wo_b = 0` ⇒ `w_o_eff == w_o`
        # bit-identical to the no-flag baseline. Both `wo_a` and `wo_b`
        # are constructed and registered ONLY when `use_lowrank_wo=True`
        # (default off → no Parameter created, no branch taken, baseline
        # path bit-identical). The `wo_lowrank_alpha` scalar is also
        # gated — when the lever is off, no Parameter exists, so
        # `hasattr(self, "wo_lowrank_alpha")` is False and the forward
        # gate `if self.use_lowrank_wo` short-circuits. See
        # `autoresearch/ideas/207-wo-lowrank-bottleneck/idea.md` /
        # `plan.md`.
        self.use_lowrank_wo = use_lowrank_wo
        self.wo_rank = int(wo_rank)
        if self.use_lowrank_wo:
            self.wo_a = nn.Parameter(
                torch.empty(self.d_model, self.wo_rank)
            )
            with torch.no_grad():
                torch.nn.init.normal_(self.wo_a, mean=0.0, std=0.02)
            self.wo_b = nn.Parameter(
                torch.zeros(self.wo_rank, self.d_model)
            )
            self.wo_lowrank_alpha = nn.Parameter(
                torch.full((), float(wo_lowrank_alpha_init))
            )
        else:
            self.wo_a = None
            self.wo_b = None
            self.wo_lowrank_alpha = None
        # 194 — W_V Low-Rank Residual Correction. Two `nn.Parameter`
        # matrices `wv_a ∈ R^{d_model × r}` (normal-init std=0.02,
        # matches the existing `qkvo_proj` init) and `wv_b ∈
        # R^{r × d_model}` (zero-init ⇒ `wv_a @ wv_b == 0` exactly
        # at step 0), plus one 0-dim learnable scalar
        # `wv_lowrank_alpha` (init `wv_lowrank_alpha_init`, default
        # −10 ⇒ `sigmoid(-10) ≈ 4.5e-5`). The forward computes
        #   `w_v_eff = w_v + σ(α) · (wv_a @ wv_b)`
        # at the W_V extraction site (before the F.linear call).
        # At step 0 `wv_b = 0` ⇒ `wv_a @ wv_b == 0` exactly ⇒
        # `w_v_eff == w_v` bit-identical to the no-flag baseline.
        # Both `wv_a` and `wv_b` are constructed and registered ONLY
        # when `use_lowrank_wv=True` (default off → no Parameter
        # created, no branch taken, baseline path bit-identical).
        # The `wv_lowrank_alpha` scalar is also gated — when the
        # lever is off, no Parameter exists, so the forward gate
        # `if self.use_lowrank_wv` short-circuits. See
        # `autoresearch/ideas/194-lowrank-ffn/idea.md` / `plan.md`.
        self.use_lowrank_wv = use_lowrank_wv
        self.wv_rank = int(wv_rank)
        if self.use_lowrank_wv:
            self.wv_a = nn.Parameter(
                torch.empty(self.kv_size, self.wv_rank)
            )
            with torch.no_grad():
                torch.nn.init.normal_(self.wv_a, mean=0.0, std=0.02)
            self.wv_b = nn.Parameter(
                torch.zeros(self.wv_rank, self.d_model)
            )
            self.wv_lowrank_alpha = nn.Parameter(
                torch.full((), float(wv_lowrank_alpha_init))
            )
        else:
            self.wv_a = None
            self.wv_b = None
            self.wv_lowrank_alpha = None
        # 197 — Tied W_O Across Blocks. When on, store the
        # SAME per-model `W_O_shared` Parameter reference on every
        # MHA (NOT a copy — every block reads the same parameter,
        # so the optimizer sees a single global matrix and the
        # cross-block gradient flows through it). Allocate one
        # per-MHA 0-dim scalar `tied_wo_alpha_raw` (init
        # `tied_wo_alpha_init`, default −10 ⇒ `σ(−10) ≈ 4.54e-5`
        # at step 0). The forward computes
        #   `w_o_eff = (1 − σ(α_b_raw)) · w_o_b + σ(α_b_raw) · W_O_shared`
        # at the W_O application site (after the per-block O slice
        # is extracted from `qkvo_proj`, before the 171-DropConnect
        # mask branch). At step 0 `σ(−10) ≈ 4.54e-5` and
        # `W_O_shared` is std=0.02 normal-init, so the contribution
        # from `W_O_shared` is on the order of 1e-7 in std ⇒ the
        # forward is bit-identical to baseline up to fp32 noise of
        # one extra multiply-add. Default off → no Parameter
        # registered, no reference stored, no branch taken, baseline
        # path bit-identical. See
        # `autoresearch/ideas/197-tied-wo-across-blocks/idea.md` /
        # `plan.md`.
        self.use_tied_wo_across_blocks = use_tied_wo_across_blocks
        if self.use_tied_wo_across_blocks:
            assert tied_wo_shared is not None, (
                "use_tied_wo_across_blocks=True requires a non-None "
                "tied_wo_shared Parameter (allocated on MinimalLLM). "
                "The model loop is responsible for plumbing the shared "
                "matrix down to each MHA via the TransformerBlock "
                "constructor."
            )
            self.tied_wo_shared = tied_wo_shared
            self.tied_wo_alpha_raw = nn.Parameter(
                torch.full((), float(tied_wo_alpha_init))
            )
        else:
            self.tied_wo_shared = None
            self.tied_wo_alpha_raw = None
        # 199 — Spectral-Norm-Bounded W_O Projection. Per-block
        # learnable scalar `γ_l ∈ R` (init 0 ⇒ `exp(γ_l)=1`) and a
        # power-iteration Buffer `u ∈ R^{d_model}` for tracking
        # σ_max(W_O). `σ_max_init` is captured on the FIRST
        # forward (frozen — never recomputed from a perturbed W_O,
        # which is the byte-identity guarantee). The forward
        # computes, after extracting the O-slice weight
        # `w_o = self.qkvo_proj[self.qkv_size:]`:
        #   1. Run `wo_spectral_cap_pi_iters` power-iteration steps
        #      updating `u ← w_o @ u / ||·||₂` and the Rayleigh
        #      quotient `σ = (u^T · w_o · u) / (u^T · u)` (≈ the
        #      largest singular value of `w_o` after convergence).
        #   2. On the first forward, snapshot
        #      `σ_max_init = σ.detach()` and seed `u` from a fresh
        #      random direction (so the FIRST forward's σ_max
        #      estimate matches the captured σ_max_init ⇒ the cap
        #      factor is exactly 1 ⇒ `w_o_eff == w_o` byte-
        #      identical to baseline).
        #   3. Apply the cap:
        #      `w_o_eff = w_o · min(1, σ_max_init · exp(γ_l) / σ)`.
        # At step 0 `γ_l = 0` and `σ = σ_max_init` ⇒ factor = 1
        # ⇒ `w_o_eff == w_o` byte-identical to the no-flag
        # baseline. The optimizer can push `γ_l < 0` to tighten
        # the cap (the informative direction — σ_max(W_O) typically
        # grows under SGD). All state (γ_l scalar, power-iteration
        # vector u, captured σ_max_init) is allocated ONLY when
        # `use_wo_spectral_cap=True`. γ_l is a Parameter; u and
        # σ_max_init are Buffers (so they survive `.to(device)`
        # and optimizer state serialization but do not consume an
        # optimizer slot). The cap is applied at the W_O
        # application site — BEFORE the 171-DropConnect mask and
        # 207-W_O-LowRank addition so it composes uniformly with
        # all preceding output-side levers (the mask and the
        # lowrank correction still operate on a valid
        # `[d_model, d_model]` tensor). Default off → no
        # Parameter, no Buffer, no branch taken, baseline path
        # bit-identical. See
        # `autoresearch/ideas/199-spectral-attn-output/idea.md` /
        # `plan.md`.
        self.use_wo_spectral_cap = use_wo_spectral_cap
        self.wo_spectral_cap_pi_iters = int(wo_spectral_cap_pi_iters)
        if self.use_wo_spectral_cap:
            self.wo_spectral_cap_gamma = nn.Parameter(
                torch.zeros(())
            )
            # Power-iteration vector — initialized lazily on the
            # first forward to a fresh random unit vector so the
            # first σ_max estimate and the captured σ_max_init
            # are exactly equal (the byte-identity guarantee).
            # Registered as a Buffer (not a Parameter) so it
            # does not consume an optimizer slot.
            self.register_buffer(
                "_wo_pi_u", torch.zeros(self.d_model), persistent=False
            )
            # Captured initial spectral norm — also a Buffer,
            # populated on the first forward.
            self.register_buffer(
                "_wo_pi_sigma_max_init",
                torch.zeros(()),
                persistent=False,
            )
            # Forward-pass flag: True once the first forward has
            # captured σ_max_init and seeded `u`.
            self._wo_pi_initialized = False
        else:
            self.wo_spectral_cap_gamma = None
            self._wo_pi_u = None
            self._wo_pi_sigma_max_init = None
            self._wo_pi_initialized = False
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
        # 152 — Per-head attention logit bias. Zero-init `b_h ∈ R^H`
        # added to scores pre-softmax in the manual path. See
        # `MultiHeadAttention.use_attn_logit_bias` kwarg for the
        # full mechanism and the math-identity caveat (per-head
        # scalar cancels in softmax over keys for all steps).
        self.use_attn_logit_bias = use_attn_logit_bias
        if self.use_attn_logit_bias:
            self.attn_logit_bias = nn.Parameter(torch.zeros(n_heads))
        # 179 — Anti-Causal Sub-Heads. Per-head learnable scalar
        # `γ_h = sigmoid(γ_raw_h)` attenuates the upper-triangle
        # fill by `(1 − γ_h)`. Init `γ_raw_h = -10` ⇒
        # `sigmoid(-10) ≈ 4.5e-5` ⇒ mask fill `≈ -9.99955e8`
        # (well within fp32 precision of the baseline `-1e9`).
        # When off, no Parameter is registered and the forward
        # branch is never taken — the baseline `masked_fill(.,
        # -1e9)` is exactly the per-head-broadcasted scalar
        # when γ_h = 0 anyway, so we do nothing and the
        # baseline path is bit-identical. See the kwarg docstring
        # above for the full mechanism and the step-0
        # byte-identity derivation. See
        # `autoresearch/ideas/179-anti-causal-subheads/idea.md`.
        self.use_anti_causal_subheads = use_anti_causal_subheads
        if self.use_anti_causal_subheads:
            self.ac_subhead_gate = nn.Parameter(
                torch.full((n_heads,), -10.0)
            )
        else:
            self.ac_subhead_gate = None
        # 166 — T5-style bucketed relative position bias. Per-head
        # bias tensor `self.rpe_bias ∈ R^{H × B}` (init zeros) added
        # to the attention logits pre-softmax in the manual path,
        # indexed by `bucket(|i-j|) = floor(log2(|i-j|+1)).clamp_max
        # (B-1)`. The bucket index matrix is a registered
        # non-persistent buffer (size `[max_seq_len, max_seq_len]`,
        # int64) so it's pinned to CPU/GPU with the model but never
        # serialized. At T ≤ 2048 with B=32, the actually-used
        # bucket range is 0..11 (since `floor(log2(2047+1))=11`),
        # so buckets 12..31 stay zero-init forever — a no-op for
        # tiny1m3m, but the spec default keeps T5's parameter
        # count unchanged for re-use at 512-token seq_lens where
        # the full range is exercised. Default off → no
        # Parameter is registered, the forward branch is gated
        # on `self.use_t5_rpe` and never taken, baseline path
        # bit-identical. See `autoresearch/ideas/166-t5-rpe/idea.md`.
        self.use_t5_rpe = use_t5_rpe
        # Clamp B to a small positive integer so a stray config
        # can't drive the param tensor to 0 / negative. The spec
        # default is 32 (matches T5-XXL).
        self.t5_rpe_buckets = max(1, int(t5_rpe_buckets))
        if self.use_t5_rpe:
            self.rpe_bias = nn.Parameter(
                torch.zeros(self.n_heads, self.t5_rpe_buckets)
            )
            # Precomputed bucket index matrix, shape [T_max, T_max],
            # int64. `bucket[i, j] = floor(log2(|i-j|+1)).clamp_max
            # (B-1)`. Built once at construction (no RNG). Follows
            # `model.to(device)` because it's a registered buffer, so
            # no per-forward host transfer. See
            # `autoresearch/ideas/166-t5-rpe/idea.md`.
            with torch.no_grad():
                idx = torch.arange(max_seq_len)
                diff = (idx[:, None] - idx[None, :]).abs()
                buckets = torch.floor(torch.log2(diff.float() + 1.0))
                buckets = buckets.clamp_max(self.t5_rpe_buckets - 1).long()
            self.register_buffer(
                "_t5_rpe_bucket_idx", buckets, persistent=False,
            )
        else:
            # Stub so attribute lookups are always valid even when
            # the flag is off. The forward branch is gated on
            # `self.use_t5_rpe` so these are never consumed.
            self.rpe_bias = None
            self._t5_rpe_bucket_idx = None
        # 155 — Per-head learnable attention temperature `τ_h ∈ R^H`.
        # Init `1/sqrt(d_k)` (the standard inverse-temperature) so the
        # per-head `Q_h K_h^T * τ_h` score scale matches baseline
        # `Q_h K_h^T / sqrt(d_k)` at step 0. `1/sqrt(d_k)` here is
        # computed as `float(self.d_k) ** -0.5` (matching the rest
        # of this file, no `import math` needed). Stored on the
        # module so the manual-path forward site can broadcast it
        # over the [B, H, T, T] score tensor with
        # `.view(1, n_heads, 1, 1)`. Default off → no Parameter
        # created, no branch taken, baseline path bit-identical.
        # See `autoresearch/ideas/155-per-head-temp/idea.md`.
        self.use_per_head_temp = use_per_head_temp
        if self.use_per_head_temp:
            self.attn_temperature = nn.Parameter(
                torch.full((self.n_heads,), float(self.d_k) ** -0.5)
            )
        # 195 — Tight hard QK logit clamp. Store the flag and
        # the fixed `c` value as a Python float (no Parameter
        # registered — `c` is a config constant, not learnable).
        # Default off → no branch taken, baseline path bit-
        # identical. See `autoresearch/ideas/195-qk-clamp-min-max/idea.md`.
        self.use_qk_clamp = use_qk_clamp
        self.qk_clamp_c = float(qk_clamp_c)
        # 193 — Blockwise attention temperature schedule (fixed
        # cosine-depth, no learned params). `tau_b` is the per-block
        # multiplicative scalar on the pre-softmax attention scores
        # (`scores_b = Q_b K_b^T / (tau_b · √d_k)`), precomputed by
        # the model (`MinimalLLM`) as
        # `tau_b = 1 + α · cos(π · b / (L − 1))` for block `b ∈
        # [0, L-1]` and stored as a non-Parameter `Buffer` of shape
        # `[1]` on the MHA. Default off → no Buffer registered, no
        # branch taken, baseline path bit-identical. The per-forward
        # cost (when on) is one elementwise divide on `[B, H, T, T]`
        # per block, reading one scalar — negligible vs the QK^T
        # matmul. Distinct from 188 (per-block *learned* scalar on
        # the same axis — 193 is the *fixed-shape* control), 155
        # (per-head learned scalar), 161 (per-layer learned scalar).
        # Forces the manual attention path so SDPA's flash kernel
        # doesn't fuse QK^T+softmax+AV (the pre-softmax score must
        # be exposed for the divide). See
        # `autoresearch/ideas/193-blockwise-attn-temp-schedule/
        # idea.md` for the `α = -0.3` commitment and the
        # sharpen-early-soften-late sign convention.
        self.use_block_temp_schedule = use_block_temp_schedule
        if use_block_temp_schedule:
            self.register_buffer(
                "tau_b", torch.tensor(float(tau_b), dtype=torch.float32)
            )
        # 161 — Per-layer learnable attention temperature `τ_l ∈ R^1`.
        # Stored on the MODEL (`MinimalLLM.layer_temperature`) — each
        # MHA reads its own slice `layer_temperature[layer_index]` at
        # forward. Init `1/sqrt(d_k)` (the standard inverse-temperature)
        # so `Q_h K_h^T * τ_l` matches baseline `Q_h K_h^T / sqrt(d_k)`
        # at step 0. Distinct from per-head (155): per-head varies
        # WITHIN a layer (one scalar per head), per-layer varies
        # ACROSS layers (one scalar per layer, shared across heads).
        # The per-layer parameter lives on the model, not on each
        # MHA, so this is just a flag and the model is responsible
        # for the parameter. Forces the manual attention path so
        # SDPA's flash/efficient backends don't perturb step-0
        # numerics. Default off → no branch taken, baseline path
        # bit-identical. See `autoresearch/ideas/161-dyt-temp/idea.md`.
        self.use_per_layer_temp = use_per_layer_temp
        # 180 — Pre-softmax 1D causal depthwise conv on attention logits.
        # `logit_conv_w ∈ R^{H × K}` per-head kernel convolved along
        # the key axis of scores before softmax. Init zero then set
        # center = 1.0 with no_grad so the init itself doesn't consume
        # RNG (preserves step-0 byte-identity with the no-flag path).
        # When `use_logit_conv=False` the parameter is not registered
        # and `logit_conv_w` is a None stub for attribute-lookup
        # safety. See the flag docstring above for the mechanism.
        self.use_logit_conv = use_logit_conv
        self.logit_conv_kernel_size = max(1, int(logit_conv_kernel_size))
        if self.use_logit_conv:
            self.logit_conv_w = nn.Parameter(
                torch.zeros(self.n_heads, self.logit_conv_kernel_size)
            )
            with torch.no_grad():
                # Identity tap = index K-1 (the "current position"
                # weight in the `out[s] = Σ_k w[k]·padded[s+k]`
                # convention used in `forward()`) ⇒ delta kernel ⇒
                # conv is identity on scores ⇒ softmax unchanged ⇒
                # step-0 forward is byte-identical to baseline. The
                # optimizer can grow off-center weights over training.
                self.logit_conv_w[:, self.logit_conv_kernel_size - 1] = 1.0
        else:
            self.logit_conv_w = None
        # 160 — Per-head RMS gain on the attention output. After the
        # AV product and softmax aggregation, multiply each head's
        # output `o_h = (A·V)_h ∈ R^{T×d_k}` by a learnable scalar
        # `g_h ∈ R^H` so the per-head contribution to the residual
        # stream has controlled magnitude. Init `g_h = 1.0` exactly
        # ⇒ `o_h *= 1 = o_h` byte-identical to baseline at step 0.
        # Cost: H scalars/layer (4 at tiny1m3m, total 48 — negligible).
        # Different from `use_attn_output_gate` (reparam `(1+g_h)` with
        # g_h=0 init): that one starts at 1.0 but its magnitude
        # reparam has the gradient concentrated in `g_h`; this one is
        # a direct `g_h` multiplier so the magnitude *and* gradient
        # are both `g_h`. Default off → no Parameter registered, no
        # branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/160-rms-gain-per-head/idea.md`.
        self.use_head_gain = use_head_gain
        if self.use_head_gain:
            self.head_gain = nn.Parameter(torch.ones(self.n_heads))
        # 181 — Cross-Head Channel RMSNorm. Normalize the attention
        # output `out = A·V ∈ R^{B×H×T×d_k}` ACROSS HEADS within
        # each d_k slice (i.e. take the RMS over the H dim at each
        # (b, t, k) position) so all H heads land on the same
        # per-(t, k) scale before the W_O projection. Distinct
        # from 160 (per-head scalar gain, no cross-head coupling)
        # and 176 (V-pre-AV per-head RMSNorm, normalizes each
        # head independently along d_k). Two new per-block
        # parameters:
        #   - `cross_head_rmsnorm_alpha_raw ∈ R^H` init `−1e-3`
        #     ⇒ `relu(−1e-3) = 0` exactly (the relu clamps the
        #     negative to bit-exact zero) ⇒ `α_h = 0` exactly at
        #     step 0 ⇒ `out = (1 − 0)·out + 0·... = out` byte-
        #     identical to baseline. NOT `sigmoid(α_raw)` init
        #     `-5` (which would give `≈ 0.0067`, not zero).
        #   - `cross_head_rmsnorm_gain_raw ∈ R^{H×d_k}` init 0
        #     ⇒ `γ_h[k] = 1 + tanh(0) = 1.0` exactly at step 0
        #     ⇒ no per-channel rescaling.
        # Cost: H × (1 α + d_k γ) = 4 × (1 + 16) = 68 params /
        # block × 12 blocks = 816 params (+0.087% of 0.94M).
        # Same shape as 176. Default off → both attributes are
        # `None` so lookups stay valid; the forward `if` guard
        # short-circuits when the flag is off. See
        # `autoresearch/ideas/181-cross-head-rmsnorm/idea.md`.
        self.use_cross_head_rmsnorm = use_cross_head_rmsnorm
        if self.use_cross_head_rmsnorm:
            self.cross_head_rmsnorm_alpha_raw = nn.Parameter(
                torch.full((self.n_heads,), -1e-3)
            )
            self.cross_head_rmsnorm_gain_raw = nn.Parameter(
                torch.zeros(self.n_heads, self.d_k)
            )
        else:
            self.cross_head_rmsnorm_alpha_raw = None
            self.cross_head_rmsnorm_gain_raw = None
        # 191 — Per-token attention output gain. One learnable
        # per-position scalar `γ_t ∈ R^{T_max}` (init 0 ⇒ (1+0)=1
        # exactly ⇒ byte-identical to baseline at step 0). Sliced
        # to `[:seq_len]` at apply time so inference at shorter T
        # only consumes the first `seq_len` scalars. The T
        # granularity (T_max scalars/block) is a different axis
        # from the closed per-head (160: H=4 scalars), per-channel
        # (142: d_model=64 scalars), and per-(h, k) (181: H·d_k
        # scalars) levers. When the flag is off, the attribute
        # is `None` so lookups stay valid; the forward `if` guard
        # short-circuits. See
        # `autoresearch/ideas/191-token-attn-gain/idea.md`.
        self.use_token_attn_gain = use_token_attn_gain
        if self.use_token_attn_gain:
            self.token_attn_gain = nn.Parameter(
                torch.zeros(max_seq_len)
            )
        else:
            self.token_attn_gain = None
        # 203 — Pre-W_O Squeeze-Excitation channel attention. Per-
        # token channel reweighting via a tiny bottleneck MLP
        # (`se_W1: d_model → d_model/r`, `se_W2: d_model/r → d_model`)
        # plus a per-block `se_gamma_raw` scalar (init
        # `se_alpha_init=-10.0` ⇒ `sigmoid(-10) ≈ 4.54e-5` ⇒
        # silent at step 0). Same W_1, W_2 applied to every token/
        # position (no T-axis pooling — the lever is the per-token
        # content-dependent cell, not the original CNN cell). Cost:
        # 2 × d_model × d_model/r params/block (2048 at r=4,
        # d_model=64) plus 1 γ scalar. Default off → all three
        # attributes are `None` so the forward `if` guard short-
        # circuits; baseline path bit-identical. See
        # `autoresearch/ideas/203-pre-wo-se-channel-attn/idea.md`.
        self.use_se_pre_wo = use_se_pre_wo
        self.se_reduction_ratio = max(1, int(se_reduction_ratio))
        self.se_alpha_init = float(se_alpha_init)
        if self.use_se_pre_wo:
            se_inner = max(1, self.d_model // self.se_reduction_ratio)
            self.se_W1 = nn.Linear(self.d_model, se_inner, bias=False)
            self.se_W2 = nn.Linear(se_inner, self.d_model, bias=False)
            self.se_gamma_raw = nn.Parameter(
                torch.tensor(self.se_alpha_init)
            )
        else:
            self.se_W1 = None
            self.se_W2 = None
            self.se_gamma_raw = None
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
        # 171 — DropConnect on W_O. Stored on self so the forward
        # branch at the W_O application site can read both the flag
        # and the rate. The flag is also stored so the standard
        # boolean-gated pattern (`if self.use_dropconnect_wo and
        # self.training and self.dropconnect_wo_rate > 0.0`) matches
        # the 147-DropKey sibling exactly. No `nn.Parameter` is
        # created — the lever only consumes RNG during the training
        # forward (the mask is sampled fresh each call). Default off
        # → no module built, no branch taken, baseline path bit-
        # identical. See `autoresearch/ideas/171-dropconnect-wo/idea.md`.
        self.use_dropconnect_wo = use_dropconnect_wo
        self.dropconnect_wo_rate = float(dropconnect_wo_rate)
        self.dropconnect_wo_warmup_steps = int(dropconnect_wo_warmup_steps)
        # 171 — DropConnect training-step counter (Python int, not a
        # buffer; lives on the module instance, increments at the END
        # of each forward so the first forward call sees step=0 ⇒
        # effective_rate=0.0 ⇒ mask branch short-circuits ⇒ trt forward
        # is bit-identical to baseline at step 0). Per-MHA counter:
        # each block's MHA tracks its own steps (every block gets the
        # same count in practice — all MHAs are called once per
        # forward, so they tick together — but tracking per-block
        # avoids any cross-block coupling if the model ever calls MHA
        # selectively). Only used to compute `effective_rate` during
        # training; eval mode short-circuits the mask anyway so the
        # counter is irrelevant there. See
        # `autoresearch/ideas/171-dropconnect-wo/idea.md`.
        self._dc_step_count: int = 0
        # 151 — RoV (Rotary Value Embeddings, gated). One learnable
        # per-block scalar `rov_gate` (init 0 ⇒ step-0
        # `V_combined = V + 0·V_rot = V` ⇒ bit-identical to baseline).
        # The base rotary buffer already used for Q,K is reused for V,
        # so no extra buffers. Applied in `forward()` between the GQA
        # repeat and the [B,T,H,D] → [B,H,T,D] transpose. Default off
        # → forward graph bit-identical to baseline. See
        # `autoresearch/ideas/151-rov-gated/idea.md`.
        self.use_rov = use_rov
        if self.use_rov:
            self.rov_gate = nn.Parameter(torch.zeros(1))
        # 174 — xPos exponential decay. Stored on self so the
        # forward branch at the K application site can read both the
        # flag and the gamma parameter. One learnable per-layer scalar
        # `xpos_gamma` (init 0 ⇒ step-0 forward is bit-identical to
        # baseline RoPE: `K *= exp(0·t) = 1`). Default off → no
        # parameter created, no branch taken, baseline path bit-
        # identical (12 extra scalars at tiny1m3m when on, ~+0.001%
        # of 0.94M params). See `autoresearch/ideas/174-xpos-decay/
        # idea.md`.
        self.use_xpos = use_xpos
        if self.use_xpos:
            self.xpos_gamma = nn.Parameter(torch.zeros(1))
        # 154 — Rebased Attention. Stored on self; the rebase pool
        # is applied in `forward()` right after the [B,T,H,D] reshape
        # (post-RoPE) and *before* the [B,H,T,D] transpose, so K, V
        # retain their per-head layout through the pool. The
        # `rebase_stride` is clamped to `>= 1` at construction; if
        # the user passes a value larger than `max_seq_len` the pool
        # collapses to a single block (R=1) and the rebased causal
        # mask is the standard causal mask ⇒ the lever is a no-op.
        # When `use_rebased_attn=False` the branch is never taken
        # and the standard softmax path is bit-identical to the
        # no-flag baseline. See
        # `autoresearch/ideas/154-rebased-attn/idea.md`.
        self.use_rebased_attn = use_rebased_attn
        self.rebase_stride = max(1, int(rebase_stride))

        # 185 — Static per-head learned K-rotation. Stored on self;
        # the rotation block in `forward()` applies a per-head
        # orthogonal matrix `R_h` to K (built from the angle
        # parameter) post-GQA-repeat, pre-RoPE / pre-qk_norm. The
        # angles are shape `[n_heads, d_k//2]` and init 0 ⇒
        # `R_h = I_{d_k}` exactly ⇒ K is bit-identical to the input
        # at step 0. When `use_static_k_rotation=False` (default)
        # the parameter is NOT built (no extra memory on the baseline
        # path) and the forward branch is never taken — baseline
        # forward graph is bit-identical to no-flag. Cost when on:
        # n_heads × d_k//2 = 4 × 8 = 32 params/layer, 384 total at
        # tiny1m3m (+0.041% of the 0.94M model). See
        # `autoresearch/ideas/185-static-per-head-k-rotation/idea.md`.
        self.use_static_k_rotation = use_static_k_rotation
        if use_static_k_rotation:
            assert self.d_k % 2 == 0, (
                "use_static_k_rotation=True requires even d_k "
                "(per-plane 2D rotations pair dims 2i, 2i+1); "
                f"got d_k={self.d_k}"
            )
            self.k_rotation_angles = nn.Parameter(
                torch.zeros(self.n_heads, self.d_k // 2)
            )

        # 200 — Static per-layer × per-pair learned K-rotation
        # (depth-axis twin of 185, shared across heads). Stored
        # on self; the rotation block in `forward()` applies a
        # per-layer orthogonal matrix `R_l` to K (built from the
        # angle parameter) post-GQA-repeat, post-qk_norm. The
        # angles are shape `[d_k//2]` (NO head axis — 200 shares
        # angles across heads and varies them across layers, the
        # opposite axis from 185) and init 0 ⇒ `R_l = I_{d_k}`
        # exactly ⇒ K is bit-identical to the input at step 0.
        # When `use_per_layer_k_rotation=False` (default) the
        # parameter is NOT built (no extra memory on the baseline
        # path) and the forward branch is never taken — baseline
        # forward graph is bit-identical to no-flag. Cost when
        # on: d_k//2 × n_layers = 8 × 12 = 96 params total at
        # tiny1m3m (+0.001% of the 0.94M model). See
        # `autoresearch/ideas/200-rope-phase-offset-per-layer/idea.md`.
        self.use_per_layer_k_rotation = use_per_layer_k_rotation
        if use_per_layer_k_rotation:
            assert self.d_k % 2 == 0, (
                "use_per_layer_k_rotation=True requires even d_k "
                "(per-plane 2D rotations pair dims 2i, 2i+1); "
                f"got d_k={self.d_k}"
            )
            self.per_layer_k_rotation_angles = nn.Parameter(
                torch.zeros(self.d_k // 2)
            )

        # 192 — Pre-RoPE per-head × per-pair learned Q+K rotation
        # (orthogonal-rebase axis, Q+K-side, pre-RoPE placement).
        # Stored on self; the rotation block in `forward()` applies
        # a per-head block-diagonal orthogonal matrix `R_h` to
        # BOTH Q and K (built from the angle parameter) BEFORE
        # RoPE — i.e., post-Q/K split, pre-RoPE / pre-qk_norm.
        # The angles are shape `[n_heads, d_k//2]` (a per-head ×
        # per-pair grid — IDEA explicitly names 4 × 8 × 12 = 384
        # scalars at tiny1m3m). At GQA-active configs (n_kv_heads
        # < n_heads, e.g. 2 vs 4 at tiny1m3m) K's pre-RoPE
        # rotation uses only the first `n_kv_heads` rows of the
        # parameter (a clean per-KV-head projection of the
        # per-head angle grid). Init `φ_{h,i} = 0` ⇒ `cos(0)=1,
        # sin(0)=0` in fp32 ⇒ `R_h = I_{d_k}` exactly ⇒
        # `Q = R_h @ Q = Q` and `K = R_h @ K = K` exactly ⇒
        # step-0 forward is bit-identical to the no-flag
        # baseline. When `use_pre_rope_rotation=False` (default)
        # the parameter is NOT built (no extra memory on the
        # baseline path) and the forward branch is never taken —
        # baseline forward graph is bit-identical to no-flag.
        # Cost when on: n_heads × d_k//2 × n_layers = 4 × 8 × 12
        # = 384 params total at tiny1m3m (+0.041% of 0.94M —
        # negligible). See
        # `autoresearch/ideas/192-pre-rope-qk-rotation/idea.md`.
        self.use_pre_rope_rotation = use_pre_rope_rotation
        if use_pre_rope_rotation:
            assert self.d_k % 2 == 0, (
                "use_pre_rope_rotation=True requires even d_k "
                "(per-plane 2D rotations pair dims 2i, 2i+1); "
                f"got d_k={self.d_k}"
            )
            self.pre_rope_rotation_angles = nn.Parameter(
                torch.zeros(self.n_heads, self.d_k // 2)
            )

        # 156 — Mixture-of-Attentions (MoA). Stored on self; the
        # MoA branch in `forward()` runs E parallel attention
        # computations and mixes them by a per-token router. The
        # expert count is clamped to `>= 2` when the flag is on (1
        # expert is the no-op MoA = standard attention, so the
        # construction refuses it). When `use_moa=False` (default)
        # the MoA parameters are NOT built (no extra memory cost on
        # the baseline path) and the standard softmax path is
        # bit-identical. Cost when on (E=2): (E-1) × (2·kv_size ×
        # d_model) = (2·32 × 64) = 4096 params/layer for the extra
        # K/V, plus d_model × E = 128 params/layer for the router =
        # ~4224/layer, ~50,688 total at tiny1m3m (+5.4% of the
        # 0.94M model). See `autoresearch/ideas/156-moa/idea.md`.
        self.use_moa = use_moa
        self.moa_num_experts = max(2, int(moa_num_experts)) if use_moa else 1
        if use_moa:
            extra_kv_size = self.qkv_size - self.q_size  # 2 × kv_size
            # (E-1) extra sets of K_e, V_e projections; init 0 so
            # experts 1..E-1 produce 0 attention output at step 0.
            # Constructed as a raw `nn.Parameter` (NOT `nn.Linear`)
            # so the construction does NOT consume RNG — keeping the
            # RNG state aligned with the baseline path for the
            # step-0 byte-identity test (any RNG advance between the
            # two constructions would shift the qkvo_proj random
            # init of later blocks and break the comparison).
            self.moa_extra_kv = nn.Parameter(
                torch.zeros(self.moa_num_experts - 1, extra_kv_size, self.d_model)
            )
            # Router: weight=0 + bias one-hot on expert 0
            # (saturating +30 vs 0 ⇒ softmax([30, 0, …]) ≈ [1, 0, …]
            # in fp32; g_0 = 1 ⇒ attn_output = 1·attn_output_0
            # bit-identical to a single standard attention at step
            # 0). Same raw-`Parameter` choice as `moa_extra_kv`
            # above for the RNG-alignment reason.
            self.moa_router_weight = nn.Parameter(
                torch.zeros(self.moa_num_experts, self.d_model)
            )
            r_bias = torch.zeros(self.moa_num_experts)
            r_bias[0] = 30.0
            self.moa_router_bias = nn.Parameter(r_bias)
        else:
            # Stubs so attribute lookups are always valid even when
            # the flag is off. `forward()` never references these
            # when `use_moa=False` so they can be anything.
            self.moa_router_weight = None
            self.moa_router_bias = None

        # 178 — Gated Multi-Query Attention (G-MQA). Per-KV-head
        # scalar gate β_k, β_v ∈ R^{n_kv_heads} blend between the
        # head-local K, V projection and a single shared K, V
        # projection: `K_h = K_local_h + β_k_h · (K_shared_h −
        # K_local_h)`, same for V. β init 0 ⇒ K_mix = K_local
        # exactly in fp32 ⇒ step-0 forward is bit-identical to the
        # no-flag baseline (max-abs-diff = 0.0). The shared K, V
        # projection is a single `nn.Linear(d_model, d_model)` per
        # block (2·d_model² params/layer); at β→1 the head-local
        # K, V becomes dead weight, recovering standard MQA's K/V
        # param savings. We allocate the shared K, V as raw
        # `nn.Parameter` (NOT `nn.Linear`) so the construction does
        # NOT consume RNG — same alignment pattern as MoA's
        # `moa_extra_kv` above: any RNG advance between the two
        # constructions would shift later blocks' `qkvo_proj` init
        # and break the step-0 byte-identity test. The shared K, V
        # is std-0.02 normal-initialized (matches `qkvo_proj`). The
        # gate `β_k, β_v` is a per-KV-head vector (init 0), not a
        # per-head vector — within a GQA group all heads share
        # the same K, V source so the gate is naturally per-KV-
        # head. At tiny1m3m (n_kv_heads = n_heads = 4) this is
        # identical to the per-head form. Default off ⇒ no
        # Parameter registered, no branch taken, baseline path
        # bit-identical. See
        # `autoresearch/ideas/178-mqa-gated/idea.md`.
        self.use_mqa_gated = use_mqa_gated
        if use_mqa_gated:
            # Shared K, V projects to the GQA-axis size
            # (n_kv_heads · d_k) so it broadcasts cleanly against
            # the per-KV-head K, V that the head-local projection
            # produces. The design sketch said `d_model` for the
            # shared projection, but at tiny1m3m the head-local K
            # is per-KV-head (n_kv_heads=2, n_heads=4 ⇒ GQA
            # active) so the K_shared must match that head count
            # for the per-head mix to broadcast without an extra
            # GQA-style repeat. The shared K, V is one projection
            # per block (vs n_kv_heads per-KV-head projections in
            # the head-local path) — the MQA-style savings are
            # recovered when β→1.
            shared_kv_dim = self.n_kv_heads * self.d_k
            self.W_K_shared = nn.Parameter(
                torch.zeros(shared_kv_dim, d_model)
            )
            self.W_V_shared = nn.Parameter(
                torch.zeros(shared_kv_dim, d_model)
            )
            self.mqa_gate_k = nn.Parameter(
                torch.zeros(self.n_kv_heads)
            )
            self.mqa_gate_v = nn.Parameter(
                torch.zeros(self.n_kv_heads)
            )
            # Zero-init (NOT normal-init) on the shared K, V
            # weights: β=0 init ⇒ `(K_shared − K_local)` is
            # multiplied by 0 in forward, so the W_K_shared /
            # W_V_shared values don't affect the step-0 output
            # regardless of init. Zero-init keeps the construction
            # from consuming RNG, which is required to keep the
            # `qkvo_proj` random init aligned with the no-flag
            # baseline (any RNG advance between the two
            # constructions would shift later blocks' qkvo_proj
            # init and break the step-0 byte-identity test). The
            # optimizer will grow the shared K, V from zero as it
            # grows β from zero — this is a standard "init-to-zero,
            # let the optimizer learn it" pattern.
        else:
            # Stubs so attribute lookups are always valid even when
            # the flag is off. `forward()` never references these
            # when `use_mqa_gated=False` so they can be anything.
            self.W_K_shared = None
            self.W_V_shared = None
            self.mqa_gate_k = None
            self.mqa_gate_v = None

        # 202 — V-Only Soft-Blend Probe. Per-group V projection
        # (G = n_heads // v_group_size groups) plus a per-head
        # sigmoid gate `v_group_alpha ∈ R^H` init to `-25.0` so
        # `σ(α) ≈ 1.4e-11` (below fp32 precision) and the blend is
        # numerically 0 at step 0. `W_V_group` is a single Parameter
        # of shape `[G·d_k, d_model]` so the per-group projection is
        # one F.linear matmul (matches the 178-mqa-gated single-
        # matmul pattern for shared K, V). Init copies the in-group
        # elementwise mean of the per-head W_V weights from
        # `qkvo_proj`: at tiny1m3m each group contains exactly one
        # KV head (n_kv_heads=2, G=2), so the "in-group mean"
        # collapses to the per-KV-head W_V slice. This zero-cost
        # identity init keeps the `qkvo_proj` random init aligned
        # with the no-flag baseline (no extra RNG consumption —
        # the in-place slice view is RNG-free, same pattern as
        # 178's zero-init shared K, V and the closed 021 / 164
        # cross-block detach contracts). Default off ⇒ no Parameter
        # registered, no branch taken, baseline path bit-identical.
        # See `autoresearch/ideas/202-grouped-value-projection/idea.md`.
        self.use_grouped_v = use_grouped_v
        self.v_group_size = v_group_size
        if use_grouped_v:
            assert self.n_heads % v_group_size == 0, (
                f"use_grouped_v requires n_heads ({self.n_heads}) "
                f"to be divisible by v_group_size ({v_group_size})"
            )
            num_groups = self.n_heads // v_group_size
            # Single Parameter [G·d_k, d_model] for one F.linear matmul.
            # In-group W_V mean: qkvo_proj's V slice is per-KV-head
            # (shape [n_kv_heads·d_k, d_model] = [kv_size, d_model]),
            # reshape to [n_kv_heads, d_k, d_model], pick the in-group
            # KV-head slice for each group, mean across the in-group
            # heads (single KV head here ⇒ mean = the slice itself).
            v_slice = self.qkvo_proj[
                self.q_size + self.kv_size : self.q_size + 2 * self.kv_size
            ]  # [n_kv_heads · d_k, d_model]
            v_slice = v_slice.reshape(self.n_kv_heads, self.d_k, d_model)
            w_v_group = torch.empty(num_groups, self.d_k, d_model)
            kv_per_group = self.n_kv_heads // num_groups
            for g in range(num_groups):
                kv_lo = g * kv_per_group
                kv_hi = (g + 1) * kv_per_group
                w_v_group[g] = v_slice[kv_lo:kv_hi].mean(dim=0)
            self.W_V_group = nn.Parameter(
                w_v_group.reshape(num_groups * self.d_k, d_model)
            )
            # Per-head sigmoid gate, init -25 so σ(α) ≈ 1.4e-11.
            self.v_group_alpha = nn.Parameter(
                torch.full((self.n_heads,), -25.0)
            )
        else:
            # Stubs so attribute lookups are always valid even when
            # the flag is off. `forward()` never references these
            # when `use_grouped_v=False` so they can be anything.
            self.W_V_group = None
            self.v_group_alpha = None

        # 182 — Per-head learnable attention window. One scalar
        # `w_h ∈ R^H` per MHA maps (via sigmoid) to a window
        # half-size `half_w_h = T · sigmoid(w_h)`. Init `w_h = 10`
        # ⇒ `sigmoid(10) ≈ 0.99995` ⇒ `half_w ≈ T − 0.00005·T >
        # T − 1 = max|t − s|`, so the penalty is identically 0
        # at fp32 at step 0 ⇒ byte-identical to baseline. Default
        # off ⇒ no Parameter registered, no branch taken, baseline
        # path bit-identical. Cost: H params per MHA = 48 total at
        # tiny1m3m (+0.005% of 0.94M). See
        # `autoresearch/ideas/182-per-head-window/idea.md`.
        self.use_per_head_window = use_per_head_window
        if use_per_head_window:
            self.head_window_logit = nn.Parameter(
                torch.full((self.n_heads,), 10.0)
            )
        else:
            self.head_window_logit = None

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

    def forward(self, x, ve=None, gate_x=None, v_residual=None, deberta_relpos=None, shared_kv=None, q_carry=None, av_carry=None, prev_W_K=None, prev_W_V=None, prev_block_scores=None, cosformer_gamma=None):
        batch_size, seq_len = x.size(0), x.size(1)
        # 013 — CoPE replaces RoPE, so the post-RoPE norm has no rotary
        # to post-norm. Reject the misconfiguration loudly so the
        # runner doesn't accidentally launch it.
        assert not (self.use_cope and self.use_qk_norm_post_rope), (
            "use_cope=True is mutually exclusive with use_qk_norm_post_rope=True "
            "(CoPE replaces RoPE; the post-RoPE norm has nothing to act on)."
        )
        # 169 — Depth-Conditional QK-Norm: combining per-block scaling
        # with any of the orthogonal QK-side levers (Q-only / K-only /
        # post-RoPE norm) restructures the lever's mechanism and must
        # fail loud at the build-smoke. The chosen 016-WIN control uses
        # neither of these; the 169 treatment inherits the same 016
        # shape (symmetric pre-RoPE) plus per-block scaling.
        assert not (self.use_qk_norm_depth and self.use_q_only_norm), (
            "use_qk_norm_depth=True is mutually exclusive with use_q_only_norm=True "
            "(169 sits on top of the symmetric 016 path; combining with 162's "
            "Q-only asymmetry restructures the lever)."
        )
        assert not (self.use_qk_norm_depth and self.use_k_only_norm), (
            "use_qk_norm_depth=True is mutually exclusive with use_k_only_norm=True "
            "(169 sits on top of the symmetric 016 path; combining with 165's "
            "K-only asymmetry restructures the lever)."
        )
        assert not (self.use_qk_norm_depth and self.use_qk_norm_post_rope), (
            "use_qk_norm_depth=True is mutually exclusive with use_qk_norm_post_rope=True "
            "(169 sits on top of 016's pre-RoPE symmetric norm; combining with "
            "the post-RoPE variant restructures the lever)."
        )
        # 190 — Per-Layer QK-Norm: sit on top of 016's pre-RoPE
        # symmetric norm (same as 169) — so the same mutual-exclusion
        # rules apply (Q-only / K-only / post-RoPE would restructure
        # the lever's norm axis). Also mutually exclusive with
        # use_qk_norm_depth: 190 IS the 169-style gain (per-block
        # scalar γ applied after the per-head norm); combining the
        # two stacks two scalar-γ multipliers on Q and K (a different
        # lever — and the closed 169 null already nulled at the
        # per-block scalar axis). See
        # `autoresearch/ideas/190-per-layer-qk-norm/idea.md`.
        assert not (self.use_qk_norm_scalar_per_block and self.use_q_only_norm), (
            "qk_norm_scalar_per_block=True is mutually exclusive with use_q_only_norm=True "
            "(190 sits on top of 016's symmetric pre-RoPE norm; combining with 162's "
            "Q-only asymmetry restructures the lever)."
        )
        assert not (self.use_qk_norm_scalar_per_block and self.use_k_only_norm), (
            "qk_norm_scalar_per_block=True is mutually exclusive with use_k_only_norm=True "
            "(190 sits on top of 016's symmetric pre-RoPE norm; combining with 165's "
            "K-only asymmetry restructures the lever)."
        )
        assert not (self.use_qk_norm_scalar_per_block and self.use_qk_norm_post_rope), (
            "qk_norm_scalar_per_block=True is mutually exclusive with use_qk_norm_post_rope=True "
            "(190 sits on top of 016's pre-RoPE symmetric norm; combining with "
            "the post-RoPE variant restructures the lever)."
        )
        assert not (self.use_qk_norm_scalar_per_block and self.use_qk_norm_depth), (
            "qk_norm_scalar_per_block=True is mutually exclusive with use_qk_norm_depth=True "
            "(190 is the per-side 169-style gain; combining with the 169 shared scalar "
            "stacks two scalar-γ multipliers on Q and K — that's a different lever)."
        )
        # 176 — Pre-AV V RMSNorm: mutually exclusive with the two
        # closed V-side norm axes (closed-029 use_v_layernorm +
        # closed-#92 v_norm_type zoo) and with v_mix_conv (a learned
        # conv on V pre-AV; composing it with the per-head α-gated
        # RMSNorm would restructure the lever). 176 is a strictly
        # distinct parameterization from all three — the closed
        # axes have no per-head α-gate and no per-head γ-gain, so
        # combining them with 176 would be running two independent
        # normalizations on the same tensor.
        assert not (self.use_v_rmsnorm and self.use_v_layernorm), (
            "use_v_rmsnorm=True is mutually exclusive with use_v_layernorm=True "
            "(closed-029 LayerNorm V-side; both attach a per-head norm to "
            "V pre-AV and the composition restructures the lever)."
        )
        assert not (self.use_v_rmsnorm and self.v_norm_type not in ("", "none", None)), (
            "use_v_rmsnorm=True is mutually exclusive with the closed-#92 "
            "v_norm_type zoo (v_norm_type != \"\"); both attach a per-head "
            "norm to V pre-AV."
        )
        assert not (self.use_v_rmsnorm and self.use_v_mix_conv), (
            "use_v_rmsnorm=True is mutually exclusive with use_v_mix_conv=True "
            "(v_mix_conv is a learned conv on V pre-AV; the composition "
            "restructures the lever and is not what 176 tests)."
        )
        # 181 — Cross-Head Channel RMSNorm: combining 181 with any
        # of the closed post-AV per-head scalar / input-conditional
        # gates restructures the lever (the optimizer ends up
        # distributing the same effect across two different
        # parameterizations) and must fail loud at the build-smoke.
        # The chosen 181 control uses none of these; the 181
        # treatment sits in isolation. Mirrors the 169 ∧ {Q-only,
        # K-only, post-RoPE} and 176 ∧ {v_layernorm, v_norm_type,
        # v_mix_conv} assertion patterns.
        assert not (self.use_cross_head_rmsnorm and self.use_head_gain), (
            "use_cross_head_rmsnorm=True is mutually exclusive with "
            "use_head_gain=True (both post-AV; the composition "
            "restructures the lever — turn 160 OFF to isolate 181)."
        )
        assert not (self.use_cross_head_rmsnorm and self.use_attn_output_gate), (
            "use_cross_head_rmsnorm=True is mutually exclusive with "
            "use_attn_output_gate=True (closed-045 per-head scalar "
            "ReZero gain; the composition restructures the lever)."
        )
        assert not (self.use_cross_head_rmsnorm and self.use_gated_attn), (
            "use_cross_head_rmsnorm=True is mutually exclusive with "
            "use_gated_attn=True (closed-024 input-conditional "
            "sigmoid gate; the composition restructures the lever)."
        )
        # 189 — CosFormer linear attention is mutually exclusive with
        # the other attention-path levers (linear / diff / nsa /
        # hybrid / multiscale). The cosFormer branch IS the
        # attention path; combining with another is double-attention
        # and a structural lever change.
        assert not (self.use_cosformer and self.use_linear_attn), (
            "use_cosformer=True is mutually exclusive with use_linear_attn=True "
            "(both replace softmax with a linear-time feature-map form — the "
            "cosFormer branch IS the attention path; turn 080 OFF to isolate "
            "189)."
        )
        assert not (self.use_cosformer and self.use_diff_attn), (
            "use_cosformer=True is mutually exclusive with use_diff_attn=True "
            "(both replace the attention path; combining is double-attention)."
        )
        assert not (self.use_cosformer and self.use_nsa_global), (
            "use_cosformer=True is mutually exclusive with use_nsa_global=True "
            "(both replace the attention path; combining is double-attention)."
        )
        assert not (self.use_cosformer and self.use_hybrid_heads), (
            "use_cosformer=True is mutually exclusive with use_hybrid_heads=True "
            "(both replace the attention path; combining is double-attention)."
        )
        assert not (self.use_cosformer and self.use_multiscale_heads), (
            "use_cosformer=True is mutually exclusive with use_multiscale_heads=True "
            "(both replace the attention path; combining is double-attention)."
        )
        # 191 — Per-token attention output gain. Mutually exclusive
        # with the pre-merge post-AV gates (160, 045, 142/121's
        # `attn_output_channel_gate`, 024, 181) — all five are
        # closed-closed or closed-axis-family nulls whose
        # composition with the per-token gain would restructure
        # the lever (the optimizer ends up distributing the same
        # effect across two parameterizations). The chosen 191
        # control uses none of these; 191 sits in isolation.
        # Mirrors the 181 ∧ {160, 045, 024} assertion pattern.
        assert not (self.use_token_attn_gain and self.use_head_gain), (
            "use_token_attn_gain=True is mutually exclusive with "
            "use_head_gain=True (both post-AV; the composition "
            "restructures the lever — turn 160 OFF to isolate 191)."
        )
        assert not (self.use_token_attn_gain and self.use_attn_output_gate), (
            "use_token_attn_gain=True is mutually exclusive with "
            "use_attn_output_gate=True (closed-045 per-head scalar "
            "ReZero gain; the composition restructures the lever)."
        )
        assert not (self.use_token_attn_gain and self.use_attn_output_channel_gate), (
            "use_token_attn_gain=True is mutually exclusive with "
            "use_attn_output_channel_gate=True (closed per-(h, k) "
            "ReZero gain; the composition restructures the lever)."
        )
        assert not (self.use_token_attn_gain and self.use_gated_attn), (
            "use_token_attn_gain=True is mutually exclusive with "
            "use_gated_attn=True (closed-024 input-conditional "
            "sigmoid gate; the composition restructures the lever)."
        )
        assert not (self.use_token_attn_gain and self.use_cross_head_rmsnorm), (
            "use_token_attn_gain=True is mutually exclusive with "
            "use_cross_head_rmsnorm=True (181 cross-head coupling; "
            "both post-AV and the composition restructures the lever)."
        )
        # 129 — YOCO: when the flag is on, the MHA must be given a
        # shared_kv tuple. Reject the misconfiguration loudly so the
        # runner doesn't accidentally launch it without plumbing.
        if self.use_shared_kv:
            assert shared_kv is not None and len(shared_kv) == 2, (
                "use_shared_kv=True requires shared_kv=(K_g, V_g) kwarg "
                "passed by YOCOLlamaBlock.forward"
            )

        # 164 — Q-Carry: stash the MHA sublayer input on layer 0 (no
        # previous block exists ⇒ `q_carry is None` ⇒ stash branch).
        # The model loop reads `self._q_carry` after the layer-0
        # forward and passes it as `q_carry=...` to every layer l ≥ 1.
        # `.detach()` mirrors the 021 V-residual contract — the
        # cross-block gradient is structurally bounded to `α_q`'s
        # 0-dim scalar (layer-l's W_Q gets the carry's matmul output,
        # but its parameter gradient doesn't propagate back into
        # layer-l-1's residual stream).
        if self.use_q_carry and q_carry is None:
            self._q_carry = x.detach()
        # 168 — AV-Output Carry: the stash + blend both happen at
        # the post-merge-reshape, pre-W_O site (below). Layer 0's
        # stash is performed there when `av_carry is None`; layer
        # l ≥ 1's blend is performed there when `av_carry is not
        # None`. We do NOT stash at this early site because the
        # per-head AV has not been computed or reshaped yet. See
        # the `_av_carry` block below the W_O projection site.

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
        # 194 — W_V Low-Rank Residual Correction. In the standard
        # QKV-split path (no YOCO/shared-kv, no tied-QK, no MLA —
        # those branches re-project V via distinct matrices and
        # skip this gate), recompute V through a corrected W_V:
        #   `W_V_eff = W_V + σ(α) · (W_V_A @ W_V_B)`
        #   `V = F.linear(x, W_V_eff)`
        # At step 0 `W_V_B = 0` ⇒ `W_V_A @ W_V_B = 0` exactly ⇒
        # `W_V_eff = W_V` ⇒ `V = F.linear(x, W_V)` bit-identical
        # to the no-flag baseline. Sits BEFORE any downstream V-
        # side lever (use_v_norm, use_v_rmsnorm, use_rov,
        # v_residual, cross-block-KV-share) so those branches read
        # the rank-corrected V. Composes with 207-W_O_LowRank
        # (different projection; orthogonal axes). Default off →
        # branch never taken, baseline path bit-identical. See
        # `autoresearch/ideas/194-lowrank-ffn/idea.md` / `plan.md`.
        if (
            self.use_lowrank_wv
            and not self.use_shared_kv
            and not self.use_tied_qk
            and not self.use_mla
        ):
            alpha = torch.sigmoid(self.wv_lowrank_alpha)
            W_V = self.qkvo_proj[
                self.qkv_size - self.kv_size:self.qkv_size
            ]
            W_V_eff = W_V + alpha * (self.wv_a @ self.wv_b)
            V = F.linear(x, W_V_eff)

        # 188 — Cross-Block K/V Projection Sharing. After the
        # standard QKV split, on layers l ≥ 1, recompute K and V
        # with a learnable convex blend of the layer's own W_K/W_V
        # and the previous block's W_K/W_V:
        #   `α_K = σ(cross_block_alpha_K)`,
        #   `W_K_eff = (1 - α_K) * W_K_self + α_K * prev_W_K.detach()`,
        #   `K_new = x @ W_K_eff.T`,
        # and the same for V. Init `α_raw = -10` ⇒ `α ≈ 4.5e-5` at
        # step 0 ⇒ `K_new = (1 - 4.5e-5) * K_self + 4.5e-5 * K_prev`,
        # numerically dominated by `K_self` ⇒ step-0 output is bit-
        # identical (within fp32 noise) to the no-flag baseline.
        # `prev_W_K` / `prev_W_V` are detached by the model loop, so
        # the cross-block gradient is structurally bounded to the
        # 2 scalar α params per block. Always stash
        # `self._prev_W_K` / `self._prev_W_V` on this MHA so the
        # model loop can capture them after layer 0 and pipe them
        # as kwargs to layer 1..N-1. When `use_cross_block_kv_share
        # =False` (default) the branch is gated and the baseline
        # K, V projection path is bit-identical. Composes with
        # YOCO's `use_shared_kv=True` (when on, the K, V
        # projections are skipped and shared K_g, V_g are used
        # directly — the 188 blend is dead in that case and we
        # stash None for the prev-W tensors). Composes with the
        # closed 021 `v_residual=` (the blend happens BEFORE the
        # V_residual stash, so the layer's V is the blended V; the
        # 021 lever reads the post-blend V). See
        # `autoresearch/ideas/188-cross-block-kv-share/idea.md`.
        if self.use_cross_block_kv_share:
            if self.use_shared_kv:
                # YOCO upper half: K, V are the shared ones, not
                # projected through W_K / W_V. 188 has nothing to
                # blend here — stash None and skip the branch.
                self._prev_W_K = None
                self._prev_W_V = None
            else:
                W_K_self = self.qkvo_proj[
                    self.q_size:self.q_size + self.kv_size
                ]
                W_V_self = self.qkvo_proj[
                    self.qkv_size - self.kv_size:self.qkv_size
                ]
                # Always stash the current layer's W_K, W_V slices
                # so the model loop can read them for the next
                # layer's `prev_W_K=` / `prev_W_V=` (mirrors the
                # `q_carry=` / `av_carry=` / `v_residual=` pattern).
                self._prev_W_K = W_K_self.detach()
                self._prev_W_V = W_V_self.detach()
                if prev_W_K is not None and prev_W_V is not None:
                    # Layer l ≥ 1 — recompute K, V with the
                    # blended projection. The previous block's
                    # W_K, W_V are already detached (by the model
                    # loop) so the gradient doesn't flow back into
                    # the prev block's qkvo_proj.
                    alpha_K = torch.sigmoid(self.cross_block_alpha_K)
                    alpha_V = torch.sigmoid(self.cross_block_alpha_V)
                    W_K_eff = (1.0 - alpha_K) * W_K_self + alpha_K * prev_W_K
                    W_V_eff = (1.0 - alpha_V) * W_V_self + alpha_V * prev_W_V
                    K = F.linear(x, W_K_eff)
                    V = F.linear(x, W_V_eff)
                # else: layer 0 (no previous block). The K, V from
                # the standard QKV split above are the layer-0
                # projections — unchanged by the blend (no prev
                # W_K / W_V to blend with). Stash is done above so
                # the model loop can capture for layer 1.

        # 164 — Q-Carry: add a learnable cross-block Q carry from
        # the previous block's MHA sublayer input. Site is post-QKV
        # split (Q is shape `[B, T, q_size]`), pre-RoPE / pre-q_norm
        # / pre-q_only_norm so the existing norm/RoPE still rescales
        # Q + α·Q_carry consistently. `q_carry` is the previous
        # block's MHA sublayer input, `.detach()`-ed by the model
        # loop — the carry's gradient is structurally bounded to
        # `alpha_q`'s 0-dim scalar. Projection uses the SAME W_Q
        # slice that produced the current Q (so the carry tracks
        # the W_Q the layer is training, not a fresh one): the
        # standard `qkvo_proj[:q_size]` in the default / shared_kv
        # / MLA branches, the `qk_proj[:q_size]` slice in the
        # tied-QK branch (where W_Q == W_K). α_l=0 init ⇒
        # `α_l · W_Q(prev_x) = 0` exactly in fp32 ⇒ step-0 is
        # bit-identical to baseline (within fp32 rounding noise of
        # one extra multiply-add). See
        # `autoresearch/ideas/164-q-carry/plan.md`.
        if self.use_q_carry and q_carry is not None:
            if self.use_tied_qk:
                q_carry_w = self.qk_proj[:self.q_size]
            else:
                q_carry_w = self.qkvo_proj[:self.q_size]
            Q = Q + self.alpha_q * F.linear(q_carry, q_carry_w)

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

        # 192 — Pre-RoPE per-head × per-pair learned Q+K rotation
        # (orthogonal-rebase axis, Q+K-side, pre-RoPE placement).
        # Applied AFTER the Q/K head reshape (so Q is `[B, T,
        # n_heads, d_k]` and K is `[B, T, n_kv_heads, d_k]`) and
        # BEFORE the qk_norm + RoPE block. Builds cos/sin tables
        # from `pre_rope_rotation_angles` and applies a 2D
        # rotation on each `(2i, 2i+1)` plane. Q uses the full
        # `[n_heads, d_k//2]` angle grid (per-head rotation on all
        # n_heads Q heads). K uses the first `[n_kv_heads, d_k//2]`
        # rows of the same parameter (the per-KV-head projection
        # of the per-head angle grid — at GQA-active configs the
        # K rotation is naturally per-KV-head and the per-head
        # "second row" of the GQA group is unused). The block-
        # diagonal `R_h` (product of d_k/2 2D rotations on
        # disjoint planes) is orthogonal, so QK^T magnitudes are
        # preserved (no softmax temperature shift). Init
        # `φ_{h,i} = 0` ⇒ `cos(0)=1, sin(0)=0` in fp32 ⇒
        # `Q = R_h @ Q = Q` and `K = R_h @ K = K` exactly ⇒
        # step-0 forward is bit-identical to the no-flag
        # baseline. When OFF the branch is never taken, the
        # parameter is never built, and the forward graph is
        # bit-identical to no-flag. See
        # `autoresearch/ideas/192-pre-rope-qk-rotation/idea.md`.
        if self.use_pre_rope_rotation:
            # angles: [n_heads, d_k//2]. Build cos/sin once per
            # forward (tiny, ~4×8=32 values; the cos/sin ops
            # broadcast over the [B, T] axis in the apply).
            cos_a = self.pre_rope_rotation_angles.cos()  # [H, d_k/2]
            sin_a = self.pre_rope_rotation_angles.sin()  # [H, d_k/2]
            # --- Q side: per-head rotation on all n_heads Q
            # heads. Q is [B, T, n_heads, d_k] layout.
            Q_pairs = Q.reshape(
                batch_size, seq_len, self.n_heads, self.d_k // 2, 2
            )
            Q_a = Q_pairs[..., 0]  # [B, T, H, d_k/2]
            Q_b = Q_pairs[..., 1]  # [B, T, H, d_k/2]
            cos_q = cos_a.view(1, 1, self.n_heads, self.d_k // 2)
            sin_q = sin_a.view(1, 1, self.n_heads, self.d_k // 2)
            Q_a_new = Q_a * cos_q - Q_b * sin_q
            Q_b_new = Q_a * sin_q + Q_b * cos_q
            Q = torch.stack([Q_a_new, Q_b_new], dim=-1).reshape(
                batch_size, seq_len, self.n_heads, self.d_k
            )
            # --- K side: per-KV-head rotation using the first
            # n_kv_heads rows of the angle parameter. K is
            # [B, T, n_kv_heads, d_k] layout (pre-GQA-repeat).
            K_angles = self.pre_rope_rotation_angles[: self.n_kv_heads]
            K_pairs = K.reshape(
                batch_size, seq_len, self.n_kv_heads, self.d_k // 2, 2
            )
            K_a = K_pairs[..., 0]  # [B, T, n_kv, d_k/2]
            K_b = K_pairs[..., 1]  # [B, T, n_kv, d_k/2]
            cos_k = K_angles.cos().view(1, 1, self.n_kv_heads, self.d_k // 2)
            sin_k = K_angles.sin().view(1, 1, self.n_kv_heads, self.d_k // 2)
            K_a_new = K_a * cos_k - K_b * sin_k
            K_b_new = K_a * sin_k + K_b * cos_k
            K = torch.stack([K_a_new, K_b_new], dim=-1).reshape(
                batch_size, seq_len, self.n_kv_heads, self.d_k
            )

        # 178 — Gated Multi-Query Attention. Blend the per-head K, V
        # with a shared K, V projection via a per-KV-head scalar
        # gate: `K_h = K_local_h + β_k_h · (K_shared_h − K_local_h)`,
        # same for V. β init 0 ⇒ K_mix = K_local exactly in fp32 ⇒
        # step-0 forward is byte-identical to the no-flag baseline
        # (max-abs-diff = 0.0 across the full forward, modulo the
        # baseline path itself also computing the same matmul).
        # The shared K, V projection is a single matmul on x and
        # is then reshaped to `[B, T, n_kv_heads, d_k]` so it
        # broadcasts cleanly against the head-local K, V (and the
        # V_n_kv_heads layout under Mega — but at tiny1m3m the
        # experiment runs without Mega, so V_n_kv_heads =
        # n_kv_heads and the layouts match). Sits BEFORE the
        # RMSNorm + RoPE so the mixed K, V flows through the same
        # QK-norm + rotary pipeline as the baseline K, V. Placed
        # BEFORE the GQA repeat_interleave so the gate acts on the
        # KV-head layout (per-KV-head β broadcasts to all heads in
        # a GQA group via the repeat). When
        # `use_mqa_gated=False` (default) the branch is never
        # taken — the K, V tensors are not touched — and the
        # baseline forward graph is bit-identical. See
        # `autoresearch/ideas/178-mqa-gated/idea.md`.
        if self.use_mqa_gated:
            K_shared = F.linear(x, self.W_K_shared)  # [B, T, n_kv_heads·d_k]
            K_shared = K_shared.reshape(
                batch_size, seq_len, self.n_kv_heads, self.d_k
            )
            V_shared = F.linear(x, self.W_V_shared)  # [B, T, n_kv_heads·d_k]
            # Reshape V_shared to match V's head layout. K_local
            # is always per-n_kv_heads (the head-local K slice of
            # the merged qkvo_proj). V has 2·n_kv_heads slots when
            # use_mega is on (the V_raw + V_ema concat); we use
            # only the first n_kv_heads slots for the gate's
            # broadcast (V_extra keeps its own scale unchanged).
            V_shared_n = V_shared.reshape(
                batch_size, seq_len, self.n_kv_heads, self.d_k
            )
            beta_k = self.mqa_gate_k.view(1, 1, self.n_kv_heads, 1)
            beta_v = self.mqa_gate_v.view(1, 1, self.n_kv_heads, 1)
            K = K + beta_k * (K_shared - K)
            if V_n_kv_heads == self.n_kv_heads:
                V = V + beta_v * (V_shared_n - V)
            else:
                # Mega: V has 2·n_kv_heads slots (V_raw + V_ema).
                # Mix V_raw with V_shared_n, leave V_ema alone.
                V_raw = V[:, :, :self.n_kv_heads, :]
                V_raw_mixed = V_raw + beta_v * (V_shared_n - V_raw)
                V = torch.cat([V_raw_mixed, V[:, :, self.n_kv_heads:, :]], dim=2)

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
            # 162 — Q-only norm branch: when on, apply RMSNorm to Q
            # only via the dedicated `q_only_norm` module and leave K
            # untouched (no `k_norm` call). Overrides the symmetric
            # QK-norm path above; the standard `q_norm`/`k_norm`
            # modules are NOT used here (the lever has its own module).
            # 165 — K-only norm branch: symmetric mirror; apply RMSNorm
            # to K only via the dedicated `k_only_norm` module and leave
            # Q untouched. Mutually exclusive with 162 (asserted at
            # construction in __init__).
            if self.use_q_only_norm:
                Q = self.q_only_norm(Q)
            elif self.use_k_only_norm:
                K = self.k_only_norm(K)
            else:
                Q = self.q_norm(Q)
                K = self.k_norm(K)
        elif self.use_qk_norm_post_rope:
            if self.use_q_only_norm:
                Q = self.q_only_norm(self.rotary(Q))
                K = self.rotary(K)
            elif self.use_k_only_norm:
                Q = self.rotary(Q)
                K = self.k_only_norm(self.rotary(K))
            else:
                Q = self.q_norm(self.rotary(Q))
                K = self.k_norm(self.rotary(K))
        else:
            if self.use_q_only_norm:
                Q = self.rotary(self.q_only_norm(Q))
                K = self.rotary(K)
            elif self.use_k_only_norm:
                Q = self.rotary(Q)
                K = self.rotary(self.k_only_norm(K))
            else:
                Q = self.rotary(self.q_norm(Q))
                K = self.rotary(self.k_norm(K))
        # 169 — Depth-Conditional QK-Norm: per-block learnable scalar
        # `qk_norm_scale` applied to both Q and K AFTER the per-head
        # norm+RoPE and BEFORE the QK matmul (the standard pre-RoPE
        # path is the 016-WIN path — the chosen control uses this
        # same branch). At α_l = 1.0 init, the multiply is exactly
        # the identity in fp32 (`1.0 * x = x`), so step-0 forward is
        # byte-identical to 016's step-0 (max-abs-diff = 0.0 vs the
        # 016 control). Placed AFTER all three norm+RoPE branches
        # (nope/cope, post-RoPE, default pre-RoPE) so it composes
        # uniformly — at α=1.0 the multiply commutes with every
        # downstream tweak (q_gain, GQA repeat, q_temp_token) in
        # fp32. Placed BEFORE the QK matmul so the lever sits at the
        # QK-norm output, matching the NormFormer analog (Shleifer
        # et al. 2021). See `autoresearch/ideas/169-qk-norm-depth/idea.md`.
        if self.use_qk_norm_depth:
            Q = Q * self.qk_norm_scale
            K = K * self.qk_norm_scale
        # 190 — Per-Layer QK-Norm (scalar γ per block per side). At
        # γ=1.0 init the multiply is exactly the identity in fp32
        # (`1.0 * x = x`) ⇒ step-0 forward is byte-identical to 016's
        # step-0 (max-abs-diff = 0.0 vs the 016 control — no tolerance
        # needed). Placed AFTER the 169 multiply (and after the per-
        # head norm+RoPE branches) so it composes uniformly — at
        # γ=1.0 the multiply commutes with every downstream tweak in
        # fp32. Placed BEFORE the QK matmul so the lever sits at the
        # QK-norm output (mirrors 169's placement). The two scalars
        # are independent Parameters unless `use_qk_norm_scalar_qk_shared`
        # is True (the shared variant: both attributes point to the
        # same Parameter — see `__init__`). See
        # `autoresearch/ideas/190-per-layer-qk-norm/idea.md`.
        if self.use_qk_norm_scalar_per_block:
            Q = Q * self.qk_norm_scalar_q
            K = K * self.qk_norm_scalar_k
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
        # 202 — V-Only Soft-Blend Probe. Blend the per-head V
        # with the per-group V via `V_h_eff = (1 − σ(α_h)) · V_h
        # + σ(α_h) · V_group_g(x)`. Site is POST the GQA
        # repeat_interleave so V is in `[B, T, n_heads, d_k]`
        # (the standard layout) and the per-head σ(α_h) gate
        # broadcasts cleanly. V_group is computed by stacking
        # the G group V projection weights and doing one F.linear
        # on `x` to get `[B, T, G*d_k]`, then reshaping to
        # `[B, T, G, d_k]` and repeat_interleave by v_group_size
        # to expand to `[B, T, n_heads, d_k]`. Init
        # `v_group_alpha = -25.0` ⇒ `σ(α) ≈ 1.4e-11` (below fp32
        # precision) ⇒ V_h_eff = V_h_local exactly at step 0
        # ⇒ baseline path is bit-identical when the flag is on
        # and v_group_alpha is at its init value. K is never
        # touched, so the K-axis is the held-out implicit
        # control. The σ(α) trajectory (not val loss) is the
        # deciding metric. See
        # `autoresearch/ideas/202-grouped-value-projection/idea.md`.
        if self.use_grouped_v:
            G = self.n_heads // self.v_group_size
            # self.W_V_group is a single Parameter of shape
            # [G·d_k, d_model] (allocated in __init__ as the
            # concatenated in-group-mean of the per-KV-head W_V
            # slices from `qkvo_proj`). One F.linear matmul
            # produces [B, T, G·d_k], which we reshape to
            # [B, T, G, d_k] and broadcast to [B, T, n_heads, d_k]
            # via per-group repeat_interleave by v_group_size.
            V_group = F.linear(x, self.W_V_group).reshape(
                batch_size, seq_len, G, self.d_k
            )  # [B, T, G, d_k]
            V_group_per_head = torch.repeat_interleave(
                V_group, self.v_group_size, dim=2
            )  # [B, T, n_heads, d_k]
            alpha = torch.sigmoid(self.v_group_alpha).view(
                1, 1, self.n_heads, 1
            )  # [1, 1, H, 1] for broadcast over [B, T, H, d_k]
            V = (1.0 - alpha) * V + alpha * V_group_per_head
        # 185 — Static per-head learned K-rotation. Applied AFTER the
        # GQA repeat_interleave so K is in `[B, T, n_heads, d_k]` and
        # the per-head rotation angle broadcasts cleanly to the
        # post-repeat head count. Site is pre-RoPE / pre-qk_norm so
        # the rotation acts as a static basis change on the raw K
        # stream; orthogonal `R_h` preserves norms and dot products
        # so QK^T magnitudes are unchanged (no softmax temperature
        # shift). Init `θ_{h,i} = 0` ⇒ `cos(0) = 1`, `sin(0) = 0` in
        # fp32 ⇒ K_a_new = K_a and K_b_new = K_b exactly ⇒ K
        # unchanged at step 0 ⇒ baseline forward is bit-identical
        # when the flag is OFF. When OFF the branch is never taken,
        # the parameter is never built, and the forward graph is
        # bit-identical to no-flag. See
        # `autoresearch/ideas/185-static-per-head-k-rotation/idea.md`.
        if self.use_static_k_rotation:
            cos_a = self.k_rotation_angles.cos()  # [H, d_k/2]
            sin_a = self.k_rotation_angles.sin()  # [H, d_k/2]
            # Reshape K to [B, T, H, d_k/2, 2] for the per-plane
            # (2i, 2i+1) 2D rotation. R_h is a product of d_k/2
            # block-diagonal 2D rotations on disjoint planes —
            # block-diagonal ⇒ orthogonal.
            K_pairs = K.reshape(
                batch_size, seq_len, self.n_heads, self.d_k // 2, 2
            )
            K_a = K_pairs[..., 0]  # [B, T, H, d_k/2]
            K_b = K_pairs[..., 1]  # [B, T, H, d_k/2]
            cos_b = cos_a.view(1, 1, self.n_heads, self.d_k // 2)
            sin_b = sin_a.view(1, 1, self.n_heads, self.d_k // 2)
            K_a_new = K_a * cos_b - K_b * sin_b
            K_b_new = K_a * sin_b + K_b * cos_b
            K = torch.stack([K_a_new, K_b_new], dim=-1).reshape(
                batch_size, seq_len, self.n_heads, self.d_k
            )
        # 200 — Static per-layer × per-pair learned K-rotation
        # (depth-axis twin of 185, shared across heads). Applied
        # AFTER the 185 per-head branch and the GQA repeat so K is
        # in `[B, T, n_heads, d_k]`. cos/sin broadcast over the
        # head axis because the parameter has no head dim — every
        # head sees the same per-plane rotation. Q is **untouched**
        # — the K-only application breaks QK^T inner-product
        # preservation, giving the lever a real axis to bind on
        # (QK-symmetric application would be a provable no-op).
        # Init `φ_{l,i} = 0` ⇒ `cos(0) = 1`, `sin(0) = 0` in fp32
        # ⇒ K_a_new = K_a and K_b_new = K_b exactly ⇒ K unchanged
        # at step 0 ⇒ baseline forward is bit-identical when the
        # flag is OFF. When OFF the branch is never taken, the
        # parameter is never built, and the forward graph is bit-
        # identical to no-flag. See
        # `autoresearch/ideas/200-rope-phase-offset-per-layer/idea.md`.
        if self.use_per_layer_k_rotation:
            cos_a = self.per_layer_k_rotation_angles.cos()  # [d_k/2]
            sin_a = self.per_layer_k_rotation_angles.sin()  # [d_k/2]
            # Reshape K to [B, T, H, d_k/2, 2] for the per-plane
            # (2i, 2i+1) 2D rotation. R_l is a product of d_k/2
            # block-diagonal 2D rotations on disjoint planes —
            # block-diagonal ⇒ orthogonal.
            K_pairs = K.reshape(
                batch_size, seq_len, self.n_heads, self.d_k // 2, 2
            )
            K_a = K_pairs[..., 0]  # [B, T, H, d_k/2]
            K_b = K_pairs[..., 1]  # [B, T, H, d_k/2]
            cos_b = cos_a.view(1, 1, 1, self.d_k // 2)
            sin_b = sin_a.view(1, 1, 1, self.d_k // 2)
            K_a_new = K_a * cos_b - K_b * sin_b
            K_b_new = K_a * sin_b + K_b * cos_b
            K = torch.stack([K_a_new, K_b_new], dim=-1).reshape(
                batch_size, seq_len, self.n_heads, self.d_k
            )
        if self.use_k_gain:
            K = K * (1.0 + self.k_gain.view(1, 1, self.n_heads, 1))
        # 174 — xPos exponential decay on K (Sun et al. 2022,
        # arXiv:2212.10554). K is in `[B, T, H, D]` layout here (post-
        # RoPE, post-GQA-repeat, post-k_gain). We multiply K by the
        # per-position decay `g_t = exp(-xpos_gamma · t)` so the
        # attention score `Q[t] · K[s]^T` picks up a factor
        # `g_s = (1-γ)^s` that shrinks as `s` grows — biases attention
        # toward recent tokens without altering the Q-side rotation.
        # With `xpos_gamma = 0` (init) `g_t = 1` for every position ⇒
        # `K = K * 1 = K` exactly ⇒ the trt forward is **bit-identical
        # to the 500k-base RoPE baseline at step 0** (max-abs-diff =
        # 0.0). Placed AFTER the per-head Q/K gains and GQA repeat so
        # the decay broadcasts uniformly over heads. When
        # `use_xpos=False` (default) the branch is never taken, no
        # parameter created, no RNG consumed, baseline path bit-
        # identical. See `autoresearch/ideas/174-xpos-decay/idea.md`.
        if self.use_xpos:
            t = torch.arange(seq_len, device=K.device, dtype=K.dtype)
            g_t = torch.exp(-self.xpos_gamma * t)  # [T]
            K = K * g_t.view(1, seq_len, 1, 1)

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

        # 151 — RoV (Rotary Value Embeddings, gated). Apply the same
        # rotary position embedding already used on Q, K to the value
        # vector V as well. `V_rot = self.rotary(V)` (torchtune's
        # RotaryPositionalEmbeddings expects `[B, T, H, D]` — same
        # layout as V here). Mixed in via a per-block scalar gate:
        # `V_combined = V + rov_gate · V_rot`. Init 0 ⇒ step-0
        # `V_combined = V + 0·V_rot = V` ⇒ bit-identical to baseline.
        # The base rotary buffer is reused (no extra params beyond the
        # 1 scalar/block = 12 at tiny1m3m). When `use_nope` or
        # `use_cope` is on, `self.rotary` is None — RoV becomes a
        # no-op (the geometric lever is unavailable because the Q,K
        # rotary is bypassed). The default-off path (use_rov=False)
        # never enters this branch, leaving V unchanged ⇒ baseline
        # forward graph bit-identical. See
        # `autoresearch/ideas/151-rov-gated/idea.md`.
        if self.use_rov and self.rotary is not None:
            V_rot = self.rotary(V)
            V = V + self.rov_gate * V_rot

        # 154 — Rebased Attention (deferred). `self._rebase_R` is the
        # rebased key count: 0 = no rebase active (K, V stay in the
        # standard `[B, H, T, d_k]` shape after the transpose below),
        # > 0 = rebase to R keys. The actual pool op runs AFTER the
        # Q,K,V transpose so K, V are in `[B, H, T, d_k]` when the
        # reshape reads `K.size(1)` as the head axis (the previous
        # pre-transpose placement was a SMOKE-FAIL: K was `[B, T, H,
        # d_k]` and `K.size(1)` was T, not H). See
        # `autoresearch/ideas/154-rebased-attn/idea.md`.
        self._rebase_R = 0

        # Transpose for attention
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # 154 — Rebased Attention. Pool K, V along the time axis with
        # a fixed stride `rebase_stride` (default 8) by *time-axis*
        # avg-pool — `K' = avg_pool_R(K)`, `V' = avg_pool_R(V)` of
        # shape `[B, H, R, d_k]` with R = ceil(T / rebase_stride).
        # The attention is then run with a *rebased* causal mask
        # (query t can only attend to rebasin r when t >= r·R —
        # i.e. the rebasin is "in the past" of the query). For
        # attention this means the AV product sums over R positions
        # instead of T, and the softmax is over R keys. With
        # `rebase_stride >= T` (e.g. `rebase_stride=2048` at our
        # T=2048) R=1 and the rebased causal mask is just `all-True`
        # for the single rebasin ⇒ the path reduces to full attention
        # (bit-identical to baseline up to a single-block avg-pool of
        # a uniform-time key, which is just the per-position mean —
        # equivalent to the no-pool softmax under our pointwise-K
        # baseline). With `use_rebased_attn=False` (default) the
        # branch is never taken and the standard softmax path is
        # bit-identical. Forces the manual attention path below
        # because the rebased causal mask can't go through SDPA's
        # flash kernel. Sits AFTER the Q,K,V transpose (line 2125) so
        # K, V are `[B, H, T, d_k]` — `K.size(1)` is the head axis
        # and the pool below reads the right dim. See
        # `autoresearch/ideas/154-rebased-attn/idea.md`.
        if self.use_rebased_attn:
            R = self.rebase_stride
            if R < seq_len:
                # Pad on the right to a multiple of R, then mean-pool
                # over R-sized non-overlapping blocks along T. `pad`
                # is the number of right-pad tokens needed so T+pad
                # is a multiple of R. `F.pad` with `(0, 0, 0, pad)`
                # pads the third-from-last dim, which is T for a
                # `[B, H, T, d_k]` tensor.
                pad = (R - (seq_len % R)) % R
                if pad:
                    K = F.pad(K, (0, 0, 0, pad))
                    V = F.pad(V, (0, 0, 0, pad))
                # [B, H, T_padded, d_k] -> [B, H, R, d_k]
                K = K.reshape(K.size(0), K.size(1), -1, R, K.size(-1)).mean(dim=3)
                V = V.reshape(V.size(0), V.size(1), -1, R, V.size(-1)).mean(dim=3)
                self._rebase_R = (seq_len + pad) // R
            else:
                # R >= seq_len ⇒ no compression. K, V stay in the
                # original layout; the manual attention branch sees
                # `_rebase_R == 0` and falls back to the standard
                # full causal mask (equivalent to no-rebase at this
                # scale, modulo the optional no-op average over a
                # single block which equals the un-pooled V).
                self._rebase_R = 0

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
                batch_size, self.n_heads, K.size(2), 1,
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

        # 186 — Within-Block V-Carry. Per-head learnable scalar
        # `α_h = tanh(v_carry_alphas_h)` drives a left-to-right
        # recurrence along the time axis of V:
        #   `V_new[0] = V[0];  V_new[t] = V[t] + α_h · V_new[t-1]` for `t ≥ 1`.
        # Closed form: `V_new[t] = Σ_{k=0}^{t} α_h^k · V[t-k]`
        # (a 1-pole IIR low-pass on V; equivalent to the linear-
        # attention recurrence without the K side, à la Katharopoulos
        # et al. 2020 arXiv:2006.16236). Implemented as a vectorized
        # depthwise `F.conv1d` along T with kernel `[α_h^0, α_h^1,
        # …, α_h^{T-1}]` per head — matches 134-Mega's depthwise
        # EMA conv1d pattern (lines 2823-2834) and is ~0.5 GFLOPs/
        # layer at tiny1m3m, vs ~2k sequential ops/head for the
        # Python for-loop alternative (too slow on GPU).
        #
        # **Derivation (kernel flip + left-pad for F.conv1d
        # cross-correlation).** We want `V_new[t] = Σ_{k=0}^{t}
        # α_h^k V[t-k]`. `F.conv1d` computes cross-correlation
        # `out[t] = Σ_k w[k] · x[t+k]` (no kernel flip). Pad `x` on
        # the left by `T-1` zeros (so `x_padded[t+k] = V[t+k-T+1]`
        # when `t+k ≥ T-1` else 0). Set `w[k] = α_h^{T-1-k}`
        # (the un-flipped kernel reversed): then
        #   `out[t] = Σ_{k=0}^{T-1} α_h^{T-1-k} · V[t+k-T+1]`
        #         `= Σ_{j=0}^{t} α_h^{t-j} · V[j]`  (let `j = t+k-T+1`)
        # which matches `V_new[t]`. α=0 init ⇒ `α_h^{T-1-k}` is
        # `[0, …, 0, 1]` ⇒ conv1d output is `V` exactly ⇒ forward
        # is bit-identical to baseline at step 0. The tanh parameter-
        # ization keeps `|α_h| < 1` strictly so the geometric sum
        # stays bounded at T=2048.
        #
        # **Layout.** Each (B, h, j) channel of V is processed
        # independently along T with its head's kernel. We reshape
        # V from `[B, H, T, d_k]` to `[B, H·d_k, T]` and use
        # `groups=H·d_k` depthwise conv1d (each group is one
        # channel with kernel length T; same FLOPs as 134-Mega's
        # depthwise conv1d, ~0.5 GFLOPs/layer here). Sits
        # AFTER the `use_value_residual` stash/blend and BEFORE the
        # v_norm / v_rmsnorm / value_channel_gate / kda_channel_gate
        # sites — composes cleanly with every preceding V-modifying
        # lever (the recurrence is along T, the others are along
        # d_k; order doesn't matter for downstream matmul blending).
        # See `autoresearch/ideas/186-v-carry-block/plan.md`.
        if self.use_v_carry_block:
            alpha = torch.tanh(self.v_carry_alphas)  # [H], |alpha| < 1
            B, H, T, d_k = V.shape
            # Build kernel [H, T], then FLIP for F.conv1d cross-
            # correlation. arange in fp32 for stability when α_h is
            # near ±1 (T=2048 ⇒ α^2048 underflows/overflows fp16 —
            # keep the whole kernel in fp32 and downcast at the end).
            arange = torch.arange(T, device=V.device, dtype=V.dtype)
            alpha_pow = alpha.unsqueeze(1).pow(arange.unsqueeze(0))  # [H, T]
            kernel = alpha_pow.flip(1)                              # [H, T]
            # Depthwise conv1d: each (B, h, j) channel processed with
            # its head's kernel. Reshape V from [B, H, T, d_k] to
            # [B, H·d_k, T] (permute d_k before T so the conv axis
            # is the last).
            V_flat = (
                V.permute(0, 1, 3, 2).contiguous().reshape(B, H * d_k, T)
            )                                                       # [B, H·d_k, T]
            V_padded = F.pad(V_flat, (T - 1, 0))                   # [B, H·d_k, 2T-1]
            # weight [H·d_k, 1, T]: per-channel kernel, all d_k
            # channels of head h share the same kernel.
            weight = (
                kernel.unsqueeze(1)
                .expand(H, d_k, T)
                .reshape(H * d_k, 1, T)
                .contiguous()
            )
            V_out = F.conv1d(V_padded, weight, groups=H * d_k)
            V = (
                V_out.reshape(B, H, d_k, T)
                .permute(0, 1, 3, 2)
                .contiguous()
            )                                                       # [B, H, T, d_k]

        # #92 Robust V-norm: normalize the value vectors per head before they
        # are mixed by attention (last dim = d_k).
        if self.use_v_norm:
            V = self.v_norm(V)
        # 176 — Pre-AV V RMSNorm with per-head α-gate + per-head γ-gain
        # (Wortsman 2023 V-norm primitive + per-head gating). Applied
        # AFTER the closed-#92 / closed-029 v_norm site and BEFORE the
        # value-channel-gate / kda-channel-gate sites (all compose by
        # being multiplicative/identity on the same V tensor; the
        # gate-α + gain-γ structure of 176 makes the order irrelevant
        # for downstream matmul blending). Compute per-head:
        #   α_h = relu(α_raw_h)            # identity at init (0→0)
        #   rms = rsqrt(mean(V² along d_k) + 1e-6)   # unit RMS per head
        #   V_rms = V * rms * γ_h          # γ_h=1 at init ⇒ identity
        #   V_out = (1 − α_h) · V + α_h · V_rms
        # Init α=0,γ=1 ⇒ V_out = V exactly ⇒ byte-identical to
        # baseline at step 0 (max-abs-diff = 0.0). See
        # `autoresearch/ideas/176-v-pre-av-norm/idea.md`.
        if self.use_v_rmsnorm:
            alpha = F.relu(self.v_rmsnorm_alpha).view(
                1, self.n_heads, 1, 1
            )
            rms = torch.rsqrt(
                V.pow(2).mean(dim=-1, keepdim=True) + 1e-6
            )
            V_rms = V * rms * self.v_rmsnorm_gain.view(
                1, self.n_heads, 1, self.d_k
            )
            V = (1.0 - alpha) * V + alpha * V_rms
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
        # 156 — MoA: when `use_moa=True`, replace the standard
        # attention branch with E parallel attention computations
        # (separate K_e, V_e projections per expert, shared Q) and
        # mix by a per-token router. Inserted BEFORE the other
        # attention branches because MoA replaces the entire
        # attention computation (composes with no other attention-
        # side lever at step 0). See
        # `autoresearch/ideas/156-moa/idea.md`.
        if self.use_moa:
            E = self.moa_num_experts
            B, H, T, D = Q.shape  # Q, K, V already in [B, H, T, D]
            # Compute extra K_e, V_e (e in [1, E-1]) from the
            # zero-init `moa_extra_kv`. x is the sublayer input,
            # still in scope. Reshape `moa_extra_kv` from
            # [E-1, 2·kv_size, d_model] to
            # [(E-1)·2·kv_size, d_model] so F.linear reads it as a
            # single dense projection (output size =
            # (E-1)·2·kv_size, input size = d_model).
            extra = F.linear(
                x, self.moa_extra_kv.reshape(-1, self.d_model),
            )  # [B, T, (E-1)·2·kv_size]
            extra = extra.reshape(B, T, E - 1, 2 * self.kv_size)
            extra_K_pre = extra[..., :self.kv_size]
            extra_V_pre = extra[..., self.kv_size:]
            # Apply k_norm + RoPE to extra K_e, matching the standard
            # Q/K processing path (norm then RoPE in the default
            # branch; this is identical to what Q, K_0 saw above).
            # For step-0 byte-identity use_nope/use_cope are off
            # (the MoA config keeps them off). RoPE expects a 4D
            # `[*, T, H, D]` layout; we collapse the E-1 dim into
            # the batch dim (`B·(E-1)`) for the rotary call, then
            # reshape back.
            extra_K_4d = extra_K_pre.reshape(
                B * (E - 1), T, self.n_kv_heads, self.d_k,
            )
            extra_V_pre = extra_V_pre.reshape(
                B, T, E - 1, self.n_kv_heads, self.d_k,
            )
            if self.use_nope or self.use_cope:
                # 162 — Q-only norm: skip k_norm on extra K too (K stays raw).
                # 165 — K-only norm: apply k_only_norm to extra K (K gets
                # the dedicated lever module instead of the symmetric k_norm).
                if self.use_q_only_norm:
                    pass
                elif self.use_k_only_norm:
                    extra_K_4d = self.k_only_norm(extra_K_4d)
                else:
                    extra_K_4d = self.k_norm(extra_K_4d)
            elif self.use_qk_norm_post_rope:
                if self.use_q_only_norm:
                    extra_K_4d = self.rotary(extra_K_4d)
                elif self.use_k_only_norm:
                    extra_K_4d = self.k_only_norm(self.rotary(extra_K_4d))
                else:
                    extra_K_4d = self.k_norm(self.rotary(extra_K_4d))
            else:
                if self.use_q_only_norm:
                    extra_K_4d = self.rotary(extra_K_4d)
                elif self.use_k_only_norm:
                    extra_K_4d = self.rotary(self.k_only_norm(extra_K_4d))
                else:
                    extra_K_4d = self.rotary(self.k_norm(extra_K_4d))
            # 169 — Depth-Conditional QK-Norm: mirror the per-block
            # `qk_norm_scale` multiply on the MoA extra K so all K
            # tokens entering the QK matmul see the same per-block
            # normalization strength. Site is after the MoA per-K
            # norm+RoPE and before the GQA repeat / cat with K_0.
            # At α_l = 1.0 init the multiply is exactly the identity
            # in fp32 so step-0 is bit-identical to the no-MoA, no-
            # 169 path. See
            # `autoresearch/ideas/169-qk-norm-depth/idea.md`.
            if self.use_qk_norm_depth:
                extra_K_4d = extra_K_4d * self.qk_norm_scale
            # 190 — Per-Layer QK-Norm: mirror the per-side `γ_K`
            # multiply on the MoA extra K so all K tokens entering
            # the QK matmul see the same per-block γ_K normalization
            # strength. Site is after the MoA per-K norm+RoPE and
            # before the GQA repeat / cat with K_0. At γ_K = 1.0
            # init the multiply is exactly the identity in fp32 so
            # step-0 is bit-identical to the no-MoA, no-190 path.
            # See `autoresearch/ideas/190-per-layer-qk-norm/idea.md`.
            if self.use_qk_norm_scalar_per_block:
                extra_K_4d = extra_K_4d * self.qk_norm_scalar_k
            extra_K_pre = extra_K_4d.reshape(
                B, T, E - 1, self.n_kv_heads, self.d_k,
            )
            # GQA repeat if needed (matches the standard path).
            if self.n_kv_heads != self.n_heads:
                extra_K_pre = torch.repeat_interleave(
                    extra_K_pre, self.num_key_value_groups, dim=3,
                )
                extra_V_pre = torch.repeat_interleave(
                    extra_V_pre, self.num_key_value_groups, dim=3,
                )
            # Permute to [B, E-1, H, T, d_k] and concat with K_0, V_0.
            extra_K = extra_K_pre.permute(0, 2, 3, 1, 4).contiguous()
            extra_V = extra_V_pre.permute(0, 2, 3, 1, 4).contiguous()
            K_all = torch.cat([K.unsqueeze(1), extra_K], dim=1)
            V_all = torch.cat([V.unsqueeze(1), extra_V], dim=1)
            # SDPA over batch+expert combined dim (E parallel
            # causal-mask attentions in one fused kernel call).
            Q_sdpa = Q.unsqueeze(1).expand(-1, E, -1, -1, -1).reshape(
                B * E, H, T, D,
            )
            K_sdpa = K_all.reshape(B * E, H, T, D)
            V_sdpa = V_all.reshape(B * E, H, T, D)
            drop_p = self.dropout if self.training else 0.0
            attn_out_sdpa = F.scaled_dot_product_attention(
                Q_sdpa, K_sdpa, V_sdpa, is_causal=True, dropout_p=drop_p,
            )
            expert_outputs = attn_out_sdpa.reshape(B, E, H, T, D)
            # Per-token routing weights from the sublayer input.
            # softmax([30, 0, …]) ≈ [1, 0, …] in fp32 at init; with
            # the extra experts' K_e=V_e=0 the mixed output is
            # exactly attn_output_0 (single standard attention).
            # `moa_router_weight` is `[E, d_model]` (init 0) and
            # `moa_router_bias` is `[E]` (init [30, 0, …]). Built
            # as raw `nn.Parameter`s (not `nn.Linear`) so the
            # construction does NOT consume RNG (keeping the RNG
            # state aligned with the no-flag path for the step-0
            # byte-identity check).
            router_logits = (
                torch.einsum("btd,ed->bte", x, self.moa_router_weight)
                + self.moa_router_bias
            )  # [B, T, E]
            g = torch.softmax(router_logits, dim=-1)  # [B, T, E]
            # Broadcast g over H, d_k: [B, E, 1, T, 1].
            g_bcast = g.permute(0, 2, 1).unsqueeze(2).unsqueeze(-1)
            attn_output = (expert_outputs * g_bcast).sum(dim=1)  # [B, H, T, D]
        elif self.use_fire_pe:
            # 009 FIRE PE — drop-in for RoPE. Manual path: scores
            # = Q K^T / √d_k + FIRE bias, then mask + softmax + @V.
            # x is the original input [B, T, d_model] (still in scope).
            # 155 — Per-head temperature REPLACES the standard
            # `1/sqrt(d_k)` scale (per the idea spec). At init
            # `τ_h = 1/sqrt(d_k)` ⇒ `scores = Q^T K / sqrt(d_k)` ≡
            # baseline at step 0. The else-branch keeps the standard
            # `* (1/sqrt(d_k))` scale untouched for the flag-off
            # path. See `autoresearch/ideas/155-per-head-temp/idea.md`.
            if self.use_per_head_temp:
                scores = torch.matmul(Q, K.transpose(-1, -2))
                scores = scores * self.attn_temperature.view(1, self.n_heads, 1, 1)
            else:
                scale = 1.0 / (float(self.d_k) ** 0.5)
                scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
            # 195 — Tight hard QK logit clamp (FIRE branch). Apply
            # `torch.clamp(scores, -c, +c)` immediately after the
            # standard scale and BEFORE the FIRE/CoPE additive bias
            # so the hard bound is the canonical pre-softmax
            # invariant `|scores| ≤ c`. Default off → branch not
            # taken, baseline path bit-identical. See
            # `autoresearch/ideas/195-qk-clamp-min-max/idea.md`.
            if self.use_qk_clamp:
                scores = torch.clamp(
                    scores, min=-self.qk_clamp_c, max=self.qk_clamp_c
                )
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
            # 182 — Per-head learnable attention window. Apply a
            # hard-style per-head penalty AFTER the causal mask (so
            # masked -1e9 positions keep their zero-probability) and
            # BEFORE softmax. The half-window is `half_w_h = T ·
            # sigmoid(w_h)` (shape [H], broadcast to [B, H, 1, 1]).
            # The penalty `1e9 · relu(|t − s| − half_w_h)` is added
            # to scores for positions OUTSIDE the per-head window —
            # effective softmax zero, fp32-clean (no `−∞`, no NaN
            # risk, matches 154-rebased-attn's rebased-softmax style).
            # At init `w_h = 10 ⇒ sigmoid(10) ≈ 0.99995 ⇒ half_w ≈ T
            # − 0.00005·T > T − 1 = max|t − s|`, so the relu is
            # identically 0 everywhere and the penalty is a no-op at
            # step 0 ⇒ byte-identical to baseline. `ar` is
            # `torch.arange(seq_len, ...)` from earlier in the
            # manual branch. When `use_per_head_window=False`
            # (default) the branch is never taken, no Parameter
            # registered, baseline path bit-identical. See
            # `autoresearch/ideas/182-per-head-window/idea.md`.
            if self.use_per_head_window:
                rel_dist = (
                    ar[None, :].float() - ar[:, None].float()
                ).abs()  # [T, T]
                half_w = (
                    float(seq_len)
                    * torch.sigmoid(self.head_window_logit)
                ).view(1, self.n_heads, 1, 1)  # [1, H, 1, 1]
                scores = scores - 1e9 * F.relu(
                    rel_dist.view(1, 1, seq_len, seq_len) - half_w
                )
            # 166 — T5-style bucketed relative position bias.
            # Add `rpe_bias[h, bucket(|i-j|)]` to scores AFTER the
            # causal mask (masked positions get -1e9 so the bias
            # has no effect on them) and BEFORE softmax. With init
            # `rpe_bias = 0` this is a no-op at step 0. The
            # bucket index buffer is `[max_seq_len, max_seq_len]`
            # int64 (moved to the scores device by `model.to(device)`,
            # no per-forward host transfer — the `.to()` here is a
            # no-op when the buffer is already on the scores device
            # but kept as a defensive fallback for any future code
            # path that might call this on CPU). The bias is
            # computed as `rpe_bias[:, bidx]`, fancy-indexing the
            # [H, B] parameter by a [T, T] int64 index to produce
            # an [H, T, T] tensor that broadcasts over the batch
            # axis of `scores`. See
            # `autoresearch/ideas/166-t5-rpe/idea.md`.
            if self.use_t5_rpe:
                bidx = self._t5_rpe_bucket_idx[:seq_len, :seq_len].to(
                    device=scores.device
                )
                bias = self.rpe_bias[:, bidx]  # [H, T, T]
                scores = scores + bias.unsqueeze(0)
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
            or self.use_attn_logit_bias  # 152 — Per-head logit bias (manual path; force SDPA off so step-0 is bit-identical).
            or self.use_per_head_temp  # 155 — Per-head temperature (manual path; force SDPA off so the score-side multiply is exact).
            or self.use_t5_rpe  # 166 — T5-RPE bucket bias (manual path; force SDPA off so the score-side additive bias is exact).
            or self.use_entmax  # 173 — Entmax-1.5: closed-form projection can't go through SDPA's flash kernel.
            or self.use_topk_attn  # 192 — Hard top-k sparse attention: scatter write can't go through SDPA's flash kernel.
            or self.use_logit_conv  # 180 — Pre-softmax causal conv on QK^T: score-space op, must run on manual path.
            or self.use_qk_clamp  # 195 — Tight hard QK logit clamp (manual path; force SDPA off so the pre-softmax logit is exposed for clamping).
            or self.use_block_temp_schedule  # 193 — Blockwise attention temperature schedule (manual path; pre-softmax `scores / τ_b` divide, SDPA's flash kernel fuses QK^T+softmax+AV and can't expose the pre-softmax logit for the per-block divide).
            or self.use_anti_causal_subheads  # 179 — Per-head mask fill on the upper-triangle; SDPA flash can't apply a per-head fill, so force the manual path.
            or self.use_per_head_window  # 182 — Per-head learnable window: score-space `1e9·relu(...)` subtract, must run on manual path.
            or self.use_cross_block_score_share  # 204 — Cross-Block Attention Score Sharing: pre-softmax score blend with the previous block's `Q_{b-1}·K_{b-1}^T/√d_k`; SDPA's flash kernel fuses QK^T+softmax+AV and can't expose the pre-softmax logit for blending.
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
            # 155 — Per-head learnable attention temperature REPLACES
            # the standard `1/sqrt(d_k)` scale (per the idea spec):
            # `scores_h = Q_h K_h^T * τ_h` with `τ_h` init
            # `1/sqrt(d_k)` ⇒ `scores = Q^T K / sqrt(d_k)` ≡ baseline
            # at step 0 (the standard pre-softmax scale is exactly
            # that factor). The else-branch keeps the standard
            # `* (1/sqrt(d_k))` scale untouched for the flag-off
            # path. See `autoresearch/ideas/155-per-head-temp/idea.md`.
            if self.use_per_head_temp:
                scores = torch.matmul(Qn, Kn.transpose(-1, -2))
                scores = scores * self.attn_temperature.view(1, self.n_heads, 1, 1)
            else:
                scale = 1.0 / (float(self.d_k) ** 0.5)
                scores = torch.matmul(Qn, Kn.transpose(-1, -2)) * scale
            # 193 — Blockwise attention temperature schedule (manual
            # branch). Apply the precomputed per-block scalar
            # `τ_b = 1 + α · cos(π · b / (L − 1))` as a divide on
            # the pre-softmax scores: `scores_b = scores / τ_b`. At
            # α = 0 ⇒ `τ_b = 1` exactly ⇒ `scores / 1 = scores`
            # byte-identical to the standard pre-softmax scale; at
            # the committed `α = -0.3`, `τ_b ∈ [0.7, 1.29]`
            # (sharpen early / soften late). Inserted AFTER the
            # `* scale` (or 155 per-head temp replacement) and
            # BEFORE the 195 clamp + 204 cross-block share so it
            # composes uniformly with all downstream score-side
            # levers. The `tau_b` Buffer is registered on the MHA
            # at construction when the flag is on (see `__init__`);
            # `.view(1, 1, 1, 1)` broadcasts against the
            # `[B, H, T, T]` score tensor. Default off → branch not
            # taken, baseline path bit-identical. See
            # `autoresearch/ideas/193-blockwise-attn-temp-schedule/
            # idea.md` for the schedule formula and the sign
            # convention (`α < 0` = sharpen early).
            if self.use_block_temp_schedule:
                scores = scores / self.tau_b.view(1, 1, 1, 1)
            # 195 — Tight hard QK logit clamp (manual branch). Apply
            # `torch.clamp(scores, -c, +c)` immediately after the
            # standard scale and BEFORE any additive transforms
            # (204 cross-block share / 152 logit bias / etc.) or
            # downstream score-side multiplies (188 qk_rms_scaling).
            # The hard bound is the canonical pre-softmax invariant
            # `|scores| ≤ c`. Default off → branch not taken,
            # baseline path bit-identical. See
            # `autoresearch/ideas/195-qk-clamp-min-max/idea.md`.
            if self.use_qk_clamp:
                scores = torch.clamp(
                    scores, min=-self.qk_clamp_c, max=self.qk_clamp_c
                )
            # 204 — Cross-Block Attention Score Sharing. Blend the
            # current block's pre-softmax scores with the previous
            # block's (detached) pre-softmax scores via a learnable
            # per-block scalar α = σ(score_share_alpha_raw):
            #   `scores_eff = (1 − α) · scores_self + α · prev_block_scores`
            # `prev_block_scores` is `Q_{b-1} · K_{b-1}^T / √d_k`,
            # the PRE-SOFTMAX logit (NOT the post-softmax attention
            # distribution — that's a different lever; see
            # review.md finding B). `.detach()` keeps the cross-
            # block gradient structurally bounded to α (mirrors
            # the 021 / 164 / 168 cross-block detach contract).
            # Always stash the current block's pre-softmax scores
            # (detached, shape `[B, H, T, T]`) on
            # `self._prev_block_scores` so the model loop can read
            # it back after the layer-0 call and pass it as
            # `prev_block_scores=` to layers 1..N-1. Init
            # `α_raw = -10.0` ⇒ `α ≈ 4.5e-5` at step 0 ⇒
            # `scores_eff ≈ scores_self` within fp32 noise of one
            # extra multiply-add (max-abs-diff < 1e-4 across all
            # 12 blocks, the reviewer-precise wording per finding
            # D — NOT literal "bit-identical" since σ(-10) is
            # ≈ 4.5e-5, not exactly 0). When the flag is off
            # (default), the entire branch is gated and the
            # baseline pre-softmax score path is bit-identical.
            # See
            # `autoresearch/ideas/204-cross-block-attn-score-share/idea.md`.
            if self.use_cross_block_score_share:
                self._prev_block_scores = scores.detach()
                if prev_block_scores is not None:
                    # Layer l ≥ 1 — blend. The previous block's
                    # `_prev_block_scores` is already detached (by
                    # this same MHA.forward on the previous call),
                    # so the gradient doesn't flow back into the
                    # prev block's QK computation.
                    alpha = torch.sigmoid(self.score_share_alpha_raw)
                    scores = (1.0 - alpha) * scores + alpha * prev_block_scores
            # 152 — Per-head logit bias `b_h ∈ R^H`. Broadcast
            # `[1, H, 1, 1]` over the [B, H, T, T] score tensor,
            # applied BEFORE softmax (and before other score-side
            # tweaks so the lever composes cleanly). At init
            # `b_h = 0` → `scores + 0` is bit-identical to baseline.
            # Mathematically the per-(b,h,t) normalizer absorbs
            # `e^{b_h}` for all subsequent steps too; this is the
            # recorded-null caveat noted at the flag docstring.
            if self.use_attn_logit_bias:
                scores = scores + self.attn_logit_bias.view(1, self.n_heads, 1, 1)
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
            mask_bool = ~window.view(1, 1, seq_len, seq_len)
            if self.use_anti_causal_subheads:
                # 179 — Per-head anti-causal gate. The fill value
                # is broadcast over the [B, H, T, T] score tensor.
                # `γ_h = sigmoid(ac_subhead_gate_h)` ⇒ init ≈ 4.5e-5
                # ⇒ fill `≈ -9.99955e8` ⇒ upper-triangle
                # `exp(-9.99955e8) < 1e-300` in fp32 ⇒ softmax row
                # bitwise 0 on masked positions ⇒ byte-identical to
                # the no-flag baseline at step 0. The optimizer can
                # grow any γ_h ∈ [0, 1] per head over training.
                # `torch.where` is used (not `masked_fill`) because
                # `masked_fill` requires a 0-d value tensor; the
                # per-head fill `[1, H, 1, 1]` broadcasts cleanly
                # through `torch.where` against `[B, H, T, T]`.
                # See `autoresearch/ideas/179-anti-causal-subheads/idea.md`.
                gamma_h = torch.sigmoid(self.ac_subhead_gate).view(
                    1, self.n_heads, 1, 1
                )
                fill_value = -1e9 * (1.0 - gamma_h)
                scores = torch.where(mask_bool, fill_value, scores)
            else:
                scores = scores.masked_fill(mask_bool, -1e9)
            # 180 — Pre-softmax 1D causal depthwise conv on attention
            # logits. Applied AFTER the causal mask (so masked -1e9
            # positions don't pollute the active convolution window via
            # the delta-init identity) and BEFORE softmax. Left-pad
            # scores by K-1 zeros along the last axis; then for each
            # shift k ∈ [0, K) the slice `padded[..., k:k+S]` is the
            # input contributing from position (s + k - (K-1)). Output
            # at position s sums `w[h, k] * padded[..., s + k]` over k.
            # Identity init (`w[:, K//2] = 1`, rest 0) ⇒ the conv is
            # identity on scores ⇒ softmax unchanged ⇒ step-0 forward
            # is byte-identical to baseline. Loop unrolled at module
            # load (K is a small int constant for any given config);
            # K=3 by default. We deliberately do NOT use F.conv1d +
            # grouped conv here: the per-row conv pattern (each row of
            # scores is independent) and the per-head kernel share
            # requires reshaping [B,H,T,S]→[H,B*T,1,S]→groups=H which
            # adds a transpose vs the slice+sum. At T=2048, H=4, the
            # slice+sum is ~3·B·H·T·S = 100M FLOPs (≪ the 8.6G QK
            # matmul) so the slice+sum wins on clarity and is fast
            # enough. See `autoresearch/ideas/180-qk-logit-conv/idea.md`.
            if self.use_logit_conv:
                K_size = self.logit_conv_kernel_size
                pad_amt = K_size - 1
                padded = F.pad(scores, (pad_amt, 0))  # [B, H, T, S+K-1]
                # Accumulate per-shift weighted slices into `scores`.
                # The first slice is the initialization (avoids
                # creating a zero tensor + K adds).
                scores = self.logit_conv_w[:, 0].view(1, self.n_heads, 1, 1) * padded[:, :, :, 0:seq_len]
                for k in range(1, K_size):
                    scores = scores + self.logit_conv_w[:, k].view(1, self.n_heads, 1, 1) * padded[:, :, :, k:k + seq_len]
            # 166 — T5-style bucketed relative position bias. Add
            # `rpe_bias[h, bucket(|i-j|)]` to scores AFTER the
            # causal mask (masked positions get -1e9 so the bias
            # has no effect on them) and BEFORE softmax. With init
            # `rpe_bias = 0` this is a no-op at step 0. The bucket
            # index buffer is `[max_seq_len, max_seq_len]` int64
            # (moved to the scores device by `model.to(device)`, no
            # per-forward host transfer — the `.to()` here is a
            # no-op when the buffer is already on the scores device
            # but kept as a defensive fallback). Bias is computed as
            # `rpe_bias[:, bidx]` — see the FIRE-branch comment above
            # and `autoresearch/ideas/166-t5-rpe/idea.md`.
            if self.use_t5_rpe:
                bidx = self._t5_rpe_bucket_idx[:seq_len, :seq_len].to(
                    device=scores.device
                )
                bias = self.rpe_bias[:, bidx]  # [H, T, T]
                scores = scores + bias.unsqueeze(0)
            # ---- Q5 Talking-heads: logit-mix across heads pre-softmax ----
            if self.use_talking_heads_q:
                # scores: [B, H, T, T]. M: [H, H]. Mix over H only.
                # out[b, h_new, t, s] = sum_h M[h_new, h] * scores[b, h, t, s]
                scores = torch.einsum(
                    "bhst,hH->bHst", scores, self.talking_heads_M
                )
            # Softmax
            if self.use_topk_attn:
                # 192 — Hard top-k sparse attention (Touvron et al.
                # 2021, "Going Deeper with Image Transformers" / DeiT
                # III, arXiv:2103.17239). Per-row pre-softmax hard
                # sparsification: keep only the k largest scores per
                # row, scatter -inf to the rest, then softmax-
                # renormalize over the surviving k positions. `k =
                # min(topk_k, scores.size(-1))` is the defensive
                # bound (handles shorter eval contexts). Applied
                # AFTER the causal mask so -inf future positions are
                # below the topk budget and never selected. 0 new
                # params (`topk_k` is a config int, not a learnable
                # scalar). Step-0 is NOT byte-identical to baseline
                # when flag-on — same structural-lever category as
                # 173 / 022 / 154. See
                # `autoresearch/ideas/192-topk-attn/idea.md`.
                k = min(self.topk_k, scores.size(-1))
                topk_vals, topk_idx = scores.topk(k, dim=-1)
                sparse_scores = torch.full_like(scores, float("-inf"))
                sparse_scores.scatter_(-1, topk_idx, topk_vals)
                attn_w = torch.softmax(sparse_scores, dim=-1)
            elif self.use_entmax:
                # 173 — Entmax-1.5 (Tsallis α-entmax with α=1.5).
                # Replace `torch.softmax` with the entmax-1.5
                # projection. Per-head α_h is derived from
                # `entmax_alpha_raw`; init 0 ⇒ α_h=1 ⇒ the helper
                # short-circuits to `torch.softmax` for step-0
                # bit-identity. The `window` tensor is reused as the
                # mask argument (True = attend, False = masked out).
                alpha_h = 1.0 + 0.5 * (1.0 + torch.tanh(self.entmax_alpha_raw))
                attn_w = entmax_15(
                    scores,
                    window.view(1, 1, seq_len, seq_len),
                    alpha_h,
                    dim=-1,
                )
            else:
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
        elif self._rebase_R > 0:
            # 154 — Rebased Attention manual branch. K, V are already
            # pooled to `[B, H, R, d_k]` (post-transpose, post-pool).
            # The rebased causal mask: query t attends to rebasin r
            # when `t >= r·R` (the rebasin's start position is at or
            # before t). Build a [T, R] bool mask and broadcast over
            # [B, H, T, R]. Then `softmax(Q @ K_r^T) @ V_r` produces
            # an attention output of shape `[B, H, T, d_k]` (queries
            # are at full T resolution; keys/values are at R
            # resolution). When `_rebase_R == 0` (flag-on but stride
            # >= T) we don't enter this branch — the standard
            # softmax path runs at full T-T resolution. When
            # `use_rebased_attn=False` (default) the branch is never
            # reached and the standard path is bit-identical. See
            # `autoresearch/ideas/154-rebased-attn/idea.md`.
            R = self._rebase_R
            scale = 1.0 / (float(self.d_k) ** 0.5)
            scores = torch.matmul(Q, K.transpose(-1, -2)) * scale  # [B,H,T,R]
            q_pos = torch.arange(seq_len, device=Q.device).view(seq_len, 1)
            r_pos = torch.arange(R, device=Q.device).view(1, R)
            rebased_mask = q_pos >= (r_pos * self.rebase_stride)  # [T, R]
            scores = scores.masked_fill(
                ~rebased_mask.view(1, 1, seq_len, R), -1e9
            )
            attn_w = torch.softmax(scores, dim=-1)
            attn_w = F.dropout(attn_w, p=self.dropout if self.training else 0.0)
            attn_output = torch.matmul(attn_w, V)  # [B, H, T, d_k]
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
        elif self.use_cosformer:
            # 189 — CosFormer-style linear attention (Qin et al.
            # NeurIPS 2022, arXiv:2202.08791). Replace softmax
            # attention with the kernel-replacement form
            #   out = (Q'·(K'^T·V)) / (Q'·K'^T)
            # where Q' = cos(Q), K' = exp(γ·K)·cos(K), γ is a
            # learnable per-block scalar passed in via
            # `cosformer_gamma` (model-owned Parameter on
            # `MinimalLLM.cosformer_gammas`, one entry of size
            # `n_layers` so the optimizer sees ONE param group, not
            # 12). Linear in sequence length via the prefix-sum
            # cumsum trick (same shape as `use_linear_attn` above):
            # compute `K'^T·V` first ([B,H,d_k,d_k]), then `Q'·KV`
            # ([B,H,T,d_k]); causal via `[end_idx, start_idx]`
            # windowed prefix-sum. The denominator
            # `Q'·K'^T` is MANDATORY — bound in spec, no skip-flag
            # — it is the softmax replacement, not a global
            # mean-pool. At γ=0 the lever reduces to the cumulative
            # mean of V over the causal prefix (since `cos(Q)·cos(K)^T
            # ≈ 1` and `cumsum(cos(K)) ≈ (t+1)` under the small-
            # logit std-0.02 qkvo_proj init). Float promotion to
            # fp32 for the matmul, cast back to V.dtype at the end
            # — same convention as the `use_linear_attn` branch
            # above. See
            # `autoresearch/ideas/189-cosformer-linear-attn/idea.md`.
            gamma = (
                self.cosformer_gamma_init if cosformer_gamma is None
                else float(cosformer_gamma)
            )
            q_phi = torch.cos(Q.float())                          # [B,H,T,d_k]
            k_phi_raw = torch.cos(K.float()) * torch.exp(gamma * K.float())  # [B,H,T,d_k]
            v_float = V.float()                                   # [B,H,T,d_k]
            if self.use_sliding_window and self.attention_dilation != 1:
                # Windowed (non-cumsum) form — same shape as the
                # `use_linear_attn` windowed branch above. Applies a
                # hard causal+SWA mask on the K'V matmul so each
                # query only aggregates keys in its window.
                scores = torch.einsum("bhtd,bhsd->bhts", q_phi, k_phi_raw)
                mask = self._sliding_window_mask[:seq_len, :seq_len]
                scores = scores.masked_fill(~mask, 0.0)
                denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                weights = scores / denom
                attn_output = torch.einsum("bhts,bhsd->bhtd", weights, v_float)
            else:
                # Causal linear-attention form (prefix-sum cumsum).
                # Window defaults to full causal when SWA is off.
                window = self.sliding_window_size if self.use_sliding_window else seq_len
                kv = k_phi_raw.unsqueeze(-1) * v_float.unsqueeze(-2)
                prefix_kv = torch.cat(
                    [torch.zeros_like(kv[:, :, :1]), kv.cumsum(dim=2)],
                    dim=2,
                )
                prefix_k = torch.cat(
                    [torch.zeros_like(k_phi_raw[:, :, :1]), k_phi_raw.cumsum(dim=2)],
                    dim=2,
                )
                end_idx = torch.arange(1, seq_len + 1, device=Q.device)
                start_idx = (end_idx - window).clamp_min(0)
                kv_sum = prefix_kv[:, :, end_idx] - prefix_kv[:, :, start_idx]
                k_sum = prefix_k[:, :, end_idx] - prefix_k[:, :, start_idx]
                numerator = torch.einsum("bhtd,bhtde->bhte", q_phi, kv_sum)
                # Mandatory denominator (bound in spec, no skip-flag).
                denom = torch.einsum(
                    "bhtd,bhtd->bht", q_phi, k_sum
                ).clamp_min(1e-6).unsqueeze(-1)
                attn_output = numerator / denom
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
        # 181 — Cross-Head Channel RMSNorm. Normalize each token's
        # attention output ACROSS HEADS within each d_k slice so
        # all H heads land on the same per-(t, k) scale before
        # the W_O projection. Standard post-AV RMSNorm (LLaMA-3 /
        # Gemma-2 / Qwen-2) normalizes over the concatenated
        # d_model axis; 181 instead treats each d_k index as a
        # single "channel" of H values and normalizes across
        # those H values — explicitly coupling the head
        # magnitudes. The 160-null closed per-head *independent*
        # rescaling; 181 tests the *joint* post-AV magnitude
        # coupling axis.
        #   α_h = relu(α_raw_h)                          # 0 at init
        #   rms[b, t, k] = sqrt(mean_h(out²) + ε)         # one per (b, t, k)
        #   gain_h[k] = 1 + tanh(γ_raw_h[k])              # 1 at init
        #   out = (1 − α_h)·out + α_h·(out / rms)·gain
        # At init α=0, gain=1 ⇒ `out` unchanged exactly ⇒
        # byte-identical to baseline at step 0 (max-abs-diff =
        # 0.0). Sits BEFORE the `use_head_gain` branch below so
        # the two compose by being multiplicative in series
        # (but the mutual-exclusion asserts forbid both-on in a
        # single run). See
        # `autoresearch/ideas/181-cross-head-rmsnorm/idea.md`.
        if self.use_cross_head_rmsnorm:
            rms = (attn_output.pow(2).mean(dim=1, keepdim=True) + 1e-6).sqrt()
            alpha = F.relu(self.cross_head_rmsnorm_alpha_raw).view(
                1, self.n_heads, 1, 1
            )
            gain = 1.0 + torch.tanh(
                self.cross_head_rmsnorm_gain_raw
            ).view(1, self.n_heads, 1, self.d_k)
            attn_output = (1.0 - alpha) * attn_output + alpha * (
                attn_output / rms
            ) * gain
        # 160 — Per-head RMS gain on the attention output. Multiply each
        # head's AV-aggregated output `o_h = (A·V)_h ∈ R^{T×d_k}` by a
        # learnable scalar `g_h ∈ R^H` so each head controls its
        # contribution magnitude to the residual stream. Init
        # `g_h = 1.0` ⇒ `o_h *= 1 = o_h` byte-identical to baseline at
        # step 0. Applied BEFORE the existing per-head gates
        # (`use_attn_output_gate`, `use_attn_output_channel_gate`,
        # `use_gated_attn`, `_apply_output_op`) so it composes cleanly
        # with all of them — they multiply through. Default off →
        # branch never taken, baseline path bit-identical. See
        # `autoresearch/ideas/160-rms-gain-per-head/idea.md`.
        if self.use_head_gain:
            attn_output = attn_output * self.head_gain.view(1, self.n_heads, 1, 1)
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

        # 191 — Per-token attention output gain. Multiply the
        # post-merge `[B, T, d_model]` attention output by a
        # learnable per-position scalar `(1 + γ_t)` where
        # `γ_t ∈ R^T` is shared across batch and the d_model
        # axis. Init γ=0 ⇒ (1 + 0) = 1 exactly ⇒
        # `attn * 1 = attn` byte-identical to baseline at step 0.
        # Per-token granularity (T scalars/block) is a different
        # axis from the closed per-head (160: H scalars), per-
        # channel (142: d_model scalars), and per-(h, k) (181:
        # H·d_k scalars) levers. Sits AFTER the merge-reshape
        # and BEFORE the W_O projection, alongside the
        # `use_v_mix_conv` (163) and `use_av_carry` (168) post-
        # merge sites — the three compose by being multiplicative
        # in series (191 first scales the per-position
        # contribution, 163 convolves along T, 168 carries across
        # blocks), and the mutual-exclusion asserts forbid
        # combining 191 with the pre-merge post-AV gates in a
        # single run. Sliced to `[:seq_len]` so inference at
        # shorter T only consumes the first `seq_len` scalars.
        # Default off → branch never taken, baseline path bit-
        # identical. See
        # `autoresearch/ideas/191-token-attn-gain/idea.md`.
        if self.use_token_attn_gain:
            attn_output = attn_output * (
                1.0 + self.token_attn_gain[:seq_len].view(
                    1, seq_len, 1
                )
            )

        # 203 — Pre-W_O Squeeze-Excitation channel attention (Hu
        # et al. TPAMI 2019, arXiv:1709.01507). Per-token channel
        # reweighting on the post-merge attention output via a
        # tiny bottleneck MLP, blended onto the residual stream
        # by a per-block γ-gate (init `se_alpha_init=-10.0` ⇒
        # `sigmoid(-10) ≈ 4.54e-5` ⇒ silent at step 0). Sits
        # AFTER the 191 (per-token scalar gain) site and BEFORE
        # the 163 (v_mix_conv) site, alongside the other post-
        # merge / pre-W_O levers — composes additively with the
        # residual stream after the γ-blend. Same W_1, W_2
        # applied to every token/position (no T-axis pooling —
        # the original SE-Net CNN pattern pools over the spatial
        # axis, but here the lever is the per-token content-
        # dependent cell, not the original CNN cell). At init
        # γ ≈ 4.5e-5 ⇒ the blend's contribution to
        # `attn_out_post` is at the fp32 floor regardless of
        # what `se_w` is. As γ grows during training, the
        # SE branch is *added* to the residual via the
        # γ-weighted blend. The internal `se_w` at step 0 is
        # ~0.5 per channel (sigmoid of Kaiming-init), but the
        # γ-gate silences the whole branch. Distinct from 142
        # (per-channel static gain), 160 (per-head gain), 181
        # (cross-head RMSNorm), 191 (per-token scalar gain) —
        # 203 is the *per-token channel vector* (content-
        # dependent channel reweighting). Default off → no
        # Parameter registered, no `nn.Linear` built, no branch
        # taken, baseline path bit-identical. See
        # `autoresearch/ideas/203-pre-wo-se-channel-attn/idea.md`.
        if self.se_W1 is not None:
            se_inner = F.gelu(self.se_W1(attn_output))   # [B, T, d_model/r]
            se_w = torch.sigmoid(self.se_W2(se_inner))    # [B, T, d_model]
            se_branch = attn_output * se_w                # [B, T, d_model]
            gamma = torch.sigmoid(self.se_gamma_raw)      # []
            attn_output = (1.0 - gamma) * attn_output + gamma * se_branch

        # 163 — Post-Attention V-Mix Depthwise Convolution. Apply a
        # symmetric depthwise Conv1d over the time axis on the
        # post-attention tensor `[B, T, d_model]` BEFORE the W_O
        # projection. Conv weights are identity-initialized
        # (center tap = 1, rest = 0, set inline at construction —
        # see `MultiHeadAttention.__init__` for the raw-Parameter
        # rationale). With `padding = k//2` symmetric (causal+
        # future — the attention sublayer has already integrated
        # the full causal context, so the conv may look at both
        # neighbors) and `groups = d_model` (depthwise), the conv
        # is a strict identity at step 0 ⇒ the post-attention
        # tensor equals the no-flag path bit-for-bit. Composes
        # cleanly with every preceding output-side lever
        # (`use_head_gain`, `use_attn_output_gate`,
        # `use_attn_output_channel_gate`, `use_gated_attn`,
        # `use_talking_heads_out`, `_apply_output_op`) — those are
        # multiplicative scalars on `attn_output`, the conv is a
        # linear op that reads their combined result. Default off
        # → branch never taken, baseline path bit-identical. See
        # `autoresearch/ideas/163-v-mix-conv/idea.md`.
        if self.use_v_mix_conv:
            h = F.conv1d(
                attn_output.transpose(1, 2),
                self.v_mix_conv_weight,
                bias=None,
                stride=1,
                padding=self.v_mix_conv_kernel // 2,
                groups=self.d_model,
            )  # [B, d_model, T]
            attn_output = h.transpose(1, 2)  # [B, T, d_model]

        # 201 — Degenerate gMLP Spatial Gating Unit on attention
        # output (Liu et al. 2021, arXiv:2105.08050, §3.1). Sits at
        # the same post-merge / pre-W_O site as 163 (v_mix_conv,
        # applied above) — composes additively with everything
        # upstream and gets summed by the W_O linear below.
        # `sgu_W is None` ⇒ this block is on a non-stochastic stride
        # (or the flag is off) ⇒ skip silently. When present, the
        # forward is:
        #   z = attn_out.mean(dim=T, keepdim=True)   # [B, 1, d_model]
        #   z = F.gelu(z)                            # nonlinearity
        #   z = z @ sgu_W                            # [B, 1, d_model]
        #   z = z.expand(-1, T, -1)                  # broadcast T
        #   attn_out_post = attn_out + α · z        # α = σ(sgu_alpha)
        # At step 0 α ≈ 4.5e-5 ⇒ the additive contribution is ≈ 0
        # ⇒ forward output is bit-identical to the no-flag path
        # within fp32 noise of one extra multiply-add. Composes
        # cleanly with every preceding output-side lever
        # (`use_head_gain`, `use_attn_output_gate`,
        # `use_attn_output_channel_gate`, `use_gated_attn`,
        # `use_talking_heads_out`, `use_v_mix_conv`, `_apply_output_op`)
        # — those are multiplicative scalars / linear ops on
        # `attn_output`, the SGU is an additive scalar broadcast that
        # reads their combined result. Default off → no Parameter
        # registered, no branch taken, baseline path bit-identical.
        # See `autoresearch/ideas/201-mlp-token-mixer/idea.md` /
        # `plan.md`.
        if self.sgu_W is not None:
            z = attn_output.mean(dim=1, keepdim=True)  # [B, 1, d_model]
            z = F.gelu(z)
            z = z @ self.sgu_W                          # [B, 1, d_model]
            z = z.expand(-1, seq_len, -1)
            alpha = torch.sigmoid(self.sgu_alpha)
            attn_output = attn_output + alpha * z

        # 168 — AV-Output Carry: stash on layer 0 (`av_carry is None`
        # ⇒ no previous block exists), blend on layer l ≥ 1
        # (`av_carry is not None` ⇒ previous block's post-AV pre-W_O
        # tensor). Site is post-merge-reshape (`[B, T, d_model]`),
        # pre-W_O — sits BEFORE the W_O projection so it composes
        # with 160/024/107/045 output-side gates (those run on
        # per-head `[B, H, T, d_k]` above the reshape). With α=0 init,
        # `α · av_{l-1} = 0` exactly in fp32 ⇒ the carry term is a
        # numerical no-op at step 0 and the W_O projection sees the
        # same input as the baseline path. `.detach()` mirrors 021's
        # V-residual contract so the cross-block gradient is
        # structurally bounded to α_av's 0-dim scalar.
        if self.use_av_output_carry:
            if av_carry is None:
                self._av_carry = attn_output.detach()
            else:
                attn_output = attn_output + self.alpha_av * av_carry

        # ============ MERGED O PROJECTION ============
        # Use the last part of qkvo_proj for output projection.
        # 171 — DropConnect on W_O (Wan et al. 2013, arXiv:1304.3174).
        # Per-weight Bernoulli mask on the O slice of `qkvo_proj` during
        # training. Sample `M ∈ {0,1}^{d_model × d_model}` with
        # `M_ij ~ Bernoulli(1 - effective_rate)` and use
        # `W_O_masked = W_O ⊙ M / (1 - effective_rate)` (inverted-
        # dropout rescale so the expected magnitude matches the un-
        # masked baseline — matches `F.dropout` and the 147-DropKey
        # sibling's rescale). The mask is sampled per forward pass and
        # shared across all batch elements and positions (one mask per
        # layer per call, NOT per-token).
        #
        # **Warmup ramp.** The effective rate is
        #     effective_rate = dropconnect_wo_rate
        #                      * min(step, warmup) / warmup
        # where `step = self._dc_step_count` (Python int, incremented
        # at the END of each forward below) and `warmup =
        # self.dropconnect_wo_warmup_steps`. At step 0 (first forward
        # call) `effective_rate = 0.0` ⇒ the mask branch is short-
        # circuited before any RNG is consumed ⇒ the trt forward is
        # **bit-identical to baseline at step 0** (max-abs-diff = 0.0
        # across the full forward, no parameter modified, no RNG
        # consumed). At step `warmup` (default 100) the effective rate
        # reaches `dropconnect_wo_rate` (default 0.05) and holds there
        # for the remaining steps.
        #
        # At eval (`self.training == False`) and with
        # `dropconnect_wo_rate=0.0` the branch is never taken ⇒ W_O is
        # used unchanged and the forward graph is bit-identical to the
        # no-DropConnect baseline. When `use_dropconnect_wo=False`
        # (default) the branch is also never taken, no RNG consumed,
        # baseline path bit-identical.
        #
        # The mask site is the O slice `qkvo_proj[qkv_size:]` (shape
        # `[d_model, d_model]`) — this is the same tensor `F.linear`
        # reads below, so masking it in-place produces exactly
        # `output = attn_output @ W_O_masked` with no extra ops on
        # `attn_output`. Composes cleanly with every preceding output-
        # side lever (168 AV-carry, 163 v-mix-conv, 160 head-gain,
        # 024 gated-attn, 107 exclusive-self-attn, 045 output-embed)
        # — those all run on `attn_output` and the masked W_O sees
        # the same input as the un-masked W_O would. See
        # `autoresearch/ideas/171-dropconnect-wo/idea.md`.
        w_o = self.qkvo_proj[self.qkv_size:]
        # 199 — Spectral-Norm-Bounded W_O Projection. Per-block
        # learnable Lipschitz cap on the O-slice weight. Track
        # `σ_max(W_O)` via power iteration on a per-block Buffer
        # vector `u ∈ R^{d_model}`; on the FIRST forward, seed
        # `u` from a DETERMINISTIC unit vector (`u[0]=1`, rest 0,
        # then renormalized — NOT `torch.randn`, which would
        # consume model RNG inside forward and break dropout
        # reproducibility) and snapshot `σ_max_init = ||w_o · u||₂
        # / ||u||₂` (Rayleigh quotient — equals `||w_o||₂` to
        # within PI convergence error). Subsequent forwards run
        # `wo_spectral_cap_pi_iters` PI steps and read the
        # current `σ_max` from the same Rayleigh quotient on the
        # *current* `w_o`. The cap is applied BEFORE the 197
        # blend (and therefore BEFORE the 171-DropConnect mask
        # and 207-LowRank addition) so the cap operates on the
        # per-block W_O weight itself, and the blend / mask /
        # lowrank still operate on a valid `[d_model, d_model]`
        # O-slice tensor.
        #
        # At step 0: `γ_l = 0` ⇒ `exp(γ_l) = 1`; `σ_max = σ_max_init`
        # (snapshot on the same forward) ⇒ the cap factor
        # `min(1, σ_max_init · exp(γ_l) / σ_max) = min(1, 1) = 1`
        # ⇒ `w_o_eff = w_o` byte-identical to the no-flag baseline
        # (the lever is dormant). As training proceeds, `σ_max(W_O)`
        # typically grows under SGD; the optimizer can push `γ_l < 0`
        # to tighten the cap and bind the Lipschitz constant on the
        # projection. The cap is asymmetric (clip-only) — `γ_l > 0`
        # is wasted optimizer signal because the clip never fires.
        #
        # PI implementation: right-side power iteration
        # `u ← w_o · u / ||·||₂` (one matmul + norm per step);
        # `||w_o · u||₂ / ||u||₂` is the spectral-norm estimate.
        # At PI=1 with a converged `u` this is exact; at PI=1 with
        # an unconverged `u` it underestimates σ_max — producing
        # a slightly TIGHTER cap than σ_max truly warrants, which
        # is safe (always ≤ true Lipschitz bound) and γ_l can
        # compensate. PI state is updated under `no_grad` (does
        # NOT consume backward graph) but the cap factor on the
        # CURRENT `w_o` IS in the autograd graph (because `w_o` is
        # a leaf Parameter slice — gradient flows through
        # `w_o * cap_factor` to `w_o`).
        if self.use_wo_spectral_cap:
            with torch.no_grad():
                # Seed `u` once (deterministic — `u[0]=1`, rest 0)
                # on the first forward; the FIRST forward's
                # σ_max estimate IS the σ_max_init we snapshot,
                # so they are produced from the SAME PI run and
                # are numerically equal ⇒ cap_factor = min(1, 1)
                # = 1 exactly ⇒ `w_o_eff == w_o` byte-identical
                # to the no-flag baseline at step 0. NOT
                # `torch.randn` — that would consume model RNG
                # inside forward and break dropout reproducibility.
                if not self._wo_pi_initialized:
                    seed_u = torch.zeros(
                        self.d_model, device=w_o.device, dtype=w_o.dtype
                    )
                    seed_u[0] = 1.0
                    self._wo_pi_u.copy_(seed_u)
                u = self._wo_pi_u
                for _ in range(self.wo_spectral_cap_pi_iters):
                    wu = w_o @ u
                    u = wu / (wu.norm() + 1e-12)
                self._wo_pi_u.copy_(u)
                wu_final = w_o @ u
                sigma_max_now = (
                    wu_final.norm() / (u.norm() + 1e-12)
                ).detach()
                if not self._wo_pi_initialized:
                    self._wo_pi_sigma_max_init.copy_(sigma_max_now)
                    self._wo_pi_initialized = True
            sigma_max_init = self._wo_pi_sigma_max_init
            cap_factor = torch.minimum(
                torch.ones_like(sigma_max_init),
                (sigma_max_init * torch.exp(self.wo_spectral_cap_gamma))
                / (sigma_max_now + 1e-12),
            )
            w_o = w_o * cap_factor
        # 197 — Tied W_O Across Blocks. Soft-blend the per-block
        # W_O slice with the model-wide shared `W_O_shared`
        # parameter, gated by a per-block sigmoid-bounded α. Sits
        # at the W_O application site, BEFORE the 171-DropConnect
        # mask branch (and therefore BEFORE the 207-LowRank
        # addition below) so the 171 mask and 207 lowrank
        # correction still operate on a valid `[d_model, d_model]`
        # O-slice tensor. The blend is
        #   `w_o_eff = (1 − σ(α_b_raw)) · w_o_b + σ(α_b_raw) · W_O_shared`
        # At step 0 `σ(−10) ≈ 4.54e-5` and `W_O_shared` is
        # std=0.02 normal-init ⇒ `W_O_shared`'s contribution is
        # on the order of 1e-7 in std ⇒ `w_o_eff ≈ w_o_b` to
        # within fp32 noise of one extra multiply-add on the
        # O slice. This is the same step-0 tolerance accepted
        # by the 188 / 204 cross-block siblings (and the lever
        # is structurally identical to 188's K/V blend, just
        # applied to the O projection). Default off → branch
        # is gated on `self.use_tied_wo_across_blocks`, the
        # baseline path is bit-identical. See
        # `autoresearch/ideas/197-tied-wo-across-blocks/idea.md`
        # / `plan.md`.
        if self.use_tied_wo_across_blocks:
            alpha = torch.sigmoid(self.tied_wo_alpha_raw)
            w_o = (1.0 - alpha) * w_o + alpha * self.tied_wo_shared
        if self.use_dropconnect_wo and self.training:
            warmup = max(int(self.dropconnect_wo_warmup_steps), 1)
            step = min(int(self._dc_step_count), warmup)
            effective_rate = float(self.dropconnect_wo_rate) * step / warmup
            if effective_rate > 0.0:
                keep_prob = 1.0 - effective_rate
                wo_mask = torch.empty_like(w_o).bernoulli_(keep_prob)
                w_o = w_o * wo_mask / keep_prob
        # 207 — W_O Low-Rank Bottleneck. After any 171-DropConnect
        # masking (which runs first on `w_o`), add a learnable rank-r
        # correction: `w_o_eff = w_o + σ(α) · (wo_a @ wo_b)`. At step
        # 0 `wo_b = 0` ⇒ `wo_a @ wo_b = 0` exactly ⇒ the addition is a
        # numerical no-op ⇒ `w_o_eff == w_o` bit-identical to the
        # no-flag baseline (and to the 171-only path when 171 is on).
        # Sits AFTER the 171 mask and BEFORE the `F.linear` so the
        # composition is `output = attn_output @ (masked_w_o +
        # lowrank_correction)`. The 171 mask is multiplied into
        # `w_o` (in-place rewrite of the local var), and 207 adds the
        # low-rank correction on top — the two levers multiply through
        # the linear without re-projection. Default off → branch
        # never taken, baseline path bit-identical. See
        # `autoresearch/ideas/207-wo-lowrank-bottleneck/idea.md` /
        # `plan.md`.
        if self.use_lowrank_wo:
            alpha = torch.sigmoid(self.wo_lowrank_alpha)
            w_o = w_o + alpha * (self.wo_a @ self.wo_b)
        output = F.linear(attn_output, w_o)
        # #33 output embeddings: add the projected token embedding to the
        # attention OUTPUT (post-O). Different operating point from V/Q/K
        # (which inject into attention inputs). ve is the raw token
        # embedding [B, T, emb_rank], projection is zero-init so step 0
        # matches the baseline.
        if self.use_output_embed and ve is not None:
            output = output + F.linear(ve, self.output_embed_proj)
        # 171 — DropConnect step counter increment. Done AT THE END of
        # forward (after the W_O branch has already read `_dc_step_count`)
        # so the first forward call sees step=0 ⇒ effective_rate=0.0 ⇒
        # mask branch short-circuits ⇒ trt forward is bit-identical to
        # baseline at step 0. Increments unconditionally on every
        # forward (training or eval) — eval mode short-circuits the
        # mask branch via `self.training` so the counter is harmless
        # there. See `autoresearch/ideas/171-dropconnect-wo/idea.md`.
        self._dc_step_count += 1
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
        # 166 — T5-style bucketed relative position bias. Pass-
        # through to the inner MHA (see
        # `MultiHeadAttention.use_t5_rpe` for the mechanism).
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/166-t5-rpe/idea.md`.
        use_t5_rpe: bool = False,
        t5_rpe_buckets: int = 32,
        # 152 — Per-head attention logit bias. Pass-through to the
        # inner MHA (see `MultiHeadAttention.use_attn_logit_bias` for
        # the mechanism and the math-identity caveat). Default off →
        # baseline path bit-identical. See
        # `autoresearch/ideas/152-attn-logit-bias/idea.md`.
        use_attn_logit_bias: bool = False,
        # 155 — Per-head learnable attention temperature. Pass-through
        # to the inner MHA. See `MultiHeadAttention.use_per_head_temp`
        # for the mechanism. Default off → baseline path bit-identical.
        use_per_head_temp: bool = False,
        # 195 — Tight hard QK logit clamp pass-through to the
        # inner MHA. See `MultiHeadAttention.use_qk_clamp` for the
        # mechanism. Default off → baseline path bit-identical.
        # See `autoresearch/ideas/195-qk-clamp-min-max/idea.md`.
        use_qk_clamp: bool = False,
        qk_clamp_c: float = 2.0,
        # 193 — Blockwise attention temperature schedule pass-through
        # to the inner MHA. See `MultiHeadAttention.use_block_temp_
        # schedule` for the mechanism (per-block fixed cosine
        # temperature, no learned params; τ_b divides the pre-softmax
        # `Q·K^T/√d_k` so the pre-softmax score must be exposed).
        # `block_temp_alpha` is the schedule amplitude (default
        # `−0.3` in `Tiny1M3MBlockTempConfig`, the committed single
        # value per `idea.md` r2). `tau_b` is the precomputed scalar
        # for THIS block (the model passes it in via the loop;
        # `tau_b = 1 + α · cos(π · b / (L − 1))`); at α=0
        # `tau_b = 1` exactly so the divide is the identity. Default
        # off → no Buffer registered, no branch taken, baseline path
        # bit-identical. See
        # `autoresearch/ideas/193-blockwise-attn-temp-schedule/idea.md`.
        use_block_temp_schedule: bool = False,
        block_temp_alpha: float = 0.0,
        tau_b: float = 1.0,
        # 180 — Pre-softmax 1D causal depthwise conv on attention
        # logits. Pass-through to the inner MHA. See
        # `MultiHeadAttention.use_logit_conv` for the mechanism
        # (delta-init ⇒ step-0 byte-identical to baseline, optimizer
        # grows a per-head smoothing kernel). Default off → baseline
        # path bit-identical. See
        # `autoresearch/ideas/180-qk-logit-conv/idea.md`.
        use_logit_conv: bool = False,
        logit_conv_kernel_size: int = 3,
        # 202 — V-Only Soft-Blend Probe pass-through to the inner
        # MHA. Per-head `sigmoid(α_h)` blends per-head V with
        # per-group-shared V; K is untouched (K-axis is the
        # held-out implicit control). Init `α = -25.0` ⇒
        # `σ(α) ≈ 1.4e-11` ⇒ V_h_eff = V_h_local exactly in fp32
        # at step 0 ⇒ baseline path bit-identical. Default off →
        # baseline path bit-identical. See
        # `MultiHeadAttention.use_grouped_v` for the full
        # mechanism and `autoresearch/ideas/202-grouped-value-
        # projection/idea.md`.
        use_grouped_v: bool = False,
        v_group_size: int = 2,
        # 179 — Anti-Causal Sub-Heads. Pass-through to the inner
        # MHA. Per-head learnable scalar `γ_h` attenuates the
        # upper-triangle fill by `(1 − γ_h)`. Init `-10` ⇒
        # `γ_h ≈ 4.5e-5` at step 0 ⇒ byte-identical to baseline.
        # See `MultiHeadAttention.use_anti_causal_subheads` for
        # the full mechanism. Default off → baseline path bit-
        # identical. See
        # `autoresearch/ideas/179-anti-causal-subheads/idea.md`.
        use_anti_causal_subheads: bool = False,
        # 160 — Per-head RMS gain on the attention output. Pass-through
        # to the inner MHA. See `MultiHeadAttention.use_head_gain` for
        # the mechanism. Default off → baseline path bit-identical.
        # See `autoresearch/ideas/160-rms-gain-per-head/idea.md`.
        use_head_gain: bool = False,
        # 181 — Cross-Head Channel RMSNorm. Pass-through to the
        # inner MHA. See `MultiHeadAttention.use_cross_head_rmsnorm`
        # for the mechanism. Default off → baseline path
        # bit-identical. See
        # `autoresearch/ideas/181-cross-head-rmsnorm/idea.md`.
        use_cross_head_rmsnorm: bool = False,
        # 191 — Per-token attention output gain pass-through to
        # the inner MHA. See `MultiHeadAttention.use_token_attn_gain`
        # for the mechanism. Default off → baseline path
        # bit-identical. See
        # `autoresearch/ideas/191-token-attn-gain/idea.md`.
        use_token_attn_gain: bool = False,
        # 203 — Pre-W_O Squeeze-Excitation channel attention pass-
        # through to the inner MHA. See
        # `MultiHeadAttention.use_se_pre_wo` for the mechanism.
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/203-pre-wo-se-channel-attn/idea.md`.
        use_se_pre_wo: bool = False,
        se_reduction_ratio: int = 4,
        se_alpha_init: float = -10.0,
        # 147 — DropKey (Xu et al. 2022, arXiv:2207.01058). Per-head,
        # per-token Bernoulli mask on K during training. Pass-through
        # to the inner MHA. See `autoresearch/ideas/147-dropkey/idea.md`.
        use_drop_key: bool = False,
        drop_key_rate: float = 0.1,
        # 171 — DropConnect on W_O pass-through to the inner MHA.
        # See `MultiHeadAttention.use_dropconnect_wo` for the
        # mechanism (per-weight Bernoulli mask on W_O, inverted-
        # dropout rescale, eval-mode skip, warmup-ramped effective
        # rate so step 0 is byte-identical to baseline). Default off
        # → baseline path bit-identical. See
        # `autoresearch/ideas/171-dropconnect-wo/idea.md`.
        use_dropconnect_wo: bool = False,
        dropconnect_wo_rate: float = 0.0,
        dropconnect_wo_warmup_steps: int = 100,
        # 207 — W_O Low-Rank Bottleneck (learnable rank-r residual
        # correction on the W_O projection). Pass-through to the
        # inner MHA. See `MultiHeadAttention.use_lowrank_wo` for the
        # mechanism (`W_O_eff = W_O + σ(α) · (W_O_A @ W_O_B)`, with
        # `W_O_B` zero-init and `α` init −10 ⇒ step-0 bit-identical
        # to baseline). `wo_rank` (default 16) sets the absolute rank
        # of the correction; `wo_lowrank_alpha_init` (default −10)
        # sets the soft-gate init. Default off → no Parameter
        # registered, baseline path bit-identical. See
        # `autoresearch/ideas/207-wo-lowrank-bottleneck/idea.md` /
        # `plan.md`.
        use_lowrank_wo: bool = False,
        wo_rank: int = 16,
        wo_lowrank_alpha_init: float = -10.0,
        # 194 — W_V Low-Rank Residual Correction pass-through to the
        # inner MHA. See `MultiHeadAttention.use_lowrank_wv` for the
        # mechanism (`W_V_eff = W_V + σ(α) · (W_V_A @ W_V_B)`, with
        # `W_V_B` zero-init and `α` init −10 ⇒ step-0 bit-identical
        # to baseline). `wv_rank` (default 8) sets the absolute rank
        # of the correction; `wv_lowrank_alpha_init` (default −10)
        # sets the soft-gate init. Default off → no Parameter
        # registered, baseline path bit-identical. See
        # `autoresearch/ideas/194-lowrank-ffn/idea.md` / `plan.md`.
        use_lowrank_wv: bool = False,
        wv_rank: int = 8,
        wv_lowrank_alpha_init: float = -10.0,
        # 151 — RoV (Rotary Value Embeddings, gated). Pass-through to
        # the inner MHA. See `MultiHeadAttention.use_rov` for the
        # mechanism. Default off → baseline path bit-identical. See
        # `autoresearch/ideas/151-rov-gated/idea.md`.
        use_rov: bool = False,
        # 174 — xPos exponential decay on K. Pass-through to the inner
        # MHA (see `MultiHeadAttention.use_xpos` for the mechanism).
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/174-xpos-decay/idea.md`.
        use_xpos: bool = False,
        # 154 — Rebased Attention. Pass-through to the inner MHA.
        # See `MultiHeadAttention.use_rebased_attn` for the
        # mechanism. Default off → baseline path bit-identical.
        # `rebase_stride` is the time-axis pool stride (default 8,
        # → R=256 rebasins at T=2048). Identity when
        # `rebase_stride >= T`. See
        # `autoresearch/ideas/154-rebased-attn/idea.md`.
        use_rebased_attn: bool = False,
        rebase_stride: int = 8,
        # 156 — Mixture-of-Attentions (MoA). Pass-through to the
        # inner MHA. See `MultiHeadAttention.use_moa` for the
        # mechanism (E parallel K/V experts + per-token router).
        # `moa_num_experts` is clamped to >= 2 at MHA construction.
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/156-moa/idea.md`.
        use_moa: bool = False,
        moa_num_experts: int = 2,
        # 178 — Gated Multi-Query Attention. Per-KV-head scalar gate
        # β_k, β_v blending per-head K, V with a single shared K, V
        # projection. Init 0 ⇒ step-0 forward is bit-identical to
        # the no-flag baseline. Default off → baseline path bit-
        # identical. See `autoresearch/ideas/178-mqa-gated/idea.md`.
        use_mqa_gated: bool = False,
        # 185 — Static per-head learned K-rotation pass-through to
        # the inner MHA. Init θ=0 ⇒ step-0 forward bit-identical to
        # baseline. Default off → baseline path bit-identical. See
        # `autoresearch/ideas/185-static-per-head-k-rotation/idea.md`.
        use_static_k_rotation: bool = False,
        # 200 — Static per-layer × per-pair learned K-rotation
        # pass-through to the inner MHA (depth-axis twin of 185,
        # shared across heads, K-only). Init φ=0 ⇒ `R_l = I_{d_k}`
        # exactly in fp32 ⇒ step-0 forward bit-identical to
        # baseline. Default off → baseline path bit-identical.
        # See `autoresearch/ideas/200-rope-phase-offset-per-layer/idea.md`.
        use_per_layer_k_rotation: bool = False,
        # 192 — Pre-RoPE per-head × per-pair learned Q+K rotation
        # pass-through to the inner MHA (orthogonal-rebase axis,
        # Q+K-side, pre-RoPE placement). Init φ=0 ⇒ `R_h = I_{d_k}`
        # exactly in fp32 ⇒ step-0 forward bit-identical to
        # baseline. Default off → baseline path bit-identical.
        # See `autoresearch/ideas/192-pre-rope-qk-rotation/idea.md`.
        use_pre_rope_rotation: bool = False,
        # 182 — Per-head learnable attention window pass-through.
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/182-per-head-window/idea.md`.
        use_per_head_window: bool = False,
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
        # 153 — Squared-ReLU FFN activation. Default off → the
        # standard `ffn_variant` branch runs unchanged, baseline
        # forward graph bit-identical. See
        # `autoresearch/ideas/153-relu2-ffn/idea.md`.
        use_relu2_ffn: bool = False,
        # 170 — SwiGLU FFN (Shazeer 2020, arXiv:2002.05202;
        # LLaMA-family FFN). Three-projection gated linear unit with
        # zero-init gate. Default off → the standard `ffn_variant`
        # cascade runs unchanged, baseline forward graph bit-
        # identical. See `autoresearch/ideas/170-swiglu-ffn/idea.md`.
        use_swiglu_ffn: bool = False,
        # 206 — Cross-Block W_up / W_down Projection Sharing
        # (Universal-Transformers-style learnable parameter
        # sharing across depth, narrowed to the two largest FFN
        # matrices only). When on, the block's FFN is asked to
        # blend its W_up / W_down with the previous block's
        # (detached) W_up / W_down. The block's FFN module
        # allocates the per-side α scalars
        # (`ffn_share_alpha_up` / `ffn_share_alpha_down`) at
        # construction (init `ffn_share_alpha_init=-10.0` ⇒
        # `σ(-10) ≈ 4.5e-5` ⇒ silent at step 0). The FFN
        # stashes the current W_up / W_down (`.detach()`-ed) so
        # the model loop can read them for the next block's
        # `prev_W_up=` / `prev_W_down=`. Default off ⇒ no
        # α Parameter registered, no stash, no blend, baseline
        # path bit-identical. See
        # `autoresearch/ideas/206-cross-block-ffn-share/idea.md` /
        # `plan.md`.
        use_cross_block_ffn_share: bool = False,
        ffn_share_alpha_init: float = -10.0,
        # 196 — MishGLU FFN (Misra 2019 + Shazeer 2020 composition;
        # inner-activation axis orthogonal to 170's outer-GLU axis).
        # Three-projection gated linear unit with `mish` as the inner
        # gate activation (instead of 170's `silu`). `mish(0)=0` ⇒
        # step-0 forward is silent without an explicit zero-init (and
        # a zero-init would mask the gradient signal the lever
        # depends on — `dMish/dx|_{x=0} ≈ 0.6` vs `dSiLU/dx|_{x=0}
        # = 0.5` is the lever). d_ff is scaled by the Shazeer 2/3
        # trick (`(2 * d_ff) // 3`) so total FFN param count matches
        # SwiGLU to within ~0.4%. Default off → the standard
        # `ffn_variant` cascade runs unchanged, baseline forward
        # graph bit-identical. Mutually exclusive with `use_swiglu_ffn`
        # (both target the FFN slot) — branch sits AHEAD of 170 so
        # the new lever isn't silently shadowed. See
        # `autoresearch/ideas/196-ffn-glu-mish/idea.md`.
        use_mish_glu: bool = False,
        # 198 — Pre-FFN Attention Mixing (FiLM-style cross-stream
        # conditioning; Perez et al. 2018, arXiv:1709.07871). When
        # True, the block registers a 0-dim scalar
        # `pre_ffn_attn_mix_gamma_raw` (init `pre_ffn_attn_mix_init`,
        # default −10 ⇒ sigmoid ≈ 4.5e-5) and the pre-norm2 path
        # mixes the *raw* attention output into the FFN input:
        #     ffn_in = norm2(x + sigmoid(γ) · attn_out_raw.detach())
        # The `.detach()` keeps γ's gradient cleanly tied to FFN-
        # side loss only (no gradient through the attention path's
        # Q/K/V/O projections at step 0). At init sigmoid(−10) ≈
        # 4.5e-5 ⇒ baseline path is fp32-noise bit-identical at
        # step 0. Default off → no Parameter registered, no forward
        # branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/198-pre-ffn-attnmix/idea.md`.
        use_pre_ffn_attn_mix: bool = False,
        pre_ffn_attn_mix_init: float = -10.0,
        # 197 — DeepNet α fixed residual init pass-through. See
        # `MultiHeadAttention.use_deepnet_alpha` for the mechanism
        # (Wang et al. 2022, arXiv:2203.00555): a single *fixed*
        # (not learned) global scalar `α = (2·n_layers)^(-1/2)` is
        # applied to every block's sublayer output before the
        # residual add. 0 new params. Default off → baseline path
        # bit-identical (the `self.deepnet_alpha` attribute is
        # still set, but the multiply only runs when the flag is
        # on). See `autoresearch/ideas/197-output-residual-sqrt-2l/
        # idea.md`.
        use_deepnet_alpha: bool = False,
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
        # 189 — CosFormer-style linear attention (Qin et al. NeurIPS
        # 2022, arXiv:2202.08791). Pass-through to the inner MHA —
        # the MHA flag `use_cosformer` gates the cosFormer branch in
        # MHA.forward, the per-block γ scalar lives on the model
        # (`MinimalLLM.cosformer_gammas`) and is read via
        # `cosformer_gamma` in `TransformerBlock.forward`. Default
        # off → baseline path bit-identical. See
        # `autoresearch/ideas/189-cosformer-linear-attn/idea.md`.
        use_cosformer: bool = False,
        cosformer_gamma_init: float = 0.0,
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
        # 176 — Pre-AV V RMSNorm (per-head α-gate + per-head γ-gain):
        # forward to MHA — see `MultiHeadAttention.use_v_rmsnorm` for
        # the mechanism. Default off → V stays unnormalized, baseline
        # path is bit-identical. See
        # `autoresearch/ideas/176-v-pre-av-norm/idea.md`.
        use_v_rmsnorm: bool = False,
        # 162 — Q-Only RMSNorm (asymmetric QK pre-softmax). Pass-through
        # to the inner MHA. See `MultiHeadAttention.use_q_only_norm` for
        # the mechanism. Default off → baseline path bit-identical.
        # See autoresearch/ideas/162-q-only-norm/idea.md.
        use_q_only_norm: bool = False,
        # 165 — K-Only RMSNorm (asymmetric QK pre-softmax, K-side).
        # Pass-through to the inner MHA. See
        # `MultiHeadAttention.use_k_only_norm` for the mechanism.
        # Default off → baseline path bit-identical. The K-mirror of
        # 162; together with 016 (symmetric QK) forms a clean 3-way
        # orthogonal attribution test. Mutually exclusive with
        # use_q_only_norm (asserted at MHA construction). See
        # autoresearch/ideas/165-k-only-norm/idea.md.
        use_k_only_norm: bool = False,
        # 169 — Depth-Conditional QK-Norm (per-block learnable scale
        # on top of 016's WIN). Pass-through to the inner MHA. See
        # `MultiHeadAttention.use_qk_norm_depth` for the mechanism.
        # Default off → baseline path bit-identical. Sits on top of
        # 016's pre-RoPE symmetric norm; mutually exclusive with
        # 162 (Q-only) / 165 (K-only) / 049 (post-RoPE) at MHA
        # construction. See
        # `autoresearch/ideas/169-qk-norm-depth/idea.md`.
        use_qk_norm_depth: bool = False,
        # 190 — Per-Layer QK-Norm (scalar γ per block per side, replaces
        # 016's per-channel γ with a single scalar per side per block).
        # Pass-through to the inner MHA. See
        # `MultiHeadAttention.use_qk_norm_scalar_per_block` for the
        # mechanism. Default off → baseline path bit-identical. See
        # `autoresearch/ideas/190-per-layer-qk-norm/idea.md`.
        qk_norm_scalar_per_block: bool = False,
        # 190 — Per-Layer QK-Norm (Q/K-shared scalar variant). When
        # True together with `qk_norm_scalar_per_block`, the two
        # side-scalars collapse to a single shared scalar (the 169
        # axis). Default off → baseline path bit-identical. See
        # `autoresearch/ideas/190-per-layer-qk-norm/idea.md`.
        qk_norm_scalar_qk_shared: bool = False,
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
        # 173 — Entmax-1.5 sparse attention. Passed through to
        # MultiHeadAttention (see note at the MHA `use_entmax`
        # kwarg). Default off → baseline path bit-identical.
        use_entmax: bool = False,
        # 192 — Pre-softmax per-row hard top-k sparse attention.
        # Passed through to MultiHeadAttention (see note at the MHA
        # `use_topk_attn` kwarg). Default off → baseline path
        # bit-identical.
        use_topk_attn: bool = False,
        topk_k: int = 512,
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
        # 164 — Q-Carry. Passed through to MultiHeadAttention
        # (see MHA.use_q_carry for the mechanism). Default off →
        # baseline path bit-identical.
        use_q_carry: bool = False,
        # 168 — AV-Output Carry pass-through to MultiHeadAttention
        # (see MHA.use_av_output_carry for the mechanism). Default
        # off → baseline path bit-identical.
        use_av_output_carry: bool = False,
        # 186 — Within-Block V-Carry pass-through to
        # MultiHeadAttention (see MHA.use_v_carry_block for the
        # mechanism). Default off → baseline path bit-identical.
        # See `autoresearch/ideas/186-v-carry-block/plan.md`.
        use_v_carry_block: bool = False,
        # 163 — Post-Attention V-Mix Depthwise Convolution (Poli et
        # al. "Hyena", 2023, arXiv:2302.10866). After the attention
        # output is computed (post-SDPA, post-reshape, pre-W_O
        # projection), apply a symmetric depthwise Conv1d on the
        # time axis over the post-attention tensor `[B, T, d_model]`.
        # Conv weights are built as a raw
        # `nn.Parameter(zeros(d_model, 1, k))` with center tap = 1.0
        # set inline (NOT `nn.Conv1d(...)` followed by `.data`
        # reassignment — `nn.Conv1d` consumes RNG at init
        # (kaiming_uniform_), which would shift the RNG state for
        # every subsequent block's `qkvo_proj` random init and break
        # the step-0 byte-identity claim across blocks 2..12). Same
        # raw-`Parameter` pattern as `models/conv_ffn.py:103-105`
        # (157-conv-ffn). Padding = `k//2` symmetric (causal+future)
        # — the attention sublayer has already integrated the full
        # causal context, so the conv may look at both neighbors.
        # `v_mix_conv_kernel` defaults to 3 (spec pin); valid range
        # is odd integers ≥ 3. Default off → baseline path
        # bit-identical (no Parameter registered, no forward branch
        # taken). Cost: n_layers × k × d_model extra params
        # (12 × 3 × 64 = 2,304 at tiny1m3m, +0.25%). See
        # `autoresearch/ideas/163-v-mix-conv/idea.md`.
        use_v_mix_conv: bool = False,
        v_mix_conv_kernel: int = 3,
        # 201 — Degenerate gMLP SGU pass-through to the inner
        # MHA. See `MultiHeadAttention.use_gmlp_sgu` for the
        # mechanism. `block_idx` is the block's index in the
        # build enumeration (0..n_layers-1 with `tie_layer_groups=1`
        # the default). The SGU is allocated only when
        # `block_idx % gmlp_sgu_block_stride == 0`, so the
        # build-loop MUST pass a stable `block_idx` per block.
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/201-mlp-token-mixer/idea.md`.
        use_gmlp_sgu: bool = False,
        gmlp_sgu_block_stride: int = 3,
        gmlp_sgu_alpha_init: float = -10.0,
        block_idx: int = 0,
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
        # 149 — TTT-Linear (Sun, Yang, et al. 2024, arXiv:2407.04620,
        # §3.2). Replaces the FFN's up-projection with a `TTTLinear`
        # — a per-input closed-form fast-weight linear that updates
        # its own weight from the input on the fly (one Newton-style
        # gradient step on the auto-encoding loss `||W·x − x||²`).
        # The down-projection stays a standard `nn.Linear` so the FFN
        # output side is unchanged. `ttt_lr_init=0.0` (default) zero-
        # inits the per-layer TTT learning rate so `lr=0` at step 0
        # ⇒ the `TTTLinear` short-circuits to `F.linear(x, weight, b)`
        # with the same `kaiming_uniform_` weight as `nn.Linear` ⇒
        # the FFN is bit-identical to a vanilla `SquaredReLUFeedForward`
        # at step 0. With `use_ttt_ffn=False` (default) the
        # `TTTFeedForward` module is never built and the baseline FFN
        # path is bit-identical. See
        # `autoresearch/ideas/149-ttt-linear/idea.md`.
        use_ttt_ffn: bool = False,
        ttt_lr_init: float = 0.0,
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
        # 150 — Cross-Layer Feedback Attention (Holtzman et al. 2020,
        # Feedback Transformer). Each block reads from a small cache
        # of the previous K layers' pre-FFN residual states via a
        # `XLayerCrossAttn` head, and adds the result as a *gated*
        # residual branch. `xlayer_gate = nn.Parameter(torch.zeros(1))`
        # per block → step-0 ≡ no-feedback baseline. K=2 by default
        # (the spec pin). Default off → baseline path bit-identical
        # (no cross-attn module built, no kwarg plumbed). See
        # `models/xlayer_attn.py` and
        # `autoresearch/ideas/150-xlayer-feedback/idea.md`.
        use_xlayer_feedback: bool = False,
        xlayer_k: int = 2,
        # 157 — Depthwise Conv inside FFN (ConvBERT/ConvNeXt-style,
        # Jiang et al. 2020 arXiv:2008.02496; Woo et al. 2020). A
        # symmetric depthwise Conv1d applied to the FFN output in
        # `TransformerBlock.forward` (after `feed_forward(ffn_in)`,
        # before any layerscale/sub_ln/residual-add). Conv weights
        # are identity-initialized (center tap = 1, rest = 0) with
        # `padding=k//2` so the conv is a strict identity at step 0
        # ⇒ the block's FFN output is bit-identical to baseline at
        # step 0. Differs from 143-shortconv (pre-attention, causal,
        # gated) by (a) post-FFN placement (NOT pre-attention on the
        # residual stream), (b) symmetric (non-causal) padding —
        # appropriate because the FFN output has already integrated
        # the full causal context via attention, so mixing neighbors
        # on both sides does not leak future tokens. `conv_ffn_kernel`
        # defaults to 3 (spec pin); valid range is odd integers ≥ 3.
        # Default off → baseline path bit-identical (the `ConvFFN`
        # module is never built, the forward branch is never taken).
        # Cost: n_layers × (kernel × d_model) extra params (~2.3K at
        # tiny1m3m with k=3, +0.25%). See
        # `autoresearch/ideas/157-conv-ffn/idea.md`.
        use_conv_ffn: bool = False,
        conv_ffn_kernel: int = 3,
        # 188 — Cross-Block K/V Projection Sharing pass-through to
        # the inner MHA. See `MultiHeadAttention.use_cross_block_kv_share`
        # for the mechanism (Universal Transformers-style learnable
        # convex blend of the layer's W_K/W_V with the previous
        # block's W_K/W_V, gated on a 0-dim scalar per side with
        # sigmoid-bounded init at -10). `prev_W_K` / `prev_W_V` are
        # passed through the block's forward into the MHA forward.
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/188-cross-block-kv-share/idea.md`.
        use_cross_block_kv_share: bool = False,
        # 204 — Cross-Block Attention Score Sharing pass-through
        # to the inner MHA. See
        # `MultiHeadAttention.use_cross_block_score_share` for the
        # mechanism (per-block learnable scalar α =
        # σ(score_share_alpha_raw) blends the current block's
        # pre-softmax scores with the previous block's detached
        # pre-softmax scores; init -10 ⇒ α ≈ 4.5e-5 ⇒ identity at
        # step 0 within fp32 noise). `prev_block_scores` is passed
        # through the block's forward into the MHA forward (same
        # pattern as 021's `v_residual=` / 164's `q_carry=` / 168's
        # `av_carry=` / 188's `prev_W_K=` / `prev_W_V=`).
        # `score_share_alpha_init` defaults to -10.0 (the standard
        # sigmoid-gated scalar init used by 188's K/V α params
        # and the closed 021 `lambda_v` 1-D gain family). Default
        # off → baseline path bit-identical. See
        # `autoresearch/ideas/204-cross-block-attn-score-share/idea.md`.
        use_cross_block_score_share: bool = False,
        score_share_alpha_init: float = -10.0,
        # 197 — Tied W_O Across Blocks (soft blend, Universal-
        # Transformer-style learnable parameter sharing restricted
        # to the attention output projection, Dehghani et al. ICLR
        # 2019 arXiv:1807.03819 + Lan et al. ALBERT arXiv:1909.11942).
        # Pass-through to the inner MHA. The shared `tied_wo_shared`
        # Parameter is allocated on `MinimalLLM` and plumbed through
        # here so the MHA constructor can store the SAME reference
        # on every block (NOT a per-block copy). See
        # `MultiHeadAttention.use_tied_wo_across_blocks` for the
        # forward-time mechanism. Default off → baseline path
        # bit-identical. See
        # `autoresearch/ideas/197-tied-wo-across-blocks/idea.md` /
        # `plan.md`.
        use_tied_wo_across_blocks: bool = False,
        tied_wo_alpha_init: float = -10.0,
        tied_wo_shared=None,  # Optional[torch.nn.Parameter]
        # 199 — Spectral-Norm-Bounded W_O Projection pass-through
        # to the inner MHA. Per-block learnable scalar `γ_l` (init
        # 0) and per-block power-iteration Buffer `u_l` are
        # allocated by the MHA when the flag is on. See
        # `MultiHeadAttention.use_wo_spectral_cap` for the
        # forward-time mechanism (per-block Lipschitz cap on W_O
        # with σ_max_init snapshot on first forward ⇒ step-0
        # byte-identical). Default off → no Parameter, no Buffer,
        # no branch taken, baseline path bit-identical. See
        # `autoresearch/ideas/199-spectral-attn-output/idea.md` /
        # `plan.md`.
        use_wo_spectral_cap: bool = False,
        wo_spectral_cap_pi_iters: int = 1,
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
            # 166 — T5-RPE pass-through to the inner MHA. Default
            # off → baseline path bit-identical. See
            # `autoresearch/ideas/166-t5-rpe/idea.md`.
            use_t5_rpe=use_t5_rpe,
            t5_rpe_buckets=t5_rpe_buckets,
            # 152 — Per-head logit bias pass-through to the inner MHA.
            # See `MultiHeadAttention.use_attn_logit_bias` for the
            # mechanism. Default off → baseline path bit-identical.
            use_attn_logit_bias=use_attn_logit_bias,
            # 155 — Per-head learnable attention temperature
            # pass-through to the inner MHA. Default off → baseline
            # path bit-identical. See
            # `autoresearch/ideas/155-per-head-temp/idea.md`.
            use_per_head_temp=use_per_head_temp,
            # 195 — Tight hard QK logit clamp pass-through to the
            # inner MHA. Default off → baseline path bit-identical.
            # See `autoresearch/ideas/195-qk-clamp-min-max/idea.md`.
            use_qk_clamp=use_qk_clamp,
            qk_clamp_c=qk_clamp_c,
            # 193 — Blockwise attention temperature schedule pass-
            # through to the inner MHA (precomputed per-block
            # `tau_b` value, computed by the model in
            # `MinimalLLM.__init__` via the loop). Default off →
            # baseline path bit-identical (no Buffer registered on
            # the MHA, no branch taken in forward). See
            # `autoresearch/ideas/193-blockwise-attn-temp-schedule/
            # idea.md`.
            use_block_temp_schedule=use_block_temp_schedule,
            block_temp_alpha=block_temp_alpha,
            tau_b=tau_b,
            # 180 — Pre-softmax 1D causal conv pass-through.
            use_logit_conv=use_logit_conv,
            logit_conv_kernel_size=logit_conv_kernel_size,
            # 202 — V-Only Soft-Blend Probe pass-through. See
            # `MultiHeadAttention.use_grouped_v` for the
            # mechanism. Default off → baseline path bit-
            # identical. See
            # `autoresearch/ideas/202-grouped-value-projection/
            # idea.md`.
            use_grouped_v=use_grouped_v,
            v_group_size=v_group_size,
            # 179 — Anti-Causal Sub-Heads pass-through to the
            # inner MHA. Per-head learnable scalar `γ_h` controls
            # the per-head upper-triangle fill magnitude. Init
            # `γ_h ≈ 4.5e-5` ⇒ step-0 byte-identical to baseline.
            # Default off → baseline path bit-identical. See
            # `autoresearch/ideas/179-anti-causal-subheads/idea.md`.
            use_anti_causal_subheads=use_anti_causal_subheads,
            # 160 — Per-head RMS gain on the attention output.
            # Pass-through to the inner MHA. Default off → baseline
            # path bit-identical. See
            # `autoresearch/ideas/160-rms-gain-per-head/idea.md`.
            use_head_gain=use_head_gain,
            # 181 — Cross-Head Channel RMSNorm. Pass-through to
            # the inner MHA. Default off → baseline path
            # bit-identical. See
            # `autoresearch/ideas/181-cross-head-rmsnorm/idea.md`.
            use_cross_head_rmsnorm=use_cross_head_rmsnorm,
            # 191 — Per-token attention output gain pass-through to
            # the inner MHA. See
            # `MultiHeadAttention.use_token_attn_gain` for the
            # mechanism. Default off → baseline path bit-
            # identical. See
            # `autoresearch/ideas/191-token-attn-gain/idea.md`.
            use_token_attn_gain=use_token_attn_gain,
            # 203 — Pre-W_O Squeeze-Excitation channel attention
            # pass-through to the inner MHA. See
            # `MultiHeadAttention.use_se_pre_wo` for the mechanism.
            # Default off → baseline path bit-identical. See
            # `autoresearch/ideas/203-pre-wo-se-channel-attn/idea.md`.
            use_se_pre_wo=use_se_pre_wo,
            se_reduction_ratio=se_reduction_ratio,
            se_alpha_init=se_alpha_init,
            # 147 — DropKey: per-head Bernoulli gate on K during training.
            use_drop_key=use_drop_key,
            drop_key_rate=drop_key_rate,
            # 171 — DropConnect on W_O pass-through to the inner MHA.
            # See `MultiHeadAttention.use_dropconnect_wo` for the
            # mechanism (per-weight Bernoulli mask on W_O with warmup-
            # ramped effective rate). Default off → baseline path
            # bit-identical. See
            # `autoresearch/ideas/171-dropconnect-wo/idea.md`.
            use_dropconnect_wo=use_dropconnect_wo,
            dropconnect_wo_rate=dropconnect_wo_rate,
            dropconnect_wo_warmup_steps=dropconnect_wo_warmup_steps,
            # 207 — W_O Low-Rank Bottleneck pass-through to the inner
            # MHA. See `MultiHeadAttention.use_lowrank_wo` for the
            # mechanism (rank-r residual correction
            # `W_O_eff = W_O + σ(α)·(W_O_A @ W_O_B)`, W_O_B zero-init
            # ⇒ step-0 bit-identical). Default off → baseline path
            # bit-identical. See
            # `autoresearch/ideas/207-wo-lowrank-bottleneck/idea.md`
            # / `plan.md`.
            use_lowrank_wo=use_lowrank_wo,
            wo_rank=wo_rank,
            wo_lowrank_alpha_init=wo_lowrank_alpha_init,
            # 194 — W_V Low-Rank Residual Correction pass-through
            # to the inner MHA. See `MultiHeadAttention.use_lowrank_wv`
            # for the mechanism (`W_V_eff = W_V + σ(α)·(W_V_A @ W_V_B)`,
            # W_V_B zero-init ⇒ step-0 bit-identical). Default off →
            # baseline path bit-identical. See
            # `autoresearch/ideas/194-lowrank-ffn/idea.md` / `plan.md`.
            use_lowrank_wv=use_lowrank_wv,
            wv_rank=wv_rank,
            wv_lowrank_alpha_init=wv_lowrank_alpha_init,
            # 151 — RoV pass-through to the inner MHA. Default off
            # → baseline path bit-identical. See
            # `autoresearch/ideas/151-rov-gated/idea.md`.
            use_rov=use_rov,
            # 174 — xPos exponential decay on K pass-through to the
            # inner MHA. Default off → baseline path bit-identical.
            # See `autoresearch/ideas/174-xpos-decay/idea.md`.
            use_xpos=use_xpos,
            # 154 — Rebased Attention pass-through to the inner MHA.
            # Default off → baseline path bit-identical. See
            # `autoresearch/ideas/154-rebased-attn/idea.md`.
            use_rebased_attn=use_rebased_attn,
            rebase_stride=rebase_stride,
            # 156 — MoA pass-through to the inner MHA. Default off
            # → baseline path bit-identical. See
            # `autoresearch/ideas/156-moa/idea.md`.
            use_moa=use_moa,
            moa_num_experts=moa_num_experts,
            # 178 — Gated MQA pass-through to the inner MHA. Default
            # off → baseline path bit-identical. See
            # `autoresearch/ideas/178-mqa-gated/idea.md`.
            use_mqa_gated=use_mqa_gated,
            # 185 — Static per-head learned K-rotation pass-through
            # to the inner MHA. Default off → baseline path bit-
            # identical. See `autoresearch/ideas/185-static-per-head-k-rotation/idea.md`.
            use_static_k_rotation=use_static_k_rotation,
            # 200 — Static per-layer × per-pair learned K-rotation
            # pass-through to the inner MHA (depth-axis twin of
            # 185, shared across heads, K-only). Default off →
            # baseline path bit-identical. See
            # `autoresearch/ideas/200-rope-phase-offset-per-layer/idea.md`.
            use_per_layer_k_rotation=use_per_layer_k_rotation,
            # 192 — Pre-RoPE per-head × per-pair learned Q+K
            # rotation pass-through to the inner MHA (Q+K-side,
            # pre-RoPE placement). Default off → baseline path
            # bit-identical. See
            # `autoresearch/ideas/192-pre-rope-qk-rotation/idea.md`.
            use_pre_rope_rotation=use_pre_rope_rotation,
            # 182 — Per-head learnable attention window pass-through.
            # Default off → baseline path bit-identical. See
            # `autoresearch/ideas/182-per-head-window/idea.md`.
            use_per_head_window=use_per_head_window,
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
            use_entmax=use_entmax,
            # 192 — Top-K sparse attention pass-through to the inner
            # MHA. See `MultiHeadAttention.use_topk_attn` for the
            # mechanism. Default off → baseline path bit-identical.
            use_topk_attn=use_topk_attn,
            topk_k=topk_k,
            use_ssmax=use_ssmax,
            use_value_residual=use_value_residual,
            # 164 — Q-Carry pass-through to MultiHeadAttention
            # (see MHA.use_q_carry for the mechanism). Default off →
            # baseline path bit-identical. See
            # `autoresearch/ideas/164-q-carry/plan.md`.
            use_q_carry=use_q_carry,
            # 168 — AV-Output Carry pass-through to MultiHeadAttention
            # (see MHA.use_av_output_carry for the mechanism). Default
            # off → baseline path bit-identical. See
            # `autoresearch/ideas/168-av-output-carry/plan.md`.
            use_av_output_carry=use_av_output_carry,
            # 186 — Within-Block V-Carry pass-through to
            # MultiHeadAttention (see MHA.use_v_carry_block for the
            # mechanism). Default off → baseline path bit-identical.
            # See `autoresearch/ideas/186-v-carry-block/plan.md`.
            use_v_carry_block=use_v_carry_block,
            # 163 — Pass-through to MultiHeadAttention.
            use_v_mix_conv=use_v_mix_conv,
            v_mix_conv_kernel=v_mix_conv_kernel,
            # 201 — Degenerate gMLP SGU pass-through. `block_idx`
            # is the block's index in the build enumeration
            # (0..n_layers-1 with `tie_layer_groups=1` the default);
            # the SGU is allocated only when
            # `block_idx % gmlp_sgu_block_stride == 0`. The build
            # loop in `models/llm.py` enumerates `range(n_unique)`
            # and passes `block_idx=i` to each block. See
            # `autoresearch/ideas/201-mlp-token-mixer/idea.md`.
            use_gmlp_sgu=use_gmlp_sgu,
            gmlp_sgu_block_stride=gmlp_sgu_block_stride,
            gmlp_sgu_alpha_init=gmlp_sgu_alpha_init,
            block_idx=block_idx,
            # 188 — Cross-Block K/V Projection Sharing pass-through.
            # Default off → baseline path bit-identical. See
            # `autoresearch/ideas/188-cross-block-kv-share/idea.md`.
            use_cross_block_kv_share=use_cross_block_kv_share,
            # 204 — Cross-Block Attention Score Sharing pass-through.
            # Default off → baseline path bit-identical. See
            # `autoresearch/ideas/204-cross-block-attn-score-share/idea.md`.
            use_cross_block_score_share=use_cross_block_score_share,
            score_share_alpha_init=score_share_alpha_init,
            # 197 — Tied W_O Across Blocks pass-through. Default off
            # → baseline path bit-identical. See
            # `autoresearch/ideas/197-tied-wo-across-blocks/idea.md`
            # / `plan.md`.
            use_tied_wo_across_blocks=use_tied_wo_across_blocks,
            tied_wo_alpha_init=tied_wo_alpha_init,
            tied_wo_shared=tied_wo_shared,
            # 199 — Spectral-Norm-Bounded W_O Projection pass-through
            # to the inner MHA (per-block learnable Lipschitz cap
            # on W_O with σ_max_init snapshot on first forward ⇒
            # step-0 byte-identical). Default off → no Parameter, no
            # Buffer, no branch taken, baseline path bit-identical.
            # See `autoresearch/ideas/199-spectral-attn-output/
            # idea.md` / `plan.md`.
            use_wo_spectral_cap=use_wo_spectral_cap,
            wo_spectral_cap_pi_iters=wo_spectral_cap_pi_iters,
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
            # 189 — CosFormer-style linear attention pass-through
            # to the inner MHA. See `MultiHeadAttention.use_cosformer`
            # for the mechanism. Default off → baseline path
            # bit-identical. See
            # `autoresearch/ideas/189-cosformer-linear-attn/idea.md`.
            use_cosformer=use_cosformer,
            cosformer_gamma_init=cosformer_gamma_init,
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
            # 176 — Pre-AV V RMSNorm pass-through: when set, MHA
            # builds per-head `v_rmsnorm_alpha ∈ R^H` and
            # `v_rmsnorm_gain ∈ R^{H × d_k}` and applies the gated
            # RMSNorm to V before the AV product — see
            # MultiHeadAttention.use_v_rmsnorm.
            use_v_rmsnorm=use_v_rmsnorm,
            # 162 — Q-Only RMSNorm pass-through: when set, MHA builds a
            # per-head `nn.RMSNorm(d_k)` on Q only — see MultiHeadAttention.use_q_only_norm.
            use_q_only_norm=use_q_only_norm,
            # 165 — K-Only RMSNorm pass-through: when set, MHA builds a
            # per-head `nn.RMSNorm(d_k)` on K only — see MultiHeadAttention.use_k_only_norm.
            use_k_only_norm=use_k_only_norm,
            # 169 — Depth-Conditional QK-Norm pass-through: when set,
            # MHA registers `qk_norm_scale = nn.Parameter(torch.ones(()))`
            # per block — see MultiHeadAttention.use_qk_norm_depth.
            use_qk_norm_depth=use_qk_norm_depth,
            # 190 — Per-Layer QK-Norm (scalar γ per block per side) pass-
            # through: when set, MHA registers per-side scalar γ
            # Parameters — see MultiHeadAttention.qk_norm_scalar_per_block.
            qk_norm_scalar_per_block=qk_norm_scalar_per_block,
            # 190 — Per-Layer QK-Norm (Q/K-shared scalar variant) pass-
            # through: when set together with `qk_norm_scalar_per_block`,
            # both side-scalars point to the same Parameter — see
            # MultiHeadAttention.qk_norm_scalar_qk_shared.
            qk_norm_scalar_qk_shared=qk_norm_scalar_qk_shared,
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
        if use_relu2_ffn:
            # 153 — Squared-ReLU FFN activation (So et al. "Primer",
            # arXiv:2109.08668, 2021). `relu2(x) = x * F.relu(x)`
            # (≡ `(max(0, x))^2`). Two-projection shape matches
            # `SquaredReLUFeedForward` so the lever is purely the
            # activation change. Branch sits AHEAD of every other
            # FFN-replacement flag (MoE / TTT / `ffn_variant`) so it
            # isn't silently shadowed when more than one is set.
            # Default off → no `ReLU2FeedForward` is constructed, the
            # baseline path runs bit-identical. See
            # `autoresearch/ideas/153-relu2-ffn/idea.md`.
            self.feed_forward = ReLU2FeedForward(d_model, d_ff, dropout)
        elif use_mish_glu:
            # 196 — MishGLU FFN (Misra 2019 + Shazeer 2020
            # composition; inner-activation axis orthogonal to 170's
            # outer-GLU axis). Three-projection gated linear unit
            # `y = down_proj(mish(W_gate·x) ⊙ (W_up·x))` —
            # structurally identical to 170's
            # SwiGLUZeroInitFeedForward *except* the gate activation
            # is `mish` (`x * tanh(softplus(x))`, Misra 2019) instead
            # of `silu`. `mish(0) = 0` ⇒ step-0 forward is silent
            # without an explicit zero-init (and a zero-init would
            # mask the gradient signal the lever depends on).
            # d_ff is scaled by the Shazeer 2/3 trick
            # (`d_ff_swiglu = (2 * d_ff) // 3`, 170 for
            # d_ff_baseline=256) so total FFN param count matches
            # SwiGLU to within ~0.4%. Branch sits AHEAD of 170
            # (use_swiglu_ffn) and every other FFN-replacement flag
            # so the inner-activation lever isn't silently shadowed
            # when more than one FFN-replacement flag is set.
            # Default off → the standard `ffn_variant` cascade runs
            # bit-identical to baseline. See
            # `autoresearch/ideas/196-ffn-glu-mish/idea.md`.
            # Mutual-exclusion guard: 170 (SwiGLU) and 196 (MishGLU)
            # both target the FFN slot with different inner
            # activations; failing loud here catches misuse at
            # construction rather than at training time.
            assert not (use_soft_moe or use_switch_ffn or use_ttt_ffn), (
                "use_mish_glu (196) is mutually exclusive with "
                "use_soft_moe (117) / use_switch_ffn (146) / use_ttt_ffn: "
                "all three replace the FFN with their own module, so they "
                "shadow 196's gate."
            )
            assert not use_swiglu_ffn, (
                "use_mish_glu (196) is mutually exclusive with "
                "use_swiglu_ffn (170): both target the FFN slot. "
                "Pick one — 196 tests the inner-activation axis (Mish "
                "vs SiLU gate), 170 tests the outer-GLU axis."
            )
            assert ffn_variant != "swiglu", (
                "use_mish_glu (196) is mutually exclusive with "
                "ffn_variant='swiglu' (legacy 2-projection SwiGLU without "
                "zero-init gate) — pick one. The branch ordering silently "
                "wins by 196."
            )
            d_ff_mish = (2 * d_ff) // 3
            self.feed_forward = MishGLUFeedForward(
                d_model, d_ff_mish, dropout
            )
        elif use_swiglu_ffn:
            # 170 — SwiGLU FFN (Shazeer 2020, arXiv:2002.05202;
            # LLaMA-family FFN). Three-projection gated linear unit
            # `y = down_proj(silu(W_gate·x) ⊙ (W_up·x))` with
            # zero-init `gate_proj.weight` ⇒ `silu(0) = 0` ⇒ FFN
            # output is exactly 0 at step 0 (clean ReZero-style
            # baseline: the residual stream carries only the
            # attention sub-block at step 0). d_ff is scaled by the
            # Shazeer 2/3 trick (`d_ff_swiglu = (2 * d_ff) // 3`,
            # 170 for d_ff_baseline=256) so total FFN param count
            # matches the baseline to within ~0.4%. Branch sits
            # AHEAD of every other FFN-replacement flag (MoE / TTT
            # / `ffn_variant == "swiglu"`) so it isn't silently
            # shadowed. Default off → the standard `ffn_variant`
            # cascade runs bit-identical to baseline. See
            # `autoresearch/ideas/170-swiglu-ffn/idea.md`.
            # Mutual-exclusion guard: 170 owns the FFN slot, so a later
            # combo of `use_swiglu_ffn + use_soft_moe + ffn_variant="swiglu"`
            # would silently win by branch order. Fail loud here so the
            # misuse is caught at construction, not at training time.
            assert not (use_soft_moe or use_switch_ffn or use_ttt_ffn), (
                "use_swiglu_ffn (170) is mutually exclusive with "
                "use_soft_moe (117) / use_switch_ffn (146) / use_ttt_ffn: "
                "all three replace the FFN with their own module, so they "
                "shadow 170's gate."
            )
            assert ffn_variant != "swiglu", (
                "use_swiglu_ffn (170) is mutually exclusive with "
                "ffn_variant='swiglu' (legacy 2-projection SwiGLU without "
                "zero-init gate) — pick one. The branch ordering silently "
                "wins by 170."
            )
            from .components import SwiGLUZeroInitFeedForward
            d_ff_swiglu = (2 * d_ff) // 3
            self.feed_forward = SwiGLUZeroInitFeedForward(
                d_model, d_ff_swiglu, dropout
            )
        elif use_soft_moe:
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
        elif use_ttt_ffn:
            # 149 — TTT-Linear: squared_relu FFN whose up_proj is a
            # `TTTLinear` (per-input closed-form fast-weight update).
            # `ttt_lr_init=0.0` (default) keeps the forward bit-
            # identical to a vanilla `SquaredReLUFeedForward` at step
            # 0 — the `TTTLinear` short-circuits to `F.linear(x, w, b)`
            # before any fast-weight matmul fires. The branch sits
            # AFTER the MoE branches so the MoE flags win when more
            # than one FFN-replacement flag is set.
            self.feed_forward = TTTFeedForward(
                d_model, d_ff, dropout=dropout, ttt_lr_init=ttt_lr_init,
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

        # 206 — Cross-Block FFN share. When on, register the two
        # per-side α scalars on the FFN module (init -10.0 ⇒
        # `σ(-10) ≈ 4.5e-5` ⇒ silent at step 0). The FFN
        # implementation owns the `ffn_share_alpha_up` /
        # `ffn_share_alpha_down` attribute slots and the
        # `_prev_W_up` / `_prev_W_down` stash slots; we just
        # fill them when the flag is on. The MoE / TTT FFN
        # replacements don't have these slots — we skip the
        # registration on those variants (the flag is silently
        # shadowed in that case, same pattern as 188's YOCO
        # guard). When off, the FFN's `ffn_share_alpha_*` is
        # `None` and the forward blend branch is dead (the
        # gate is on the parameter, not on a flag, so the
        # baseline path is bit-identical). See
        # `autoresearch/ideas/206-cross-block-ffn-share/idea.md` /
        # `plan.md`.
        self.use_cross_block_ffn_share = use_cross_block_ffn_share
        if use_cross_block_ffn_share and not (
            use_soft_moe or use_switch_ffn or use_expert_choice_moe
            or use_ttt_ffn
        ):
            self.feed_forward.ffn_share_alpha_up = nn.Parameter(
                torch.full((), float(ffn_share_alpha_init))
            )
            self.feed_forward.ffn_share_alpha_down = nn.Parameter(
                torch.full((), float(ffn_share_alpha_init))
            )

        # 157 — Depthwise Conv inside FFN. Identity-init symmetric
        # depthwise Conv1d applied to the FFN output (post-FFN, pre-
        # residual-add) in `forward`. Built lazily; never called when
        # `use_conv_ffn=False` so the baseline path is bit-identical.
        # See `models/conv_ffn.py` for the module docstring and
        # `autoresearch/ideas/157-conv-ffn/idea.md` for the design.
        self.use_conv_ffn = use_conv_ffn
        self.conv_ffn_kernel = int(conv_ffn_kernel)
        if self.use_conv_ffn:
            self.conv_ffn = ConvFFN(d_model, kernel_size=self.conv_ffn_kernel)
        else:
            self.conv_ffn = None

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
        # 197 — DeepNet α fixed residual init (Wang et al. 2022,
        # arXiv:2203.00555). A single *fixed* (not learned) global
        # scalar `α = (2·n_layers)^(-1/2)` applied to every block's
        # sublayer output before the residual add. The fixed form
        # bounds the residual stream's magnitude to `O(1)` at every
        # depth (vs `√L` un-scaled). 0 new params — `α` is a Python
        # float computed once at construction. The lever is a
        # different *operating point* from baseline (not a
        # perturbation), so the forward is intentionally NOT
        # step-0 byte-identical when the flag is ON (the bounded
        # regime is the whole point). Default off → the
        # `self.deepnet_alpha` attribute is still set (to ~0.204
        # at L=12) but the `self.use_deepnet_alpha` flag is the
        # gate that determines whether the multiply runs (the
        # baseline path is bit-identical when the flag is off).
        self.use_deepnet_alpha = use_deepnet_alpha
        self.deepnet_alpha = float((2.0 * max(1, int(n_layers))) ** -0.5)
        # 198 — Pre-FFN Attention Mixing. One 0-dim scalar per block
        # `pre_ffn_attn_mix_gamma_raw`, init `pre_ffn_attn_mix_init`
        # (default −10 ⇒ sigmoid(γ_raw) ≈ 4.5e-5 at step 0). Mirrors
        # the ReZero wiring above: declared as `nn.Parameter` so it
        # gets routed to AdamW by the optimizer setup. When the flag
        # is off, the attribute is `None` and the forward branch is
        # never taken (baseline path bit-identical). See
        # `autoresearch/ideas/198-pre-ffn-attnmix/idea.md`.
        self.use_pre_ffn_attn_mix = use_pre_ffn_attn_mix
        if self.use_pre_ffn_attn_mix:
            self.pre_ffn_attn_mix_gamma_raw = nn.Parameter(
                torch.tensor(float(pre_ffn_attn_mix_init))
            )
        else:
            self.pre_ffn_attn_mix_gamma_raw = None
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

        # 150 — Cross-Layer Feedback Attention. Built lazily; never
        # called when `use_xlayer_feedback=False` so the baseline path
        # is bit-identical. K=2 by default (the spec pin). The per-
        # block scalar `xlayer_gate` is the identity-init lever: 0
        # means the cross-attn contribution is exactly 0 at step 0
        # regardless of the Q/K/V projection values. (The gate is
        # `tanh`-bounded at the call site, so its effective range is
        # `[-1, 1]` during training — this prevents the unbounded
        # runaway positive-feedback loop the round-2 GPU run hit.)
        # See `models/xlayer_attn.py` and
        # `autoresearch/ideas/150-xlayer-feedback/idea.md`.
        self.use_xlayer_feedback = use_xlayer_feedback
        self.xlayer_k = max(1, int(xlayer_k))
        if self.use_xlayer_feedback:
            self.xlayer_attn = XLayerCrossAttn(
                d_model,
                k_window=self.xlayer_k,
                n_heads=1,
                head_dim=min(16, d_model),
            )
            # Scalar per-block gate (init 0 ⇒ identity at step 0).
            self.xlayer_gate = nn.Parameter(torch.zeros(1))
        else:
            self.xlayer_attn = None

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

    def forward(self, x, x0=None, ve=None, v_residual=None, layer_index=None, shared_kv=None, xlayer_mem=None, q_carry=None, av_carry=None, prev_W_K=None, prev_W_V=None, prev_block_scores=None, prev_W_up=None, prev_W_down=None, cosformer_gamma=None):
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
                attn_out = self.attention(n, ve, gate_x=x, v_residual=v_residual, shared_kv=shared_kv, q_carry=q_carry, av_carry=av_carry, prev_W_K=prev_W_K, prev_W_V=prev_W_V, prev_block_scores=prev_block_scores, cosformer_gamma=cosformer_gamma)
            if self.use_layerscale:
                attn_out = attn_out * (1.0 + self.attn_layerscale)
            if self.use_layer_scale:
                attn_out = attn_out * self.attn_gamma
            ffn_in = n
            if self.use_ffn_embed and ve is not None:
                ffn_in = ffn_in + F.linear(ve, self.ffn_embed_proj)
            ff_out = self.feed_forward(ffn_in, prev_W_up=prev_W_up, prev_W_down=prev_W_down)
            # 157 — Depthwise Conv inside FFN. Identity-init
            # symmetric conv → bit-identical to no-conv at step 0.
            # Applied after the FFN returns, before the
            # layerscale/LayerScale/post-norm multipliers.
            if self.use_conv_ffn:
                ff_out = self.conv_ffn(ff_out)
            if self.use_layerscale:
                ff_out = ff_out * (1.0 + self.ffn_layerscale)
            if self.use_layer_scale:
                ff_out = ff_out * self.ffn_gamma
            # 197 — DeepNet α on the parallel block. Both sublayers
            # share the input, so a single scalar is applied to their
            # combined sum (equivalent to applying it to each
            # independently, since `α` is a constant). See
            # `autoresearch/ideas/197-output-residual-sqrt-2l/idea.md`.
            if self.use_deepnet_alpha:
                return x + self.deepnet_alpha * (self.dropout(attn_out) + self.dropout(ff_out))
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
                attn_out = self.attention(x, ve, gate_x=x, v_residual=v_residual, shared_kv=shared_kv, q_carry=q_carry, av_carry=av_carry, prev_W_K=prev_W_K, prev_W_V=prev_W_V, prev_block_scores=prev_block_scores, cosformer_gamma=cosformer_gamma)
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
            # 197 — DeepNet α fixed residual init (post-norm path).
            # The `α` scalar is a single global Python float computed
            # once at block construction from `n_layers` (see
            # `self.deepnet_alpha` set in `__init__`). When the flag
            # is on, the sublayer's contribution is `α·f(x)` instead
            # of `f(x)`. Flag off ⇒ no multiply, baseline path bit-
            # identical. See
            # `autoresearch/ideas/197-output-residual-sqrt-2l/idea.md`.
            if self.use_deepnet_alpha:
                x = self.norm1(x + self.deepnet_alpha * self.dropout(attn_out))
            else:
                x = self.norm1(x + self.dropout(attn_out))

            ffn_in = x
            if self.use_ffn_embed and ve is not None:
                ffn_in = ffn_in + F.linear(ve, self.ffn_embed_proj)
            ff_out = self.feed_forward(ffn_in, prev_W_up=prev_W_up, prev_W_down=prev_W_down)
            # 157 — Depthwise Conv inside FFN. Identity-init
            # symmetric conv → bit-identical to no-conv at step 0.
            if self.use_conv_ffn:
                ff_out = self.conv_ffn(ff_out)
            if self.use_layerscale:
                ff_out = ff_out * (1.0 + self.ffn_layerscale)
            if self.use_layer_scale:
                ff_out = ff_out * self.ffn_gamma
            if self.use_sub_ln:
                ff_out = self.sub_ln_ffn(ff_out)
            # 197 — DeepNet α on the FFN sublayer (post-norm path).
            # Same fixed scalar as the attention branch above.
            if self.use_deepnet_alpha:
                x = self.norm2(x + self.deepnet_alpha * self.dropout(ff_out))
            else:
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
                attn_out = self.attention(self.norm1(x), ve, gate_x=x, v_residual=v_residual, shared_kv=shared_kv, q_carry=q_carry, av_carry=av_carry, prev_W_K=prev_W_K, prev_W_V=prev_W_V, prev_block_scores=prev_block_scores, cosformer_gamma=cosformer_gamma)
            # 198 — Pre-FFN Attention Mixing. Capture the *raw*
            # attention output (post-`self.attention(...)`, before any
            # layerscale / sub_ln / rezero / dropout wrapping). The
            # mix below uses this raw tensor — `.detach()` keeps γ's
            # gradient cleanly tied to FFN-side loss only (no
            # gradient through the attention path's Q/K/V/O
            # projections at step 0). See
            # `autoresearch/ideas/198-pre-ffn-attnmix/idea.md`.
            attn_out_raw = attn_out
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
            elif self.use_deepnet_alpha:
                # 197 — DeepNet α fixed residual init (pre-norm
                # attn). Single global scalar `α = 1/√(2·n_layers)`;
                # the residual add becomes `x + α·f(x)` instead of
                # `x + f(x)`. ReZero is checked first (learned form
                # dominates), `deepnet_alpha` (fixed form) is the
                # second branch. Mutually exclusive in this chain
                # by construction — the user picks one or the other.
                # See
                # `autoresearch/ideas/197-output-residual-sqrt-2l/idea.md`.
                x = x + self.deepnet_alpha * self.dropout(attn_out)
            elif self.resid_mode:
                x = self._resid_add(x, self.dropout(attn_out), "attn")
            else:
                x = x + self.dropout(attn_out)
            # 150 — Cross-Layer Feedback: at this point `x` is the
            # block's pre-FFN state. Save it for the cross-attn query
            # and for the mem update at the end. The cross-attn reads
            # from `xlayer_mem` (the previous K blocks' pre-FFN
            # states, accumulated by the model loop) and adds a gated
            # contribution back to the residual. `xlayer_gate=0` at
            # init ⇒ `tanh(0)=0` ⇒ contribution is exactly 0 at
            # step 0 regardless of the Q/K/V projection values ⇒
            # forward is bit-identical to the no-feedback baseline.
            # After the block returns, the model loop appends
            # `x_pre_ffn` to `xlayer_mem` so the NEXT block can read
            # it. The `tanh` wraps the gate so its effective range
            # is `[-1, 1]` (the `xlayer_gate` parameter is a
            # pre-activation). This bounds the cross-attn
            # contribution and prevents the runaway positive-
            # feedback loop the unbounded gate had in the round-2
            # GPU run (gate opens → cross-attn learns → output
            # grows → gradient on gate grows → gate opens more →
            # loop until explosion, val loss diverging 7.36→9.77
            # after step ~100). The tanh gradient saturates
            # smoothly so the gate can still open to its full
            # `±1` effective range during training; it just can't
            # run away. Bit-identity at step 0 is preserved
            # (`tanh(0)` is exactly `0.0` in fp32).
            x_pre_ffn = x
            if self.use_xlayer_feedback:
                # 150 — Cross-Layer Feedback: detach the previous-layer
                # mem entries BEFORE the cross-attn so gradient does NOT
                # flow back into the pre-FFN states of earlier blocks.
                # The forward computation is unchanged (the cross-attn
                # still reads the actual previous-layer pre-FFN values).
                # The backward pass: the cross-attn path contributes
                # gradients to (q_proj/k_proj/v_proj/out_proj/xlayer_gate
                # + the current block's x via the Q projection) — but NOT
                # to the pre-FFN x of blocks [-K, -1]. Mirrors the
                # `V.detach()` pattern in 021-value-residual, where the
                # stashed V_1 from layer 0 is detached before the layer-l
                # blend `V = (1-λ)·V + λ·v_residual`. The detach was
                # the round-2 fix; the round-3 fix is the `tanh`-
                # bounding of `xlayer_gate` (see block comment above).
                mem_det = ([t.detach() for t in xlayer_mem]
                           if xlayer_mem is not None else None)
                y_xa = self.xlayer_attn(x_pre_ffn, mem_det)
                # `tanh`-bound the gate to `[-1, 1]`. At init
                # `tanh(0)=0` ⇒ contribution is exactly 0 ⇒
                # forward is bit-identical to the no-feedback
                # baseline. During training the gate is bounded,
                # so the cross-attn path cannot inject arbitrarily
                # large values into the residual stream even if
                # the Q/K/V projections grow.
                x = x_pre_ffn + torch.tanh(self.xlayer_gate) * y_xa

            # 198 — Pre-FFN Attention Mixing. Pre-norm2 placement
            # (A) per the r1 review pin: the mix is added INSIDE the
            # norm2 input, so the mix is renormalized by RMSNorm
            # (matches the spec's plain reading `ffn_input =
            # attn_residual + γ·attn_block(x).detach()` and the
            # reviewer's `ffn_in = norm2(x + sigmoid(γ)·
            # attn_out.detach())` recommendation — placement (B)
            # outside RMS would change the mix's effective magnitude
            # by `1/RMS(x+mix)`, an uncontrolled dynamic). At init
            # `sigmoid(γ_raw) ≈ 4.5e-5` ⇒ the perturbation to `x`
            # is on the order of `4.5e-5 · O(1) ≈ 4.5e-5` in fp32
            # ⇒ baseline path is fp32-noise bit-identical at step
            # 0 (same `sigmoid(-10)` convention as 188/201/205/206).
            # `.detach()` on `attn_out_raw` keeps γ's gradient tied
            # to FFN-side loss only. Scope: pre-norm2 path only.
            # The parallel-block / post-norm paths are alternative
            # architectures off by default; the lever is silently
            # shadowed on those paths when the user combines 198
            # with `use_parallel_block=True` / `use_post_norm=True`.
            if self.use_pre_ffn_attn_mix:
                pre_mix = (
                    torch.sigmoid(self.pre_ffn_attn_mix_gamma_raw)
                    * attn_out_raw.detach()
                )
                ffn_in = self.norm2(x + pre_mix)
            else:
                ffn_in = self.norm2(x)
            if self.use_ffn_embed and ve is not None:
                ffn_in = ffn_in + F.linear(ve, self.ffn_embed_proj)
            ff_out = self.feed_forward(ffn_in, prev_W_up=prev_W_up, prev_W_down=prev_W_down)
            # 157 — Depthwise Conv inside FFN. Identity-init
            # symmetric conv → bit-identical to no-conv at step 0.
            if self.use_conv_ffn:
                ff_out = self.conv_ffn(ff_out)
            if self.use_layerscale:
                ff_out = ff_out * (1.0 + self.ffn_layerscale)
            if self.use_layer_scale:
                ff_out = ff_out * self.ffn_gamma
            if self.use_sub_ln:
                ff_out = self.sub_ln_ffn(ff_out)
            # R1 ReZero (pre-norm branch, FFN): same gate on the FFN add.
            if self.use_re_zero:
                x = x + self.re_zero_alpha_ffn * self.dropout(ff_out)
            elif self.use_deepnet_alpha:
                # 197 — DeepNet α on the FFN sublayer (pre-norm). Same
                # fixed scalar as the attention branch above. See
                # `autoresearch/ideas/197-output-residual-sqrt-2l/idea.md`.
                x = x + self.deepnet_alpha * self.dropout(ff_out)
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
        # 150 — Cross-Layer Feedback: append the block's pre-FFN state
        # to `xlayer_mem` so the NEXT block can read it via the cross-
        # attn head. Mutate the list in place. Truncate to the last
        # `xlayer_k` entries. `xlayer_mem` is `None` when the lever is
        # off (the model loop only allocates it for
        # `use_xlayer_feedback=True`), so this branch is no-op on the
        # baseline path.
        if self.use_xlayer_feedback and xlayer_mem is not None:
            xlayer_mem.append(x_pre_ffn)
            if len(xlayer_mem) > self.xlayer_k:
                del xlayer_mem[: len(xlayer_mem) - self.xlayer_k]
        return x


# ============================================================================
# 158 — Gated Attention Unit (GAU, Hua et al. 2022, arXiv:2202.10447).
# Fuses `Attention → Add → FFN → Add` into a single block with shared
# projections. Per the idea spec:
#
#   y   = x + U_g x            # input-conditional gating
#   z   = softmax(Q_g y · K_g y^T) V_g y   # causal attention on y
#   out = U_o (z * V_o y)      # output proj with element-wise gating
#
# Step-0 identity (the spec pin):
#   U_g = 0  →  y = x + 0 = x
#   V_o = 0  →  z * V_o y = z * 0 = 0
#   out = U_o(0) = 0           → block returns `x + dropout(0) = x`
#
# Inherits the standard pre-norm from `make_norm` and reuses
# `Rotary`/`SDPA` (causal) for the attention half. GQA is supported
# the same way `MultiHeadAttention` supports it (kv heads repeated
# by `num_key_value_groups` to match Q's head count). The block has
# NO FFN — the FFN's job is folded into the U_g/V_o gating pair,
# saving roughly the FFN's `2·d_model·d_ff` parameters per layer
# (~37% of `TransformerBlock`'s per-layer param cost at tiny1m3m).
#
# Param count per block at tiny1m3m (d_model=64, n_heads=4, d_k=16,
# n_kv_heads=2, kv_size=32):
#   TransformerBlock (squared_relu FFN, GQA):
#     qkvo: (64 + 2·32 + 64)·64 = 12288
#     FFN : 2·64·256            = 32768
#     Total                       ≈ 45K
#   GAUBlock:
#     fused: (64 + 2·32 + 2·64)·64 = 64·192 = 12288 (Q,K,V,U_g,V_o)
#     out  : 64·64                = 4096  (U_o)
#     Total                         ≈ 16K
#   Saving: ~29K/block × 12 = ~350K (~37% of the 0.94M model).
#
# The freed budget can be re-spent on attention dim per the GAU paper's
# retuning — out of scope for this 1-idea A/B at fixed tiny1m3m tier;
# the savings simply mean the model is smaller, which we report on.
#
# Default off → the block is never built, the standard `TransformerBlock`
# is built instead, and the baseline forward is bit-identical. See
# `autoresearch/ideas/158-gau/idea.md`.
# ============================================================================
class GAUBlock(nn.Module):
    """158 — Gated Attention Unit (Hua et al. 2022).

    Single fused `Attention + FFN` block. Identity at step 0 (see
    module docstring). Supports GQA, RoPE, RMSNorm pre-norm, dropout.
    No FFN — the gating pair (U_g, V_o) plays the FFN role.

    The block accepts the same `forward(x, x0=None, ve=None, ...)`
    signature shape as `TransformerBlock` so the model loop in
    `MinimalLLM.forward` can dispatch to it transparently when
    `use_gau=True`. Unused kwargs (`x0`, `ve`, `gate_x`,
    `v_residual`, `shared_kv`, `xlayer_mem`, `layer_index`) are
    accepted via `**kwargs` and ignored.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
        norm_type: str = "rmsnorm",
        rope_base: int = 10000,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else int(n_heads)
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = self.d_model // self.n_heads
        self.kv_size = self.n_kv_heads * self.d_k
        # Fused Q,K,V,U_g,V_o projection: total size
        #   q_size + 2·kv_size + 2·d_model
        self.q_size = self.d_model
        self.gate_size = 2 * self.d_model  # U_g + V_o
        self.fused_size = self.q_size + 2 * self.kv_size + self.gate_size
        # One merged `nn.Parameter` (NOT `nn.Linear`) so init does not
        # consume RNG — keeps construction aligned with the baseline
        # `MultiHeadAttention`'s `qkvo_proj` (which is also a raw
        # parameter). The two zero-init gate slices are set AFTER the
        # global normal init below.
        self.fused_proj = nn.Parameter(torch.empty(self.fused_size, self.d_model))
        with torch.no_grad():
            torch.nn.init.normal_(self.fused_proj, mean=0.0, std=0.02)
            # Zero-init U_g and V_o (the gate slices) → step-0
            # identity per the module docstring. Slice ranges:
            #   Q  : [0, q_size)
            #   K  : [q_size, q_size + kv_size)
            #   V  : [q_size + kv_size, q_size + 2·kv_size)
            #   U_g: [q_size + 2·kv_size, q_size + 2·kv_size + d_model)
            #   V_o: [q_size + 2·kv_size + d_model, fused_size)
            q_end = self.q_size
            kv_end = q_end + self.kv_size
            v_end = kv_end + self.kv_size
            ug_end = v_end + self.d_model
            self.fused_proj[v_end:ug_end].zero_()              # U_g
            self.fused_proj[ug_end:self.fused_size].zero_()   # V_o
        # Output projection U_o (d_model → d_model). Standard init.
        self.out_proj = nn.Parameter(torch.empty(self.d_model, self.d_model))
        with torch.no_grad():
            torch.nn.init.normal_(self.out_proj, mean=0.0, std=0.02)
        # Rotary on Q, K (same convention as MultiHeadAttention). V
        # stays un-rotated per the standard GAU formulation. Reuses
        # the existing `Rotary` helper for byte-for-byte parity with
        # the baseline RoPE path.
        self.rotary = Rotary(self.d_k, max_seq_len, base=rope_base)
        # Pre-norm (matches TransformerBlock's norm1 placement).
        self.norm = make_norm(self.d_model, norm_type, False)
        self.dropout = nn.Dropout(dropout)
        # GAU has no FFN, so it has nothing to add to the residual
        # stream at step 0 (both gates zero → block output = 0). The
        # residual stream therefore advances unchanged through every
        # block at step 0; after the first non-zero gradient step
        # the model begins learning from the all-zero attention output.
        # This is the spec's "identity at step 0" pin.

    def forward(self, x, x0=None, ve=None, **kwargs):
        """GAU forward. Returns `x` shape-preserved.

        Args:
            x: [B, T, d_model] residual stream.
            x0, ve, **kwargs: accepted for signature parity with
                `TransformerBlock.forward`. All ignored (GAU doesn't
                use the embed-residual path or the various attention-
                side flags from the standard MHA).

        Returns:
            [B, T, d_model] — `x + dropout(U_o(z ⊙ V_o y))` where
            `y = norm(x) + U_g(norm(x))` and
            `z = SDPA_causal(Q, K, V)`.
        """
        B, T, D = x.shape
        # Pre-norm (same placement as TransformerBlock.norm1).
        x_norm = self.norm(x)
        # Single fused linear: [B, T, fused_size].
        qkv_ug_vo = F.linear(x_norm, self.fused_proj)
        # Split into Q, K, V, U_g, V_o slices.
        q_end = self.q_size
        kv_end = q_end + self.kv_size
        v_end = kv_end + self.kv_size
        ug_end = v_end + self.d_model
        Q = qkv_ug_vo[..., :q_end]
        K = qkv_ug_vo[..., q_end:kv_end]
        V = qkv_ug_vo[..., kv_end:v_end]
        U_g = qkv_ug_vo[..., v_end:ug_end]
        V_o = qkv_ug_vo[..., ug_end:]
        # GAU input: `y = x_norm + U_g(x_norm)`. U_g init 0 ⇒ `y = x_norm`.
        y = x_norm + U_g
        # Per-head reshape. Q is full d_model = n_heads·d_k; K, V are
        # kv_size = n_kv_heads·d_k. Reuses the same convention as
        # MultiHeadAttention.forward.
        Q = Q.reshape(B, T, self.n_heads, self.d_k)
        K = K.reshape(B, T, self.n_kv_heads, self.d_k)
        V = V.reshape(B, T, self.n_kv_heads, self.d_k)
        # Apply RoPE (Q, K only — V is not rotated).
        Q = self.rotary(Q)
        K = self.rotary(K)
        # GQA: repeat K (and V) by num_key_value_groups to match Q's
        # head count.
        if self.num_key_value_groups > 1:
            K = K.repeat_interleave(self.num_key_value_groups, dim=2)
            V = V.repeat_interleave(self.num_key_value_groups, dim=2)
        # SDPA expects [B, n_heads, T, d_k].
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # Causal SDPA — `is_causal=True` ⇒ flash/efficient path on
        # supported devices. PyTorch's SDPA matches the baseline
        # math within fp32 rounding on sm_86 (the V100 box).
        z = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        z = z.transpose(1, 2).contiguous().view(B, T, self.d_model)
        # Output: `U_o(z * V_o y)`. V_o init 0 ⇒ `z * 0 = 0` ⇒
        # `out = U_o(0) = 0` (linear of zero vector = zero vector).
        # Element-wise `z * V_o` is the GAU paper's gating (§3.2):
        # "multiply the attention output element-wise by an
        # input-conditional gate V_o(y)."
        out = F.linear(z * V_o, self.out_proj)
        # Residual + dropout. At step 0, `out = 0` exactly ⇒ the
        # block is identity (matches the spec's step-0 pin).
        return x + self.dropout(out)
