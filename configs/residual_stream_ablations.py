"""Residual-stream ablation configs (docs/research/residual_stream/plan.md).

The locus: how each sublayer enters the residual add `x = a·x + g·f(x)`
(baseline a=1, g=1). Every lever is one cheap scalar/vector knob — no matmuls,
no new modules.

Wiring: O1 ReZero uses the dedicated `use_re_zero` flag; the rest set one
`resid_mode` string, applied in `TransformerBlock._resid_add` (models/layers.py)
on both the attention and FFN adds. Identity-init (step-0 == baseline) unless
marked "own control".

Run: `python train_llm.py --config_class configs.residual_stream_ablations.<Name> --seed 0`
Control = clean Tiny1M3MConfig.
"""
from dataclasses import dataclass

from .llm_config import Tiny1M3MConfig


# ---- Batch 1: learnable gates (identity-init) ----
@dataclass
class Tiny1M3MReZeroConfig(Tiny1M3MConfig):
    """R1 — per-sublayer scalar gate x = x + α·f, α=0 init."""
    use_re_zero: bool = True

@dataclass
class Tiny1M3MReZeroVecConfig(Tiny1M3MConfig):
    """ReZero with a length-d_model α vector (α=0). Per-channel cousin of R1."""
    resid_mode: str = "rezero_vec"

@dataclass
class Tiny1M3MReZeroSharedConfig(Tiny1M3MConfig):
    """One α=0 shared by both sublayer adds (half the params of R1)."""
    resid_mode: str = "rezero_shared"

@dataclass
class Tiny1M3MReZeroInitOneConfig(Tiny1M3MConfig):
    """Learnable scalar gate, init 1 (identity) — A/B vs α=0 init."""
    resid_mode: str = "rezero_init_one"

@dataclass
class Tiny1M3MBranchGainConfig(Tiny1M3MConfig):
    """Per-sublayer scalar branch gain g, init 1."""
    resid_mode: str = "branch_gain"

@dataclass
class Tiny1M3MBranchGainVecConfig(Tiny1M3MConfig):
    """Per-channel branch gain g⊙f, g vec init 1 (A/B vs use_layerscale)."""
    resid_mode: str = "branch_gain_vec"

@dataclass
class Tiny1M3MInputScaleConfig(Tiny1M3MConfig):
    """Scalar gain on the residual input: a·x + f, a=1."""
    resid_mode: str = "input_scale"

@dataclass
class Tiny1M3MInputScaleVecConfig(Tiny1M3MConfig):
    """Per-channel gain on the residual input: a⊙x + f, a=1."""
    resid_mode: str = "input_scale_vec"

@dataclass
class Tiny1M3MResidMixConfig(Tiny1M3MConfig):
    """R2 — learned in/out mix a·x + b·f, scalars a=1,b=1."""
    resid_mode: str = "resid_mix"

@dataclass
class Tiny1M3MResidMixVecConfig(Tiny1M3MConfig):
    """R2 vector — a,b as length-d_model vectors (a=1,b=1)."""
    resid_mode: str = "resid_mix_vec"

@dataclass
class Tiny1M3MHighwaySigmoidConfig(Tiny1M3MConfig):
    """R3 — x + 2σ(s)·f, s=0 → gate=1 (bounded branch gate)."""
    resid_mode: str = "highway_sigmoid"

@dataclass
class Tiny1M3MTanhGateConfig(Tiny1M3MConfig):
    """x + (1+tanh(s))·f, s=0 → gate=1."""
    resid_mode: str = "tanh_gate"

@dataclass
class Tiny1M3MSoftplusGateConfig(Tiny1M3MConfig):
    """x + softplus(s)·f ≈ 1 (positive-constrained branch gate)."""
    resid_mode: str = "softplus_gate"

@dataclass
class Tiny1M3MClampGateConfig(Tiny1M3MConfig):
    """x + clamp(g,0,2)·f, g=1."""
    resid_mode: str = "clamp_gate"

@dataclass
class Tiny1M3MDoubleGateConfig(Tiny1M3MConfig):
    """x + g2·(g1·f), g1=g2=1 (overparameterized scalar gate)."""
    resid_mode: str = "double_gate"

# ---- Batch 2: sublayer-specific gates ----
@dataclass
class Tiny1M3MAttnOnlyReZeroConfig(Tiny1M3MConfig):
    """ReZero on the attention add only (FFN add stays baseline)."""
    resid_mode: str = "attn_only_rezero"

@dataclass
class Tiny1M3MFFNOnlyReZeroConfig(Tiny1M3MConfig):
    """ReZero on the FFN add only (attention add stays baseline)."""
    resid_mode: str = "ffn_only_rezero"

@dataclass
class Tiny1M3MAttnOnlyGainConfig(Tiny1M3MConfig):
    """Per-channel branch gain on the attention add only, g=1."""
    resid_mode: str = "attn_only_gain"

@dataclass
class Tiny1M3MFFNOnlyGainConfig(Tiny1M3MConfig):
    """Per-channel branch gain on the FFN add only, g=1."""
    resid_mode: str = "ffn_only_gain"

# ---- Batch 3: fixed / init-schedule scales (own control — not identity) ----
@dataclass
class Tiny1M3MFixedHalfConfig(Tiny1M3MConfig):
    """x + 0.5·f (fixed, not learned). Own control."""
    resid_mode: str = "fixed_half"

@dataclass
class Tiny1M3MFixedSqrt2Config(Tiny1M3MConfig):
    """(1/√2)x + (1/√2)f — variance-preserving add. Own control."""
    resid_mode: str = "fixed_sqrt2"

@dataclass
class Tiny1M3MFixedDeepNormConfig(Tiny1M3MConfig):
    """R5 — x + (1/√(2L))·f (DeepNorm-style branch shrink). Own control."""
    resid_mode: str = "fixed_deepnorm"

@dataclass
class Tiny1M3MReZeroInitHalfConfig(Tiny1M3MConfig):
    """Learnable scalar gate, init 0.5. Own control."""
    resid_mode: str = "rezero_init_half"

@dataclass
class Tiny1M3MInputScaleHalfConfig(Tiny1M3MConfig):
    """0.5·x + f (fixed input shrink). Own control."""
    resid_mode: str = "input_scale_half"

@dataclass
class Tiny1M3MEMAResidConfig(Tiny1M3MConfig):
    """0.9·x + 0.1·f (EMA-style residual). Own control."""
    resid_mode: str = "ema_resid"

@dataclass
class Tiny1M3MConvexHalfConfig(Tiny1M3MConfig):
    """0.5·x + 0.5·f (convex average). Own control."""
    resid_mode: str = "convex_half"

# ---- Batch 4: stochastic regularizers (identity at eval) ----
@dataclass
class Tiny1M3MBranchDropout05Config(Tiny1M3MConfig):
    """Dropout p=0.05 on the branch f before the add (eval = identity)."""
    resid_mode: str = "branch_dropout05"

@dataclass
class Tiny1M3MBranchDropout10Config(Tiny1M3MConfig):
    """Dropout p=0.10 on the branch f before the add (eval = identity)."""
    resid_mode: str = "branch_dropout10"

@dataclass
class Tiny1M3MStochDepth10Config(Tiny1M3MConfig):
    """R7 — stochastic depth: drop a sublayer p=0.1 at train (eval = identity)."""
    resid_mode: str = "stoch_depth10"

@dataclass
class Tiny1M3MStochDepth20Config(Tiny1M3MConfig):
    """Stochastic depth p=0.2 at train (eval = identity)."""
    resid_mode: str = "stoch_depth20"
