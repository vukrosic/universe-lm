"""Attention-output (W_O) ablation configs (docs/research/attention_output/plan.md).

The C1 locus: a cheap op on the **post-softmax** attention output `attn_out`
[B,H,T,D], applied just before the heads are merged and `W_O` mixes them back
into the residual stream. Distinct from the input-side `talking_heads_q` (which
mixes *scores* pre-softmax).

Wiring: every lever below sets one `out_op` string (or `use_talking_heads_out`
for O1). The op is applied at the [B,H,T,D] choke point in
`MultiHeadAttention._apply_output_op` (models/layers.py), so it covers every
attention branch. Identity-init (step-0 == baseline) unless marked "own control".

Run: `python train_llm.py --config_class configs.attention_output_ablations.<Name> --seed 0`
Control = clean Tiny1M3MConfig.
"""
from dataclasses import dataclass

from .llm_config import Tiny1M3MConfig


# ---- Batch 1: cross-head mixing on the output (the headline C1 lever) ----
@dataclass
class Tiny1M3MTalkingHeadsOutConfig(Tiny1M3MConfig):
    """O1 — post-softmax cross-head mix, M=I init (no-op)."""
    use_talking_heads_out: bool = True

@dataclass
class Tiny1M3MHeadMixReparamConfig(Tiny1M3MConfig):
    """O1' — cross-head mix as M = I + Δ, Δ=0 init (additive reparam of O1)."""
    out_op: str = "headmix_reparam"

@dataclass
class Tiny1M3MHeadMixLowRank1Config(Tiny1M3MConfig):
    """Cross-head mix M = I + uvᵀ (rank-1, zero-init) — cheap talking-heads."""
    out_op: str = "headmix_lowrank1"

@dataclass
class Tiny1M3MHeadMixLowRank2Config(Tiny1M3MConfig):
    """Cross-head mix M = I + UVᵀ (rank-2, zero-init)."""
    out_op: str = "headmix_lowrank2"

# ---- Batch 2: per-head output gains (gate-family A/B) ----
@dataclass
class Tiny1M3MOutputHeadGateConfig(Tiny1M3MConfig):
    """O2 — per-head scalar gain g_h on the output, init 1."""
    out_op: str = "head_gate"

@dataclass
class Tiny1M3MOutputHeadGateReparamConfig(Tiny1M3MConfig):
    """O2' — per-head gain as *(1+g_h), g=0 init (A/B vs shipped attn_output_gate)."""
    out_op: str = "head_gate_reparam"

@dataclass
class Tiny1M3MOutputHeadGateSigmoidConfig(Tiny1M3MConfig):
    """Bounded per-head gain 2σ(s_h), s=0 → 1."""
    out_op: str = "head_gate_sigmoid"

@dataclass
class Tiny1M3MOutputHeadGateSoftplusConfig(Tiny1M3MConfig):
    """Per-head gain softplus(w_h) ≈ 1 (positive-constrained)."""
    out_op: str = "head_gate_softplus"

@dataclass
class Tiny1M3MOutputHeadGateClampConfig(Tiny1M3MConfig):
    """Per-head gain clamped to [0,2], init 1."""
    out_op: str = "head_gate_clamp"

@dataclass
class Tiny1M3MOutputHeadTempConfig(Tiny1M3MConfig):
    """O3' — per-head temperature: divide output by τ_h, init 1."""
    out_op: str = "head_temp"

@dataclass
class Tiny1M3MOutputPerHDGainConfig(Tiny1M3MConfig):
    """Per-(head,channel) gain G[h,d], init 1."""
    out_op: str = "per_hd_gain"

@dataclass
class Tiny1M3MOutputLayerScaleConfig(Tiny1M3MConfig):
    """Per-channel (shared across heads) gain c_d, init 1."""
    out_op: str = "out_layerscale"

@dataclass
class Tiny1M3MOutputScaleConfig(Tiny1M3MConfig):
    """Single global scalar on the output, init 1."""
    out_op: str = "out_scale"

# ---- Batch 3: biases / affine ----
@dataclass
class Tiny1M3MOutputBiasConfig(Tiny1M3MConfig):
    """O6 — per-head additive bias b_h, init 0."""
    out_op: str = "out_bias"

@dataclass
class Tiny1M3MOutputBiasChannelConfig(Tiny1M3MConfig):
    """Per-(head,channel) bias b[h,d], init 0."""
    out_op: str = "out_bias_channel"

@dataclass
class Tiny1M3MOutputHeadAffineConfig(Tiny1M3MConfig):
    """Per-head affine: a_h·out + b_h (a=1, b=0)."""
    out_op: str = "head_affine"

@dataclass
class Tiny1M3MOutputPerHDAffineConfig(Tiny1M3MConfig):
    """Per-(head,channel) affine: G·out + b (G=1, b=0)."""
    out_op: str = "per_hd_affine"

# ---- Batch 4: post-softmax nonlinearities (own control — not identity) ----
@dataclass
class Tiny1M3MOutputTanhConfig(Tiny1M3MConfig):
    """O4 — tanh(α·out), α=1 (saturating smooth-clip). Own control."""
    out_op: str = "out_tanh"

@dataclass
class Tiny1M3MOutputSoftplusConfig(Tiny1M3MConfig):
    """O5 — softplus(out), output forced ≥ 0. Own control."""
    out_op: str = "out_softplus"

@dataclass
class Tiny1M3MOutputGeluConfig(Tiny1M3MConfig):
    """GELU on the output. Own control."""
    out_op: str = "out_gelu"

@dataclass
class Tiny1M3MOutputSwishConfig(Tiny1M3MConfig):
    """SiLU/Swish on the output. Own control."""
    out_op: str = "out_swish"

@dataclass
class Tiny1M3MOutputSignedSqrtConfig(Tiny1M3MConfig):
    """sign(out)·√|out| — magnitude compression. Own control."""
    out_op: str = "out_signed_sqrt"

@dataclass
class Tiny1M3MOutputSoftcapConfig(Tiny1M3MConfig):
    """Gemma-style 30·tanh(out/30) softcap. Own control (≈id for small |out|)."""
    out_op: str = "out_softcap30"

@dataclass
class Tiny1M3MOutputClampConfig(Tiny1M3MConfig):
    """clamp(out, -10, 10). Own control (≈id within range)."""
    out_op: str = "out_clamp10"

# ---- Batch 5: post-softmax normalization (own control) ----
@dataclass
class Tiny1M3MOutputRMSConfig(Tiny1M3MConfig):
    """O3 — rms-normalize per head over D, then per-(H,D) gain. Own control."""
    out_op: str = "out_rms"

@dataclass
class Tiny1M3MOutputL2NormConfig(Tiny1M3MConfig):
    """Unit-normalize each head's D-vector, then per-head gain. Own control."""
    out_op: str = "out_l2norm"

@dataclass
class Tiny1M3MOutputCenterConfig(Tiny1M3MConfig):
    """Subtract per-head mean over D. Own control."""
    out_op: str = "out_center"

# ---- Batch 6: stochastic regularizers (identity at eval) ----
@dataclass
class Tiny1M3MOutputDropout10Config(Tiny1M3MConfig):
    """Dropout p=0.1 on the post-softmax output (eval = identity)."""
    out_op: str = "out_dropout10"

@dataclass
class Tiny1M3MOutputDropout20Config(Tiny1M3MConfig):
    """Dropout p=0.2 on the post-softmax output (eval = identity)."""
    out_op: str = "out_dropout20"

@dataclass
class Tiny1M3MOutputHeadDropout10Config(Tiny1M3MConfig):
    """Drop whole heads p=0.1 at train, rescale (eval = identity)."""
    out_op: str = "head_dropout10"
