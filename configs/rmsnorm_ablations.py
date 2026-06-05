"""RMSNorm ablation configs (docs/research/rmsnorm/plan.md).

One class per lever. Each flips a single `norm_type` string that `make_norm`
parses in `models/layers.py`; the chosen norm flows to norm1, norm2 and the
final `self.norm` via the existing `llm.py` plumbing (no edits to `llm.py`).

Three groups:
  * Batch 1-3 — RMSNorm lever family (`rmsnorm_<lever>`): one extra param or
    one free trick on `y = g·x/√(mean(x²)+eps)`. Identity-init (step-0 == plain
    RMSNorm) unless marked "own control".
  * Batch 4 — existing norm zoo, run here as a ranking sweep (no new code).

Run: `python train_llm.py --config_class configs.rmsnorm_ablations.<Name> --seed 0`
Control = clean Tiny1M3MConfig (plain RMSNorm).
"""
from dataclasses import dataclass

from .llm_config import Tiny1M3MConfig


# ---- Batch 1: free / 1-param tweaks (identity-init) ----
@dataclass
class Tiny1M3MReparamGainConfig(Tiny1M3MConfig):
    """N1 — g = 1 + g0, learn g0 (g0=0). Identity init."""
    norm_type: str = "rmsnorm_reparam_gain"

@dataclass
class Tiny1M3MRMSBiasConfig(Tiny1M3MConfig):
    """N2 — y = g·x_norm + b (b=0)."""
    norm_type: str = "rmsnorm_bias"

@dataclass
class Tiny1M3MGlobalTempConfig(Tiny1M3MConfig):
    """N3 — y = g·x/(rms·τ), scalar τ=1."""
    norm_type: str = "rmsnorm_temp"

@dataclass
class Tiny1M3MPartialNormMixConfig(Tiny1M3MConfig):
    """N4 — y = g·x_norm + λ·x, scalar λ=0."""
    norm_type: str = "rmsnorm_partial_mix"

@dataclass
class Tiny1M3MLearnableFloorConfig(Tiny1M3MConfig):
    """N5 — rms = √(mean(x²)+eps+c²), c=0."""
    norm_type: str = "rmsnorm_floor"

@dataclass
class Tiny1M3MSoftplusGainConfig(Tiny1M3MConfig):
    """N7 — g = softplus(w), w s.t. g≈1."""
    norm_type: str = "rmsnorm_softplus_gain"

# ---- Batch 2: structural tweaks ----
@dataclass
class Tiny1M3MPartialNormVectorConfig(Tiny1M3MConfig):
    """N8 — y = g·x_norm + λ⊙x, λ a length-d vector (λ=0)."""
    norm_type: str = "rmsnorm_partial_vec"

@dataclass
class Tiny1M3MGroupRMS4Config(Tiny1M3MConfig):
    """N9 — per-group rms, G=4 (different op, own control)."""
    norm_type: str = "rmsnorm_group4"

@dataclass
class Tiny1M3MGroupRMS8Config(Tiny1M3MConfig):
    """N9 — per-group rms, G=8 (different op, own control)."""
    norm_type: str = "rmsnorm_group8"

@dataclass
class Tiny1M3MStopGradRMSConfig(Tiny1M3MConfig):
    """N10 — y = g·x / detach(rms): no gradient through the denominator."""
    norm_type: str = "rmsnorm_stopgrad"

@dataclass
class Tiny1M3MAsymGainConfig(Tiny1M3MConfig):
    """N11 — separate gain for x>0 and x≤0 (both init 1)."""
    norm_type: str = "rmsnorm_asym"

@dataclass
class Tiny1M3MGainClamp50Config(Tiny1M3MConfig):
    """N12 — bound the gain to [0.5, 1.5]."""
    norm_type: str = "rmsnorm_clamp50"

@dataclass
class Tiny1M3MLearnableEpsConfig(Tiny1M3MConfig):
    """N14 — learn the eps inside the sqrt (init = default eps)."""
    norm_type: str = "rmsnorm_learnable_eps"

# ---- Batch 3: norm-replacement probes ----
@dataclass
class Tiny1M3MDynTanhConfig(Tiny1M3MConfig):
    """N15 — y = g·tanh(α·x), no statistics (DyT). Own control."""
    norm_type: str = "rmsnorm_dyntanh"

@dataclass
class Tiny1M3MDoubleNormConfig(Tiny1M3MConfig):
    """N16 — apply RMSNorm twice."""
    norm_type: str = "rmsnorm_double"

@dataclass
class Tiny1M3MCenterMixConfig(Tiny1M3MConfig):
    """N17 — y = g·(x − μ·x̄)/rms, scalar μ=0 (RMS↔LayerNorm knob)."""
    norm_type: str = "rmsnorm_centermix"

@dataclass
class Tiny1M3MScaledGainInitConfig(Tiny1M3MConfig):
    """N6 — init gain 0.5 not 1.0. Own control."""
    norm_type: str = "rmsnorm_scaled_init"

# ---- Batch 4: existing norm zoo (no new code — ranking sweep) ----
@dataclass
class Tiny1M3MNormPNorm15Config(Tiny1M3MConfig):
    norm_type: str = "pnorm1.5"

@dataclass
class Tiny1M3MNormPNorm175Config(Tiny1M3MConfig):
    norm_type: str = "pnorm1.75"

@dataclass
class Tiny1M3MNormPNorm3Config(Tiny1M3MConfig):
    norm_type: str = "pnorm3"

@dataclass
class Tiny1M3MNormClip2Config(Tiny1M3MConfig):
    norm_type: str = "clipnorm2"

@dataclass
class Tiny1M3MNormClip3Config(Tiny1M3MConfig):
    norm_type: str = "clipnorm3"

@dataclass
class Tiny1M3MNormClip4Config(Tiny1M3MConfig):
    norm_type: str = "clipnorm4"

@dataclass
class Tiny1M3MNormChannelScaleConfig(Tiny1M3MConfig):
    norm_type: str = "channelscale"

@dataclass
class Tiny1M3MNormManhattanConfig(Tiny1M3MConfig):
    norm_type: str = "manhattan"

@dataclass
class Tiny1M3MNormCenterConfig(Tiny1M3MConfig):
    norm_type: str = "center"

@dataclass
class Tiny1M3MNormCenteredL1Config(Tiny1M3MConfig):
    norm_type: str = "centeredl1"

@dataclass
class Tiny1M3MNormManifoldConfig(Tiny1M3MConfig):
    norm_type: str = "manifold"

@dataclass
class Tiny1M3MNormMedianConfig(Tiny1M3MConfig):
    norm_type: str = "median"

@dataclass
class Tiny1M3MNormPeakConfig(Tiny1M3MConfig):
    norm_type: str = "peak"

@dataclass
class Tiny1M3MNormSquashConfig(Tiny1M3MConfig):
    norm_type: str = "squash"

@dataclass
class Tiny1M3MNormLayerNormConfig(Tiny1M3MConfig):
    norm_type: str = "layernorm"
