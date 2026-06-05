"""Muon ablations — see docs/research/muon/plan.md.

30 diverse ablations. Knob families (kept small on purpose — diversity
comes from cross-family combos, not from sweeping one knob):
  - Orthogonalization:    muon_ns_steps, muon_orthogonalize, muon_coeffs_mode
  - Shape scaling:        muon_shape_scale, muon_scale_mode
  - Momentum / nesterov:  muon_momentum, muon_nesterov
  - LR coupling:          muon_lr, adamw_lr
  - Compute savings:      muon_lazy_ortho_steps

Structure: 10 anchors (one per sub-knob) + 20 cross-family combos so each
config is testing a different interaction, not the same knob at finer
resolution.
"""
from dataclasses import dataclass

from configs.llm_config import Tiny1M3MConfig


# ---- Anchors — one per sub-knob (10) -----------------------------------------

@dataclass
class Tiny1M3MMuonNS3Config(Tiny1M3MConfig):
    """A1 anchor — ns_steps=3. Light orthogonalization."""
    muon_ns_steps: int = 3


@dataclass
class Tiny1M3MMuonNS5Config(Tiny1M3MConfig):
    """A2 anchor — ns_steps=5 (production default)."""
    muon_ns_steps: int = 5


@dataclass
class Tiny1M3MMuonNoOrthoConfig(Tiny1M3MConfig):
    """A3 anchor — NoOrtho (skip polar-express = momentum SGD on 2D)."""
    muon_orthogonalize: bool = False


@dataclass
class Tiny1M3MMuonNSCoeffsConfig(Tiny1M3MConfig):
    """A4 anchor — NewtonSchulz quintic coeffs (vs polar-express)."""
    muon_coeffs_mode: str = "newton_schulz"


@dataclass
class Tiny1M3MMuonNoShapeScaleConfig(Tiny1M3MConfig):
    """A5 anchor — NoShapeScale (drop max(1, fanout/fanin)^0.5)."""
    muon_shape_scale: bool = False


@dataclass
class Tiny1M3MMuonSpectralScaleConfig(Tiny1M3MConfig):
    """A6 anchor — SpectralScale (0.2·sqrt(max(dims)))."""
    muon_scale_mode: str = "spectral"


@dataclass
class Tiny1M3MMuonRMSMatchScaleConfig(Tiny1M3MConfig):
    """A7 anchor — RMSMatchScale (update RMS == AdamW update RMS)."""
    muon_scale_mode: str = "rms_match"


@dataclass
class Tiny1M3MMuonMomentum099Config(Tiny1M3MConfig):
    """A8 anchor — Momentum=0.99 (heavy momentum)."""
    muon_momentum: float = 0.99


@dataclass
class Tiny1M3MMuonNoNesterovConfig(Tiny1M3MConfig):
    """A9 anchor — NesterovOff (plain heavy-ball momentum)."""
    muon_nesterov: bool = False


@dataclass
class Tiny1M3MMuonLRRatio4to1Config(Tiny1M3MConfig):
    """A10 anchor — LRRatio 4:1 (production default muon_lr=0.024, adamw_lr=0.006)."""
    muon_lr: float = 0.024


# ---- Ortho × scale (3) ------------------------------------------------------

@dataclass
class Tiny1M3MMuonSpectralNS5Config(Tiny1M3MConfig):
    """C1 — SpectralScale + ns_steps=5. Different scale target with default ortho budget."""
    muon_scale_mode: str = "spectral"
    muon_ns_steps: int = 5


@dataclass
class Tiny1M3MMuonSpectralNS3Config(Tiny1M3MConfig):
    """C2 — SpectralScale + ns_steps=3. Cheaper ortho + spectral scale."""
    muon_scale_mode: str = "spectral"
    muon_ns_steps: int = 3


@dataclass
class Tiny1M3MMuonRMSMatchNS5Config(Tiny1M3MConfig):
    """C3 — RMSMatchScale + ns_steps=5. Fairer AdamW coupling with default ortho budget."""
    muon_scale_mode: str = "rms_match"
    muon_ns_steps: int = 5


# ---- Ortho × momentum / nesterov (4) -----------------------------------------

@dataclass
class Tiny1M3MMuonNS3NoNesterovConfig(Tiny1M3MConfig):
    """C4 — ns_steps=3 + NesterovOff. Light ortho + plain heavy-ball."""
    muon_ns_steps: int = 3
    muon_nesterov: bool = False


@dataclass
class Tiny1M3MMuonNS5NoNesterovConfig(Tiny1M3MConfig):
    """C5 — ns_steps=5 + NesterovOff. Default ortho + plain heavy-ball."""
    muon_ns_steps: int = 5
    muon_nesterov: bool = False


@dataclass
class Tiny1M3MMuonNS3Momentum099Config(Tiny1M3MConfig):
    """C6 — ns_steps=3 + momentum=0.99. Light ortho + heavy momentum."""
    muon_ns_steps: int = 3
    muon_momentum: float = 0.99


@dataclass
class Tiny1M3MMuonNoOrthoNesterovOffConfig(Tiny1M3MConfig):
    """C7 — NoOrtho + NesterovOff. Pure heavy-ball SGD. The non-Muon baseline of the family."""
    muon_orthogonalize: bool = False
    muon_nesterov: bool = False


# ---- Scale × momentum / nesterov (4) ----------------------------------------

@dataclass
class Tiny1M3MMuonSpectralNoNesterovConfig(Tiny1M3MConfig):
    """C8 — SpectralScale + NesterovOff. Does spectral absorb the nesterov lookahead?"""
    muon_scale_mode: str = "spectral"
    muon_nesterov: bool = False


@dataclass
class Tiny1M3MMuonSpectralMomentum099Config(Tiny1M3MConfig):
    """C9 — SpectralScale + momentum=0.99. Heavy momentum + spectral scale."""
    muon_scale_mode: str = "spectral"
    muon_momentum: float = 0.99


@dataclass
class Tiny1M3MMuonRMSMatchNoNesterovConfig(Tiny1M3MConfig):
    """C10 — RMSMatchScale + NesterovOff."""
    muon_scale_mode: str = "rms_match"
    muon_nesterov: bool = False


@dataclass
class Tiny1M3MMuonNoShapeNoNesterovConfig(Tiny1M3MConfig):
    """C11 — NoShapeScale + NesterovOff. Flat LR + plain heavy-ball."""
    muon_shape_scale: bool = False
    muon_nesterov: bool = False


# ---- LR × everything (3) -----------------------------------------------------

@dataclass
class Tiny1M3MMuonSpectralLR2to1Config(Tiny1M3MConfig):
    """C12 — SpectralScale + LRRatio 2:1. Halve the muon step, see if spectral target needs it."""
    muon_scale_mode: str = "spectral"
    muon_lr: float = 0.012


@dataclass
class Tiny1M3MMuonNoShapeLR2to1Config(Tiny1M3MConfig):
    """C13 — NoShapeScale + LRRatio 2:1. No shape, halved step."""
    muon_shape_scale: bool = False
    muon_lr: float = 0.012


@dataclass
class Tiny1M3MMuonSpectralLR8to1Config(Tiny1M3MConfig):
    """C14 — SpectralScale + LRRatio 8:1. Aggressive step with spectral target."""
    muon_scale_mode: str = "spectral"
    muon_lr: float = 0.048


# ---- Lazy ortho × everything (3) ---------------------------------------------

@dataclass
class Tiny1M3MMuonLazy4Config(Tiny1M3MConfig):
    """C15 — LazyOrtho every 4 steps. The headline speed lever."""
    muon_lazy_ortho_steps: int = 4


@dataclass
class Tiny1M3MMuonLazy4SpectralConfig(Tiny1M3MConfig):
    """C16 — Lazy4 + SpectralScale. Speed lever + different scale target."""
    muon_lazy_ortho_steps: int = 4
    muon_scale_mode: str = "spectral"


@dataclass
class Tiny1M3MMuonLazy4NoNesterovConfig(Tiny1M3MConfig):
    """C17 — Lazy4 + NesterovOff. Speed lever + plain heavy-ball."""
    muon_lazy_ortho_steps: int = 4
    muon_nesterov: bool = False


# ---- Three-way combos (3) ----------------------------------------------------

@dataclass
class Tiny1M3MMuonSpectralNS3Momentum099Config(Tiny1M3MConfig):
    """C18 — SpectralScale + ns_steps=3 + momentum=0.99. Three levers stacked."""
    muon_scale_mode: str = "spectral"
    muon_ns_steps: int = 3
    muon_momentum: float = 0.99


@dataclass
class Tiny1M3MMuonNoOrthoNoNesterovLazy4Config(Tiny1M3MConfig):
    """C19 — NoOrtho + NesterovOff + Lazy4. Maximum compute savings on pure SGD."""
    muon_orthogonalize: bool = False
    muon_nesterov: bool = False
    muon_lazy_ortho_steps: int = 4


@dataclass
class Tiny1M3MMuonSpectralLazy4LR2to1Config(Tiny1M3MConfig):
    """C20 — SpectralScale + Lazy4 + LRRatio 2:1. Cheap + aggressive scale target + halved step."""
    muon_scale_mode: str = "spectral"
    muon_lazy_ortho_steps: int = 4
    muon_lr: float = 0.012


# ---- Coeffs × everything (3) -------------------------------------------------

@dataclass
class Tiny1M3MMuonNSCoeffsSpectralConfig(Tiny1M3MConfig):
    """C21 — NSCoeffs + SpectralScale. Quintic + spectral. Tests two non-default pieces together."""
    muon_coeffs_mode: str = "newton_schulz"
    muon_scale_mode: str = "spectral"


@dataclass
class Tiny1M3MMuonNSCoeffsLazy4Config(Tiny1M3MConfig):
    """C22 — NSCoeffs + Lazy4. Cheaper ortho coeffs + cheaper ortho cadence."""
    muon_coeffs_mode: str = "newton_schulz"
    muon_lazy_ortho_steps: int = 4


@dataclass
class Tiny1M3MMuonNSCoeffsMomentum099Config(Tiny1M3MConfig):
    """C23 — NSCoeffs + momentum=0.99. Different ortho + heavy momentum."""
    muon_coeffs_mode: str = "newton_schulz"
    muon_momentum: float = 0.99


# ---- Momentum × LR (2) -------------------------------------------------------

@dataclass
class Tiny1M3MMuonMomentum099LR8to1Config(Tiny1M3MConfig):
    """C24 — momentum=0.99 + LRRatio 8:1. Heavy momentum + aggressive step."""
    muon_momentum: float = 0.99
    muon_lr: float = 0.048


@dataclass
class Tiny1M3MMuonMomentum09LR2to1Config(Tiny1M3MConfig):
    """C25 — momentum=0.9 + LRRatio 2:1. Light momentum + conservative step."""
    muon_momentum: float = 0.9
    muon_lr: float = 0.012


# ---- Saturation / stress (2) -------------------------------------------------

@dataclass
class Tiny1M3MMuonSpectralNSCoeffsLazy4Config(Tiny1M3MConfig):
    """C26 — SpectralScale + NSCoeffs + Lazy4. Three orthogonal speed levers stacked."""
    muon_scale_mode: str = "spectral"
    muon_coeffs_mode: str = "newton_schulz"
    muon_lazy_ortho_steps: int = 4


@dataclass
class Tiny1M3MMuonNoShapeNoNesterovLR2to1Config(Tiny1M3MConfig):
    """C27 — NoShapeScale + NesterovOff + LRRatio 2:1. Pure SGD-M with halved step, no scale."""
    muon_shape_scale: bool = False
    muon_nesterov: bool = False
    muon_lr: float = 0.012


# ---- Cross-family extreme probes (3) -----------------------------------------

@dataclass
class Tiny1M3MMuonNoOrthoLazy4Config(Tiny1M3MConfig):
    """C28 — NoOrtho + Lazy4. Cheapest possible Muon schedule. Tests how much ortho matters."""
    muon_orthogonalize: bool = False
    muon_lazy_ortho_steps: int = 4


@dataclass
class Tiny1M3MMuonNS5Momentum09Config(Tiny1M3MConfig):
    """C29 — ns_steps=5 + momentum=0.9. Default ortho + lighter momentum than default."""
    muon_ns_steps: int = 5
    muon_momentum: float = 0.9


@dataclass
class Tiny1M3MMuonSpectralNS5NoNesterovConfig(Tiny1M3MConfig):
    """C30 — SpectralScale + ns_steps=5 + NesterovOff. Three-way default-ortho combo."""
    muon_scale_mode: str = "spectral"
    muon_ns_steps: int = 5
    muon_nesterov: bool = False
