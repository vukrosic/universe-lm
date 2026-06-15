---
id: 174-xpos-decay
status: implementing
round: 1
updated: 2026-06-15T01:48:15Z
transfer-risk: low
plain: Add a single learnable "decay" knob to RoPE so the model can gradually suppress attention to far-away positions, starting at zero decay so step-0 is identical to baseline RoPE.
---

# 174 — xPos Exponential Decay on RoPE (Single Learnable Global θ Decay on Rotary Frequencies)

## Source
- Sun, Dong, Patra, Ma, Huang, Majumder, Wei, "A Length-Extrapolatable Transformer" (xPos, arXiv:2212.10554, Dec 2022). xPos = RoPE + a position-dependent exponential decay applied to the rotary frequencies: `θ_{t,i} = (1 − γ)^t · (1/base^{2i/d_k})` where `γ ∈ [0, 1]` is the decay rate (one scalar across the model). With γ=0 the decay is identity (bit-equivalent to RoPE). Validated at GLM-130B (per the paper) and adopted in ChatGLM-2/3.
- In-repo context: closed axes line "RoPE base sweep — 500k winner". 172-per-head-rope-base just filed (per-head learnable base). 174 takes the *orthogonal* lever: a single global decay scalar γ applied on top of the 500k base. Distinct from 172 because 172 is per-head (4×12 = 48 scalars, per-head frequency scale) while 174 is global (1 scalar, position-dependent decay).
- 009-fire-pe WIN at tiny1m3m (Δ -0.064/-0.082, far exceeds plan bar). Locality-prior position encodings win at this tier — 174 is a different locality lever (exponential decay vs FIRE's continuous integrable PE) and provides a fresh axis.

## Mechanism
Standard RoPE (per head h, position t, dim i):
```
θ_{i} = 1 / base^{2i/d_k}
rotated pair: (cos(t·θ_i), sin(t·θ_i)) applied to Q/K
```
xPos applies an exponential decay `g_t = (1 − γ)^t` (or, in the original paper, a phase-decoupled form `g_t = (1 − γ)^t` for one half of the rotary pair and `g_t^(-1)` for the other; we use the simpler symmetric form for clarity):
```
rotated pair: (g_t · cos(t·θ_i), g_t · sin(t·θ_i))
```
With `γ = 0`, `g_t = 1` for all t ⇒ rotation is unmodified RoPE. With γ > 0, the rotation magnitude decays exponentially with distance — distant positions have *shrunk* Q/K projections toward zero, so dot products with current Q are smaller for distant K (via the magnitude shrinkage), biasing attention toward recent positions.

**Learnable γ**: one scalar `γ` (or a pair `(γ_real, γ_imag)` if using the original asymmetric form; we use the symmetric single-scalar for simplicity). Init `γ = 0` ⇒ `g_t = 1` for all t ⇒ forward is bit-identical to the 500k-base RoPE baseline.

The lever can then push γ positive (decay far-away attention) or even negative (counteract decay, extend context), though the symmetric-form constraint γ ≥ 0 is natural.

A variant lever axis: **per-layer γ_l** (12 scalars, init all 0) so each layer can have its own decay rate. This is strictly more expressive than global γ. We file 174 with per-layer γ_l as the canonical lever; per-head γ_h would be orthogonal and potentially over-parameterized.

## Design sketch
- **Files**:
  - `models/layers.py` — `MultiHeadAttention.__init__`: add
    `use_xpos: bool = False`. Add
    `self.xpos_gamma = nn.Parameter(torch.zeros(n_layers))` (per-layer
    scalar). Wire it into the rotary computation at the existing rope
    application site (≈ line 2031-2034 in `models/layers.py`).
  - `MultiHeadAttention.forward`: after computing the rotation
    magnitudes `cos/sin`, multiply by `g_t = (1 − xpos_gamma[l])^t`
    (clamp to ≥ 0 to ensure well-defined real-valued rotation).
    Use `torch.exp(xpos_gamma[l] * t)` for numerical stability.
  - `configs/llm_config.py` — add `use_xpos: bool = False`.
  - `models/llm.py` — pass the per-layer `xpos_gamma` index `l` from
    `TransformerBlock.__init__` (already has the layer index).
- **Config flag**: `use_xpos: bool = False` (off by default, baseline
  path bit-identical).
- **Step-0 byte-identical**: at init, `xpos_gamma[l] = 0` for all l
  ⇒ `g_t = exp(0·t) = 1` for all t ⇒ rotation magnitudes
  `cos(t·θ_i) · 1 = cos(t·θ_i)`, `sin(t·θ_i) · 1 = sin(t·θ_i)`. The
  Q/K rotation is unchanged ⇒ output is unchanged ⇒ **byte-identical
  to the 500k-base RoPE baseline at step 0**.
- **Intuition (why it might lower val loss)**: 009-FIRE PE already won
  by adding a continuous, integrable PE to baseline RoPE. xPos is a
  *different* locality prior: exponential decay on the rotary magnitude
  biases attention toward recent tokens without the elaborate FIRE
  machinery. At 0.94M/12L/4H/3M tokens, local context is most relevant
  for next-token prediction; xPos's decay is a learned "how local is
  local" knob that the optimizer can dial in. Per-layer γ_l lets
  early layers attend globally and late layers attend locally (or
  vice versa), which is consistent with the BERT-depth intuition.
- **LoC**: ~25 (parameter + apply-g_t). The decay is a single
  `torch.exp(-gamma * t)` broadcast over t; the math is one line.

## Scale evidence
- xPos validated at GLM-130B and adopted in ChatGLM-2/3 (per Sun et al.
  2022 and ChatGLM papers). **Direct validation at ≥100M**.
- In-repo at tiny1m3m: 009-FIRE (the locality-prior WIN) uses
  Continuous Integrable PE; 174 is a different locality prior
  (exponential decay). The fact that 009 won Δ=-0.064 suggests
  locality priors are well-suited to our tier.
- **Transfer risk: low** (validated at 130B; mechanism is scale-free;
  the lever is a single scalar per layer that any tier can absorb).

## Why it's worth a slot
The bet: xPos is a *proven* locality-prior PE (GLM-130B) and
locality-prior PEs win at our tier (009 FIRE). The orthogonal lever
axis (per-layer γ_l, init 0, bit-identical step-0) gives the model a
"how local" knob it can dial per layer, which is a richer lever than
FIRE's single formula. We expect Δval ∈ [-0.003, -0.020] (modest
because locality is already well-served by FIRE, but xPos is simpler
and may be cleaner at small scale). A null tells us xPos's decay is
redundant with FIRE/RoPE-base at this tier and the
"locality-prior PE family" can be marked exhausted. A win unlocks
the per-layer γ_l axis at Phase-2 where layer-depth specialization
gives the knob room to differentiate.
