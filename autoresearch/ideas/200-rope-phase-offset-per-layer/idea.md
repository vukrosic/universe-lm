---
id: 200-rope-phase-offset-per-layer
status: needs-repitch
round: 1
updated: 2026-06-15T08:28:51Z
transfer-risk: med
plain: Give each layer its own learnable phase shift on the RoPE rotation angles (init 0 so step-0 is byte-identical), letting depth re-orient the geometric meaning of each pair — a depth-axis extension of per-head RoPE.
---

# 200 — Per-Layer RoPE Phase Offset (Learnable Phase Shift on RoPE Angles)

## Source
- 175-alibi-slopes (in-repo WIN Δ=−0.1585) — fixed per-head additive bias on attention scores that decays with distance. Closest in-repo analog to position-bias modification.
- 172-per-head-rope-base (closed null Δ=+0.0109 wrong-sign inside band) — per-head *base frequency* in RoPE (modulates position-dependent angle). Different axis (frequency vs phase).
- 185-static-per-head-k-rotation (in-repo, needs-run) — per-head K rotation, post-RoPE. Different axis (post-RoPE vs RoPE-modulating).
- 009-fire-pe (in-repo WIN Δ=−0.064) — fixed RoPE variant with continuous position integration. 200 is a *learnable* twist on RoPE's angle structure.
- Su et al., "RoFormer / RoPE" (Neurocomputing 2024, arXiv:2104.09864) — RoPE's per-pair rotation angles `m · base^(-2i/d)`. 200 adds a learnable phase offset `m · base^(-2i/d) + φ_l` per layer.
- Chen et al., "Linearized Self-Attention with Multiplicative Position" (2021) — explored additive position bias on RoPE.

## Mechanism
Standard RoPE applies per-pair 2D rotation to Q, K where the rotation angle depends on position `m`:
```
θ_i(m) = m · base^(-2i/d_k)
[q_{2i}, q_{2i+1}] = rotate([q_{2i}, q_{2i+1}], θ_i(m))
```

Per-layer RoPE phase offset: add a learnable per-layer phase shift `φ_l` to the rotation angles:
```
θ_i(m, l) = m · base^(-2i/d_k) + φ_l           # φ_l learnable per layer, init 0
[q_{2i}, q_{2i+1}] = rotate([q_{2i}, q_{2i+1}], θ_i(m, l))
```

At init `φ_l = 0`, the angles match standard RoPE exactly (bit-identical). As `φ_l` grows, each layer's RoPE "phase" is shifted by `φ_l`, effectively rotating the position-encoding axes by a depth-dependent angle.

This is **not** the same as 172-per-head-rope-base (which modulates the *base frequency* per head, affecting the *position-encoding curve shape*). 200 modulates a *constant phase offset* per layer — a depth-wise rotational shift that doesn't change how position is encoded within a layer, only how the encoding is rotated between layers.

## Design sketch
- **File**: `models/fire_pe.py` (or wherever RoPE is applied) — add a per-layer phase offset parameter.
- **Config flag**: `use_per_layer_rope_phase: bool = False`, `per_layer_rope_phase_init: float = 0.0`.
- **Compute**: per layer l, learn a scalar `φ_l ∈ ℝ` (or per-pair `φ_l_i` for finer granularity). Apply: `θ_eff = θ_standard + φ_l` before the rotation.
- **Bit-identical at step 0**: φ_l = 0 ⇒ θ_eff = θ_standard ⇒ rotation matches baseline.
- **Params**: 1 φ scalar per block × 12 blocks = 12 scalars (+0.001% of 0.94M). Or per-pair per-block = 4×12 = 48 (still negligible).
- **Intuition**: each layer's RoPE operates in the same position-encoding curve but rotated by a depth-dependent phase. The bet: at 0.94M, position information propagates through layers and a depth-varying phase lets each layer interpret position in a slightly different coordinate system. Different from 172 (which modulates the *frequency*); 200 modulates a constant phase.

## Scale evidence
RoPE validated at LLaMA-1/2/3 / Mistral / Qwen (≥7B); ALiBi (additive analog) in-repo WIN at tiny1m3m. No published "per-layer RoPE phase offset" paper that I'm aware of. Transfer-risk: med (lever is a minor architectural change; per-head variant closed null at 0.94M).

## Why it's worth a slot
**Pattern**: 172-per-head-rope-base closed null at 0.94M (per-head frequency doesn't bind); 175-alibi-slopes WIN (depth-uniform additive bias). 200 tests a *per-layer phase offset* — neither frequency nor additive bias, but a rotational phase shift. The bet: at 0.94M, the depth dimension has unused capacity to specialize position interpretation; a per-layer phase gives the optimizer a clean axis to do so. A 200 WIN would mean depth-wise phase specialization is a missing lever; a 200 NULL would confirm the RoPE family is fully exploited at 0.94M (frequency × phase × per-head × per-layer are all closed or null).
