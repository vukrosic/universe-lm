---
id: 192-pre-rope-qk-rotation
status: planning
round: 1
updated: 2026-06-15T12:18:58Z
transfer-risk: med
plain: Apply a tiny learned rotation matrix to Q and K *before* the RoPE positional rotation kicks in (init at identity so step-0 is byte-identical), letting the model re-orient each head's geometry independently of position.
---

# 192 — Pre-RoPE Per-Head Q/K Rotation (Learned Static Orthogonal Rebase Pre-Position)

## Source
- 185-static-per-head-k-rotation (needs-run, in-repo, transfer-risk med) — applies a learned per-head rotation to K *after* position information is added (post-RoPE). Per-head gain axis on K only.
- 154-rebased-attn (in-repo WIN Δ=−3.4766 record break) — applies a *fixed* rebase matrix to K and V before softmax; the orthogonal-rebase axis is a strong lever at 0.94M.
- Su et al., "RoFormer / RoPE" (Neurocomputing 2024, arXiv:2104.09864) — RoPE's per-pair 2D rotations are *position-dependent*. 192's pre-RoPE rotation is *position-independent* — a static orthogonal rebase before RoPE's position mix.
- 172-per-head-rope-base (closed null Δ=+0.0109) — per-head *base frequency* in RoPE (position-dependent angle). Different mechanism.
- Wang et al., "ResFormer" / "Residual Rotation" (2024) — residual orthogonal transformation in attention.

## Mechanism
RoPE applies a per-pair 2D rotation to Q, K that depends on position `m`. Concretely, for pair i (covering 2i, 2i+1 channels):
```
θ_i(m) = m · base^(-2i/d)
[q_{2i}, q_{2i+1}] = rotate([q_{2i}, q_{2i+1}], θ_i(m))
```
192 applies a *static* (position-independent) per-pair rotation **before** RoPE:
```
φ_h_i: learnable per-head, per-pair angle (init 0)
[q_h_{2i}, q_h_{2i+1}] = rotate([q_h_{2i}, q_h_{2i+1}], φ_h_i)   # pre-RoPE
[k_h_{2i}, k_h_{2i+1}] = rotate([k_h_{2i}, k_h_{2i+1}], φ_h_i)   # pre-RoPE
# then RoPE with the standard position-dependent angles
```
At init φ_h_i = 0 (identity rotation), the static rotation is a no-op and the forward graph matches baseline. The 185 lever rotates K *after* position; 192 rotates both Q and K *before* position. Different placement, same learnable-angle axis.

## Design sketch
- **File**: `models/fire_pe.py` (or wherever RoPE is applied) — add a pre-RoPE rotation block.
- **Config flag**: `use_pre_rope_rotation: bool = False`, `pre_rope_rotation_init: float = 0.0` (init at 0).
- **Compute**: per head h, per pair i (d_k/2 = 8 pairs), one learnable scalar angle `φ_h_i ∈ ℝ`. Apply the 2D rotation `[x_{2i}, x_{2i+1}] = [cos(φ)x_{2i} - sin(φ)x_{2i+1}, sin(φ)x_{2i} + cos(φ)x_{2i+1}]`. Apply to Q and K before RoPE.
- **Bit-identical at step 0**: φ = 0 ⇒ `cos(0)=1, sin(0)=0` ⇒ `[x_{2i}, x_{2i+1}] = [x_{2i}, x_{2i+1}]` exactly.
- **Params**: 4 heads × 8 pairs × 12 blocks = 384 φ scalars (+0.041% of 0.94M), negligible.
- **Intuition**: RoPE's position-dependent rotation mixes position with feature geometry. A static pre-RoPE rotation lets each head pick its own *feature geometry* (which 2D sub-planes of d_k carry which meaning) before position is added — a per-head specialization axis that 154's fixed rebase and 185's post-RoPE rotation don't cover.

## Scale evidence
185-static-per-head-k-rotation in-repo (transfer-risk med, status implementing r2); RoPE-family wins at LLaMA-1/2/3 / Mistral / Qwen (≥7B); 154-rebased-attn WIN at 0.94M. Transfer-risk: med.

## Why it's worth a slot
154 WIN establishes the orthogonal-rebase axis at 0.94M; 185 tests post-RoPE K rotation (Q-side closed at this tier via 164). 192 tests *pre*-RoPE rotation on *both* Q and K. The pre-RoPE placement is fresh: 154 was fixed random rebase, 185 was K-only post-position, 192 is learned Q+K pre-position. If the orthogonal-rebase axis is binding, 192 should also bind; if the axis is specifically *post*-position (i.e., position-aware), 192 will null.
