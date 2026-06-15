---
id: 193-blockwise-attn-temp-schedule
status: needs-repitch
round: 1
updated: 2026-06-15T12:13:10Z
transfer-risk: low
plain: Make the attention "sharpness" vary smoothly with depth using a fixed cosine schedule (no learned parameters), starting with a flat schedule so step-0 matches the baseline — a depth-aware attention prior without any trainable knobs.
---

# 193 — Blockwise Attention Temperature Schedule (Cosine-Depth Soft Schedule, No Learnable Params)

## Source
- 155-per-head-temp (closed null Δ=−0.0063 inside band) — per-head learnable temperature scalar; learned but per-head axis closed at 0.94M.
- 175-alibi-slopes (in-repo WIN Δ=−0.1585) — fixed per-head ALiBi slopes; depth-uniform bias that decayed with distance. Different shape (additive, not scale).
- Press et al., "ALiBi" (arXiv:2108.12409, 2022) — fixed-position bias, depth-uniform; 193 is a depth-varying scale (analogous to ALiBi but on the multiplicative side).
- Su et al., "RoPE" (arXiv:2104.09864) — fixed-position rotation; 193 is fixed-depth scale.
- Roy et al., "Efficient Content-Based Sparse Attention" (2019/2020) — blockwise-local attention patterns; 193 is a continuous blockwise scale, not a hard mask.
- Possible related lever: cascaded / curriculum attention — early layers attend softly (broad context), late layers attend sharply (specific context). Cited in some "annealed attention" ablations (no canonical paper found).

## Mechanism
Standard attention: `scores = QK^T / √d_k` (constant temperature 1/√d_k across all blocks).

Blockwise temperature schedule: each block b has a multiplicative temperature `τ_b` on its scores:
```
scores_b = Q_b K_b^T / (τ_b · √d_k)
attn_b = softmax(scores_b)
```
Schedule (proposed): `τ_b = 1.0 + α · cos(π · b / L)` where `b ∈ [0, L-1]`, `L = n_layers`. At α=0, all `τ_b = 1` (bit-identical baseline). At α>0, early blocks have `τ_0 = 1 + α` (cooler/sharper softmax) and late blocks have `τ_{L-1} = 1 − α` (warmer/softer softmax), or vice versa depending on sign.

The bet: at 0.94M/12L, a depth-varying attention temperature may help — early layers can be sharper (capture local patterns), late layers softer (integrate context). This is a **fixed** schedule (no learnable params) — the lever is the schedule *shape*, not a parameter.

## Design sketch
- **File**: `models/layers.py` — modify the manual attention path to apply `scores / (τ_b · √d_k)`.
- **Config flag**: `use_block_temp_schedule: bool = False`, `block_temp_alpha: float = 0.0` (default = flat schedule).
- **Bit-identical at step 0**: α=0 ⇒ `τ_b = 1` for all b ⇒ scores / (1 · √d_k) = standard path exactly.
- **No params**: the schedule is hard-coded; no per-block scalar learned.
- **Intuition**: a cosine depth schedule is a soft locality prior that varies with depth. Like ALiBi but on the scale side rather than the additive side. The schedule shape (α > 0 sharpens early, softens late; α < 0 inverts) is a single HP; for the lever we explore both signs.

## Scale evidence
ALiBi validated at 0.4B–6.7B (Press et al. 2022); cascaded-attention literature exists but is mostly empirical. Transfer-risk: low (fixed-function levers transfer well; the schedule shape is a single HP).

## Why it's worth a slot
**Attribution insight**: 175-ALiBi WIN (Δ=−0.1585, the largest in-repo WIN at tiny1m3m); 155-per-head-temp NULL (per-head learnable temp doesn't bind). The gap between 175's WIN and 155's NULL suggests the *depth-uniform* fixed schedule binds (ALiBi), but the *depth-local* learnable axis doesn't. 193 tests whether a *depth-varying* **fixed** schedule binds — combining 175's depth-uniform WIN with a depth-varying twist. If 193 nulls, the depth-uniform additive axis (ALiBi) is the binding lever and depth-variation is hostile; if 193 wins, a *scale-side* depth schedule is a missing lever.
