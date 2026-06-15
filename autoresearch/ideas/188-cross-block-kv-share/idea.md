---
id: 188-cross-block-kv-share
status: needs-review
round: 1
updated: 2026-06-15T08:18:49Z
transfer-risk: med
plain: Let each attention block reuse a small fraction of the previous block's K and V projections (a learnable per-block scalar, starting at 0 so step-0 is byte-identical), like a slow re-read of upstream key/value memories.
---

# 188 — Cross-Block K/V Projection Sharing (Learnable Blend of Adjacent-Block Projections)

## Source
- Dehghani et al., "Universal Transformers" (ICLR 2019, arXiv:1807.03819) — shares parameters across depth; validated on algorithmic + small LM tasks (<100M).
- 021-value-residual (in-repo, WIN Δ=−0.034 at tiny1m3m) — carries V *across blocks via the residual stream*; the in-repo cross-block V-mixing family is residual-stream level, not projection level.
- 168-av-output-carry (closed null) — carries the attention output (AV) across blocks, on the residual stream (post-attention). Different placement.
- 164-q-carry (closed null, Δ=+0.036 wrong-sign at 0.94M) — carries Q across blocks; Q-side closed.
- 186-v-carry-block (needs-run, in-repo) — within-block V carry (recurrence along the time axis within a single block). Different axis (time vs depth).

## Mechanism
Standard attention: each block b computes its own `K_b = W_K_b @ x_b`, `V_b = W_V_b @ x_b`. Block b+1 has no awareness of block b's K, V projections.

Cross-block KV sharing: each block's K, V projection is a learnable convex blend of its own (new) projection and the previous block's projection:

```
W_K_b_eff = (1 − α_K_b) · W_K_b_self + α_K_b · W_K_{b-1}     # α_K_b init 0
W_V_b_eff = (1 − α_V_b) · W_V_b_self + α_V_b · W_V_{b-1}     # α_V_b init 0
```

At α=0, `W_K_b_eff = W_K_b_self` exactly (bit-identical to baseline). At α=1, the projection is fully shared with the previous block. Soft, learnable parameter sharing.

## Design sketch
- **File**: `models/layers.py` — modify `attention_block` to accept an optional "prev block W_K, W_V" hook. Stored on the previous block as `self.prev_W_K`, `self.prev_W_V` (set after each block's init).
- **Config flag**: `use_cross_block_kv_share: bool = False` (default).
- **Compute** (per block b): `W_K_eff = (1 − α_K) * self.W_K + α_K * self.prev_W_K.detach()`, `α_K ∈ [0,1]` via `sigmoid(α_K_raw)` init `α_K_raw = -10` (so sigmoid ≈ 0). Same for V. `detach()` so the gradient doesn't flow back through the previous block's projection.
- **Bit-identical at step 0**: `α_K_raw = -10` ⇒ `α_K ≈ 4.5e-5` ⇒ `W_K_eff ≈ self.W_K` (forward graph unchanged at step 0 up to fp32 noise).
- **Params**: 2 scalars per block × 12 blocks = 24 params (+0.003% of 0.94M), negligible.
- **Intuition**: forces adjacent blocks toward a shared KV *projection* subspace, regularizing depth. Different from residual-stream V-carrying (021) — that mixes V into the residual stream; 188 mixes W_V projections directly.

## Scale evidence
Universal Transformers validated at <100M scale; deeper models need longer horizons to amortize the parameter-sharing cost. Transfer-risk: med (the lever is a parameter-sharing regularizer; the win at 12L×3M tokens is plausible but unproven).

## Why it's worth a slot
The in-repo V-side cross-block WIN (021) hints V-side information propagation across depth is a binding constraint. 168 closed AV-carrying (post-attention) null; 164 closed Q-carrying null. 188 is the projection-level analog: if KV subspaces converge across depth, the model can re-use an upstream KV subspace for free. A null at tiny1m3m localizes the V-carrying benefit to residual-stream level only; a win would be a strong signal that projection-sharing across blocks is a missing lever.
