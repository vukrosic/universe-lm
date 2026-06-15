---
id: 196-block-residual-ema
status: needs-taste
round: 1
updated: 2026-06-15T09:00:00Z
transfer-risk: low
plain: Blend each block's residual stream with the previous block's residual stream using a learned per-block EMA coefficient (init so the contribution is 0 at step 0, byte-identical baseline).
---

# 196 — Blockwise Residual EMA (Cross-Block Residual Stream Smoothing)

## Source
- 110-weight-ema (closed null Δ=+1.0831 wrong-sign, tier-mismatch) — EMA on model *weights*, not residual stream. Different axis (weight, not activation).
- 132-born-again (closed null Δ=+0.0209 wrong-sign) — EMA teacher model for distillation; weight-level not activation-level.
- 134-mega-ema (closed null Δ=+0.0390 wrong-sign) — EMA with β=0.9999, weight-level. Tier-mismatch pattern.
- 021-value-residual (in-repo WIN Δ=−0.034) — V-side cross-block carry on the residual stream. Different tensor (V vs full residual).
- 017-sub-ln-sandwich (closed null) — sub-LN normalization sandwich. Different mechanism.
- Tarvainen & Valpola, "Mean Teacher" (NeurIPS 2017) — weight-level EMA, not residual stream.
- Polyak & Juditsky, "Acceleration of Stochastic Approximation" (1992) — running average of iterates; the residual-stream EMA analog at the activation level is novel.

## Mechanism
Standard residual: `x_{b+1} = x_b + attn_block(x_b) + ffn_block(x_b)`. The residual stream is a strict cumulative sum; each block's contribution is independent of previous blocks' contributions.

Blockwise residual EMA: each block's residual contribution is blended with the previous block's residual contribution via a per-block learnable EMA coefficient:
```
out_b = attn_block_b(x_b) + ffn_block_b(x_b)
x_{b+1} = x_b + (1 − β_b) · out_b + β_b · out_{b-1}    # β_b init 0
```
At init β_b = 0, the EMA contribution is 0 and the residual stream is exactly the standard cumulative sum. As β grows, each block's contribution is a weighted average of its own and the previous block's contribution — a depth-wise smoothing of the residual stream.

This is **not** weight EMA (which smooths the optimizer trajectory in weight space); this is residual-stream EMA, which smooths the residual stream in activation space.

## Design sketch
- **File**: `models/layers.py` — modify `attention_block` and `ffn_block` to optionally blend the previous block's contribution.
- **Config flag**: `use_block_residual_ema: bool = False`, `block_residual_ema_init: float = -10.0` (β via `sigmoid(β_raw)` init at `β_raw = -10` so sigmoid ≈ 0).
- **Compute**: maintain `prev_block_out` (set by the previous block). At block b, `out_b = (1 − sigmoid(β_raw_b)) · block_out_b + sigmoid(β_raw_b) · prev_block_out.detach()`. The `.detach()` prevents gradients from flowing through the previous block's contribution.
- **Bit-identical at step 0**: β_raw = -10 ⇒ sigmoid ≈ 4.5e-5 ⇒ `out_b ≈ block_out_b` (forward graph unchanged up to fp32 noise).
- **Params**: 1 scalar per block × 12 blocks = 12 β scalars (+0.001% of 0.94M).
- **Intuition**: weight EMA smooths the *optimizer trajectory* (closed at 0.94M for tier-mismatch reasons — EMA effective window ~1000 steps >> 92 step run). Residual stream EMA smooths the *forward-pass trajectory* (residual stream at depth b is a weighted blend of blocks b and b-1's contributions). Different mechanism, different scale (residual stream EMA is fully active at step 0, not gated by training-step EMA decay).

## Scale evidence
Mean Teacher validated at classification (CIFAR, ImageNet); weight-EMA tier-mismatch at 0.94M (closed at 92-step horizon). No published residual-stream EMA win for LMs that I'm aware of. Transfer-risk: low (the lever is a minor architectural change with few params).

## Why it's worth a slot
**Pattern attribution**: weight-EMA closed null at 0.94M (tier-mismatch). 196 is a *different* EMA — at the residual stream level, not weight level. The residual stream EMA is fully active at step 0 (not gated by training-step EMA decay), so it doesn't suffer the 92-step horizon problem that weight-EMA does. A 196 WIN would mean residual-stream smoothing is a binding lever that weight-EMA isn't; a 196 NULL would mean the EMA family is generally hostile at 0.94M regardless of placement.
