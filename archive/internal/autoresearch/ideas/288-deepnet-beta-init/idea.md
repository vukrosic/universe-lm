---
id: 288-deepnet-beta-init
status: done
round: 1
updated: 2026-06-16T13:10:52Z
transfer-risk: low
plain: The champion uses DeepNet's forward half only — α, the 0.204 residual-branch scale. Add its other half, β init-downscaling — multiply the value, output, and FFN projection weights by β = (8·n_layers)^(-1/4) ≈ 0.319 right after init. DeepNet's theorem couples α and β as a matched pair so the model's update stays bounded; shipping α alone is the incomplete mechanism. Zero new params, init-only change.
---

# 288 — DeepNet β init-downscaling on the deepnet+poly-alibi champion

## Why this, why now
Only **step-0 conditioning** levers bind at 0.94M/92 steps (see 285/286/287 and
closed.md — every capacity/FFN/attention-internal lever is NULL). This is the
purest possible step-0 conditioner: it changes nothing but the **initial weight
magnitude** of the residual-writing projections, and it does so by **completing
the champion's own winning mechanism**.

## The gap it closes
The champion's `use_deepnet_alpha` (Wang 2022, arXiv:2203.00555) implements only
the **forward** half — `x + α·(attn+ffn)` with α = (2L)^(-1/2) ≈ 0.204
(models/layers.py:8089). Canonical DeepNet pairs that forward scale with an
**init** down-scaling β of the value + output + FFN projections (Theorem 1: α and
β are derived *together* so the model update is bounded at initialization). We
ship α without β — the half-applied mechanism. β is absent from the entire repo
and from closed.md.

## Mechanism
`use_deepnet_beta_init` + `deepnet_beta`. AFTER the global `_init_weights`,
multiply by β: the **V-slice** (`qkv_size-kv_size : qkv_size`) and **O-slice**
(`qkv_size :`) of each block's fused `qkvo_proj`, plus both FFN projections
(`up_proj`, `down_proj`). Q and K are left at std=0.02 — β conditions what writes
*into* the residual stream, not the attention scores. `deepnet_beta <= 0` ⇒
canonical decoder gain β = (8·n_layers)^(-1/4) ≈ **0.319** at L=12. **0 new
params** (verified: 949,200, identical to champion). Smoke-checked: V/O + FFN
init scaled by exactly 0.319, Q/K untouched (ratio 1.000), forward finite.

## Distinct from the NULL init/depth levers
130-ReZero (per-block *learned* α init 0), 142-LayerScale (per-channel *learned*
γ), 197/`use_deepnet_alpha` (forward scalar) all touch the **forward** path or add
learned parameters. This is a **fixed, parameter-free, init-only** rescale of the
projection *weights themselves* — a different site (initialization, not forward)
and the documented companion to a lever that already WON here.

## NOT step-0 byte-identical — by design
β changes the init magnitude, so step-0 logits differ from the champion. That is
the point: it is an init-CONDITIONING lever (the family that binds at this tier),
not an identity-at-step-0 add. The flag-off path is bit-identical to the champion.

## Hypothesis
Right-sign Δ if the residual-writing projections are initialized too hot for the
α=0.204 forward scale (so early gradients are mis-scaled and β re-balances the
α·β update product); NULL or wrong-sign if α alone already conditions the
residual and β over-suppresses the 12 blocks over only 92 steps. Genuinely open —
289 brackets the strength with the α-matched β = (2L)^(-1/2) ≈ 0.204.

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then a
paired 3-seed confirm before any promotion). Single seed (42).
