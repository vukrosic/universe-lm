---
id: 057-normformer
status: needs-plan
round: 1
updated: 2026-06-11T01:21:40Z
transfer-risk: med
---

# 057 — Mid-FFN LayerNorm (post-W1)

## Source
Shleifer, Weston, and Ott, "NormFormer: Improved Transformer Pretraining with Extra Normalization" (arXiv:2110.09456). The paper ships three extra-normalization ops per block; r1 of this idea was the full bundle and was sent back for poor info value at single-A/B granularity. The piece being re-pitched here is the paper's L_m: an LN applied to the hidden activations *between* the FFN up-projection (W1) and the non-linearity (SiLU/gating) — i.e. a **mid-MLP** LN, distinct in placement from anything else in the queue.

## Mechanism
One ~10 LoC insertion in the FFN: after `W1 @ x` (the up-projection) and before the SiLU (or gating), apply `LayerNorm(W1 @ x)`. Identity-initialized: at step 0 with `LayerNorm` defaults (gain 1) the residual path is bit-equivalent to the baseline, so the A/B isolates the LN's effect on post-projection magnitude rather than changing initialization. Distinct from:
- 016-qk-norm (DONE) — pre-softmax attention stabilization, different site.
- 021-value-residual (DONE) — channel rescale on V output, not a hidden-state LN.
- 029-v-norm (RUNNING) — per-head LN on V, also attention-side.
- 051-scalenorm / 052-fixnorm / 055-deepnorm / 056-branchnorm — all norm the residual *stream* (pre- or post-block), not the mid-MLP hidden state.

## Scale evidence
NormFormer reports 0.27 PPL gain at 1.3B (LM pretraining, equal compute) with the full bundle; the paper does not ablate L_m in isolation, so the L_m-alone evidence is weaker. Indirect support: the same family of "stabilize activations between projections" levers (Pre-LN vs. Sandwich-LN; ReZero/DeepNorm residuals) all show their largest effect at 100M+ scale. transfer-risk: **med** — the bundled 1.3B evidence is the right scale, but L_m is being untangled from the other two ops without prior isolated evidence.

## Why it's worth a slot
The repo's existing norm portfolio (016/021/029/026/027/051/052/055/056) saturates the *attention path* and the *residual stream*. The MLP up-projection W1 is the only major site without an explicit magnitude-bounding lever, and it's the site where activation outliers are most likely to form (SiLU of `W1·x` with growing activation magnitudes across depth is the textbook failure mode this kind of mid-MLP LN targets). Bet: a mid-FFN LN moves loss because **post-W1 magnitudes drift faster than the entry pre-norm can correct**, and a null then *means* something ("post-projection magnitudes are fine, the bottleneck is elsewhere").
