---
id: 061-alibi-bias
status: tasting
round: 2
updated: 2026-06-10T23:49:04Z
transfer-risk: low
---

# 061 — ALiBi bias

## Source
Press, Smith, and Lewis, "Train Short, Test Long: Attention with Linear Biases
Enables Input Length Extrapolation" (arXiv:2108.12409). Aug 2021.

## Mechanism
**Stack on FIRE, not replace it.** Add a per-head linear distance penalty
*on top of* the existing FIRE-PE attention path:

    attn_logits = q·kᵀ / √d  +  FIRE(i, j)  −  m_h · (i − j)        (causal: i ≥ j)

The slope vector `m_h` is the fixed Press et al. geometric schedule
`m_h = 2^(−8·h/H)` for `h = 0..H−1`, so with `H = 6` heads at tiny1m3m the
slopes are `{2⁻⁰·⁰, 2⁻¹·³³, 2⁻²·⁶⁷, 2⁻⁴·⁰, 2⁻⁵·³³, 2⁻⁶·⁶⁷}` (≈
1.000, 0.397, 0.158, 0.063, 0.025, 0.010) — no schedule sweep, no learnable
slopes (one-seed rule: a single configuration). The bias is added before
the (scaled) softmax; non-learnable, no extra parameters.

**Identity / zero-init.** At step 0 we set `m_h = 0` for all heads, making the
attention logits bit-identical to the FIRE-only baseline. Slopes ramp
linearly from 0 → Press-schedule over the first 10% of training steps
(then held constant). A step-0-equal-baseline lets a null read as "the
lever didn't fire" rather than "the init was bad", per the taste bar.

Implementable as a small precomputed `(H, T, T)` lower-triangular bias buffer
times a scalar warmup factor — drop-in addition inside `MultiHeadAttention`
in `models/layers.py`, < 50 LoC.

## Scale evidence
Press et al. (2021) train a 1.3B LM at length 1024 and match perplexity of a
sinusoidal model trained at 2048, with 11% faster training and 11% less
memory. ALiBi is also the recency-bias path used in BLOOM-176B and MPT-7B/30B
(production-scale). `transfer-risk: low` because the linear-distance penalty
is a pure attention-logit mechanism with 1B+ deployment evidence, and the
stack-on-FIRE framing is mechanistically agnostic to model size — the
question "does an explicit recency prior add signal on top of a content-aware
PE kernel?" is the same lever at 0.94M as at 135M.

## Why it's worth a slot
**The bet (one sentence): we expect FIRE+ALiBi to beat FIRE-alone by ≥ 0.01 nats
at tiny1m3m because FIRE's content-aware MLP learns *what* distances matter
while ALiBi imposes a fixed *monotone* recency prior — and at 3M tokens the
prior should carry signal the MLP hasn't yet absorbed.** This is the only PE
idea in the active queue (061/063-YaRN/064-XPOS/065-bilevel-pe/072-T5-RPE/
073-DeBERTa) that tests a **fixed, non-learnable, zero-content-coupling**
distance schedule — every other PE candidate is either learnable, content-
aware, or rotation-based. So a null here cleanly closes the "is content-free
recency redundant with FIRE?" question — informative whether it wins or loses.

We measure against the **FIRE-equipped control at 6.3234** (009-fire-pe WIN,
closed.md), not the V+q+SWA+HighRoPE reference at 6.4287 (which is no longer
the live baseline). A drift inside the ~±0.01 ctrl-gap at this tier (per the
020–025 cluster) is logged as inconclusive, not passing. The cautionary
prior is 013-CoPE (stacked-on-FIRE, +0.069 destructive) — but ALiBi differs
sharply: CoPE adds a *content-conditioned* extra position signal that
competes with FIRE's MLP, while ALiBi adds a *content-free* monotone term
the MLP cannot reproduce (FIRE's bias is symmetric in content; ALiBi's is
not). The mechanistic case for "ALiBi adds, CoPE collided" is what makes
this not just "another PE try".
