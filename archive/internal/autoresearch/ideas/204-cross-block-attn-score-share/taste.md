# Taste review — 204-cross-block-attn-score-share

## r1 — 2026-06-15 — verdict: accept

- **Sharp bet, last axis in a real family.** The cross-block mixing family at
  0.94M has three prior legs: **021-value-residual (WIN Δ=-0.034)** on the
  residual-stream V axis, **164-q-carry (NULL Δ=+0.036 wrong-sign)** on the
  residual-stream Q axis, **168-av-output-carry (NULL)** on the post-attention
  AV axis. 188 (KV-projection-share) is *implementing* on a different axis
  (projection W_K, W_V reuse, not score reuse). 204 is the **pre-softmax
  attention-logit axis** — a fifth and arguably the last untested leg, because
  scores (Q·K^T/√d_k) are functionally distinct from V (value), Q (query),
  AV (post-softmax weighted values), and W_K/W_V (projection matrices). The
  miner's bet is crisp: WIN = score propagation across depth is a missing
  lever; NULL = V-side carry is uniquely binding and the other four cross-block
  mixing axes are dead ends. That is a real mechanistic bet, not a vibe.

- **Clean implementation, niche-fit.** α=0 init (sigmoid(−10) ≈ 4.5e-5)
  guarantees bit-identical baseline at step 0. `detach()` on the previous
  block's scores prevents gradient leakage back through the upstream block —
  this matters because 150-xlayer-feedback collapsed on exactly the
  cross-block autograd path. +12 params (+0.001%) is the cheapest cost
  profile in the family. Mechanism (not HP), identity-init, tiny1m3m
  testable, no data/infra dependencies — passes every niche test.

- **High information value either way.** A clean null closes the cross-block
  mixing family at 0.94M (V is the uniquely binding axis; Q/AV/score/proj
  are all null), which is exactly the kind of attribution insight that
  bounds what we should try at 135M. A clean win would be a new cross-block
  lever that doesn't go through the residual stream, a meaningful
  architectural signal. The 92-step run is enough to detect either
  outcome — the prior V-residual WIN of Δ=-0.034 is well above the ±0.04
  noise band, so the effect size would be visible if present.

- **Portfolio fit is the only hesitation — and it argues for accept, not
  revise.** 17+ ideas sit in needs-taste, with several cross-block / depth-
  mixing variants. But 188 is *implementing* (not stale) and 204 tests a
  *different* tensor (scores vs projections). Revising the family to "merge
  into 188" would erase the axis separation that makes 188's null
  informative. The right move is to let 188 and 204 run sequentially: 188
  tells us about projection sharing, 204 tells us about score sharing.
  After both, the cross-block axis is fully closed.

- **No red flags from the writing.** The design sketch flags the
  detach-on-prev-block path (the 150-xlayer-feedback failure mode), the
  α=0 bit-identity is asserted correctly, and the prior 021/164/168/186/188
  attributions are honest (204 calls out that V-residual is *residual-stream
  V*, not V-projection, and AV-carry is *post-softmax output*, not score).
  The Memorizing-Transformers citation is honest about being an external
  validation (cross-document retrieval, not within-model cross-block
  reuse) and the transfer-risk: med tag is consistent with the lever type.

**Verdict: accept.** Round reset to 1 for the definition gate's budget.
