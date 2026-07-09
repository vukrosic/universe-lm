---
id: 044-attnres
status: needs-plan
round: 1
updated: 2026-06-11T01:34:07Z
transfer-risk: low
---

# 044 — Attention Residuals

## Source
Kimi Team, "Attention Residuals" (arXiv:2603.15031), 2026.
https://arxiv.org/abs/2603.15031 (stable URL alongside the arXiv ID per r1 flag).

## Mechanism
Replace fixed residual accumulation with a learned softmax over the
sequence of earlier block outputs. The standard `x ← x + h` becomes
`x ← Σ_i α_i · h_i` with `α = softmax(W · [h_0, h_1, …, h_L] + bias)`,
where `h_i` is the residual update produced by block `i`. The query
comes from the current pre-residual state; keys/values are the historical
block updates. **Pinned shape for tiny1m3m**: a single learned linear
`W ∈ R^{d × 1}` per block (scalar logit per past block, so the routing
softmax has ≤5 keys at L=6) plus a fixed large positive bias on the
identity index at init (see "Init" below). Cost: O(L·d) per block, with
L=6 → ~30 trainable floats total across the model.

### Init (identity-preserving, kills the wiring-artifact question)
- At step 0, every routing logit is set to 0 except the identity index
  (the most recent block's update), which receives a large positive bias
  `+b_id` with `b_id` chosen so `softmax` puts ~1.0 mass on identity
  (e.g. `b_id = 10` with temperature 1 ⇒ α_identity ≈ 0.9995, the rest
  split across 5 keys ⇒ each ≤ ~0.0001).
- With that init, the per-block residual at step 0 is within ~1e-3 of the
  plain-sum path. A stricter init (b_id=20) drives it to <1e-9. The
  choice between b_id=10 and b_id=20 is the only init knob; the rest is
  the standard scaled-W shape from the paper.
- **Step-0 ≡ baseline** (within the bias's `1 − softmax(0)/softmax(b_id)`
  residual) is asserted in the unit test (e) listed below, mirroring
  023-canon's gate-of-zeros identity test.

### Where the lever fires at L=6
The r1 finding is right that the original paper demonstrates the lever
at L≈64 and budget for 1.4T tokens — that's a *transfer-up* argument,
not a *fires-at-L=6* argument. The mechanism-side argument for why
softmax-over-depth still has room to express at L=6:

1. **The lever is not "more depth helps"; it is "uniform residual sum is
   not the only interpolation across the L previous updates."** Even
   with only L=5 historical keys per block, the per-block interpolation
   has 4 free parameters (5 logits up to softmax symmetry) that can move
   from the identity point. At L=6 this is enough for the model to (a)
   skip the contribution of an early block whose update is noisy or
   stale, or (b) re-weight an early block whose features are still
   useful at the current depth. Neither move is available in the
   uniform-sum baseline.
2. **The lever is independent of depth scale in the same sense that a
   1D convolution is independent of sequence length**: it expresses a
   unit of capacity per block (a routing decision), and that unit is
   present whether L=6 or L=64. What changes with L is the *number of
   keys per routing decision*, not whether the decision is meaningful.
3. **6 layers is shallow enough that each block's update carries a
   non-trivial fraction of the loss gradient**: at L=64 a single block's
   update is one of 64 summands (low leverage per routing choice); at
   L=6 a single block's update is one of 6 summands (high leverage per
   routing choice). The lever is *stronger* per-decision at L=6, even
   if the *total* capacity is smaller.

The null hypothesis is therefore specific: at L=6 with FIRE+RoPE and the
023-canon mixer already on the residual stream, the per-block
interpolation has nothing useful to do — the uniform sum is already the
optimum.

### Differentiation vs 023-canon-conv (r1 crowded-family check)
023-canon-conv inserts a *local, token-axis* depthwise Conv1d **inside
the block, on the residual stream, before the attention sublayer**. It
operates on the **time axis** within a single block's residual stream
(kernel=3, mixes the current token with its 2 left neighbors).

AttnRes is a different lever on the **depth axis across blocks**:
- Different axis: Canon mixes **tokens** (sequence dimension T) within
  a block; AttnRes mixes **blocks** (depth dimension L) across blocks.
- Different time scale: Canon runs at the per-token granularity of one
  block; AttnRes runs once per block, choosing how to interpolate the L
  previous block updates into the next residual state.
- Different placement: Canon is a sublayer in the residual stream
  (additive update inside one block); AttnRes is a *replacement* for
  the residual accumulation across blocks (interpolative update
  spanning the depth axis).
- Stacking them is **additive in capacity, not redundant**: Canon frees
  the attention head to model long-range, AttnRes frees the residual
  stream to choose how to interpolate depth. The two are orthogonal
  axes (token-axis vs depth-axis, inside-block vs across-blocks).
  023-canon measures local-mixing; 044-attnres measures depth-routing.

## Scale evidence
The paper reports scaling-law gains and integrates AttnRes into a
48B-total / 3B-active Kimi Linear run trained on 1.4T tokens. The
"transfer-up" reading: gains were demonstrated at very large scale. The
"fires-at-small-scale" reading: the paper's ablations include ~125M
configs where the same lever reduces val loss, so the effect is not
gated on 48B-scale training. The mechanism is a per-block routing
softmax — its capacity scales as O(L), not as O(L²) or with model width
in any non-trivial way, so it remains a degree of freedom at any depth.

`transfer-risk: low` (revised justification): the lever is
demonstrated at 100M+ scale, the mechanism is a per-block routing
softmax (a primitive the pipeline already has machinery for), and the
implementation is a sub-50-LoC drop-in alongside `use_canon_conv`.

## Why it's worth a slot
We expect **Δ ∈ [−0.005, −0.02] at tiny1m3m** if depth-routing is
non-redundant with 023-canon, and **Δ ∈ [−0.005, +0.005]** (null) if
L=6 is too shallow for the lever to express — both of which are
informative. The 023-canon WIN was ~−0.06 (Δ vs FIRE-equipped ctrl);
AttnRes targets a different axis, so a 1/3-magnitude win (~−0.02) is
a credible expected value for the depth-routing axis at this scale.
A null is just as informative: it would mean the per-block residual
sum is already near-optimal at L=6, which constrains future depth-axis
lever design. A win is a sub-50-LoC, transferable structural lever
that composes with Canon's token-axis mixer rather than replacing it.
**Sharp bet**: "we expect trt_val ≤ ctrl_val − 0.005 with FIRE
equipped, and the failure mode is null (|Δ| ≤ 0.005), not regression
(Δ ≥ +0.01), because the identity-init ensures the lever can only help
or stay neutral, never hurt."
