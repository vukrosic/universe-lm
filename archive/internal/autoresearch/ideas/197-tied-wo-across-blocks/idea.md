---
id: 197-tied-wo-across-blocks
status: rejected
round: 1
updated: 2026-06-16T00:44:03Z
transfer-risk: med
plain: Force every attention block to use the same output projection matrix W_O (init at the baseline's W_O so step-0 is byte-identical), like tying the final step of attention across depth — a cheap regularizer on what each block is allowed to write back to the residual stream.
---

# 197 — Tied W_O Across Blocks (Soft Blend, Sigmoid-Bounded, α_b_raw Init −10)

## Source
- Dehghani et al., "Universal Transformers" (ICLR 2019, arXiv:1807.03819) — share *all* parameters across blocks; 197 shares only W_O and only via a learnable soft blend.
- Lan et al., "ALBERT" (arXiv:1909.11942, 2020) — shares attention and FFN parameters across blocks; validated at BERT-base/large/xlarge.
- In-repo priors: (a) `closed.md` "layer tying" axis (full Universal-Transformer tying closed null at this tier). (b) `171-dropconnect-wo` closed 2026-06-15, Δ=+0.0478 wrong-sign on a *different* W_O intervention.

## Mechanism
Standard attention: each block b has its own `W_O_b ∈ R^{d_model × d_model}` that projects the attention output back to the residual stream.

**197 commits to the soft blend only.** Every block keeps its own `W_O_b`; a single shared `W_O_shared` is added; an effective output projection is built per-block:

```
α_b_raw ∈ R, init -10.0
α_b = sigmoid(α_b_raw)            # ∈ [0, 1] — convex blend
W_O_eff_b = (1 − α_b) · W_O_b + α_b · W_O_shared
```

- **At init (sigmoid(−10) ≈ 4.54e-5):** `W_O_eff_b = W_O_b` numerically (max-abs-diff on the projection < 1e-4 in fp32, within the champion-noise band). Step-0 is byte-identical to baseline up to fp32 noise.
- **At α_b = 1:** block b degenerates to a copy of `W_O_shared`; with all 12 blocks at α_b = 1, this reduces to a *hard* tied W_O. That extreme is a *limit*, not a target — the sweet spot is somewhere in (0, 1), learnable per block, with no manual schedule. The sigmoid bound prevents the optimizer from extrapolating off the convex hull (the rejected unconstrained-real `α_b ∈ R` form would let α_b = 2 produce `−W_O_b + 2·W_O_shared`, a non-convex linear combination — *not* a blend).
- **Why sigmoid-bounded (not unconstrained real).** Matches the in-repo precedent at `188-cross-block-kv-share` and `206-cross-block-ffn-share`, both of which use `α_raw init = −10.0` with `sigmoid(α_raw)` for the same "init-near-0 AND bounded [0,1]" semantics. The implementer should follow the 188 pattern (sigmoid, raw init −10) — do NOT copy the original draft's `α_b ∈ R` form, which would let the optimizer walk off the convex hull and break the "convex blend" interpretation.
- **Params added:** 1 shared `W_O` (4096) + 12 `α_b_raw` scalars (12) = 4,108 params. **No per-block parameters are removed.** Treatment has +4,108 params vs control; param delta is *symmetric* — this is a parameter-shape lever, not a model-size lever.
- **Optimization detail:** `α_b_raw` is a free parameter of the per-block `attention_block` module (e.g. `nn.Parameter(torch.full((), -10.0))`); `α_b = torch.sigmoid(α_b_raw)` is computed inside `forward()`. Standard SGD/AdamW on `α_b_raw` is fine; no special optimizer. No LR warmup trick, no α-schedule — let the model learn whether each block wants to share.

**Why soft over hard (the rejected alternative):** the hard version removes 12 `d_model × d_model` matrices (-49,152 params, -5.2% of 0.94M). A treatment smaller than control produces the wrong null — "tied W_O is just smaller, smaller loses, conclude W_O tying is bad." The soft blend makes the A/B fair: same per-block slot, same per-block learning, same total compute, with α_b as a *probe* into how much sharing each block wants. The hard version is a secondary ablation reserved for the definition gate, not this pitch.

## Scale evidence
ALBERT validated at BERT-base/large/xxlarge (110M-235M); Universal Transformers validated at <100M. Both share *all* parameters; 197 shares *only W_O* and only via a soft blend. The closest published W_O-only-tying analog is the *Universal Transformer* limit (α_b → 1 for all b), but no published paper isolates W_O. Transfer-risk: med — the lever is novel; the closest analogs (full layer tying) closed null in-repo, so 197 is a *narrower* form of tying and a *weaker* regularizer. A 197 WIN would be more informative than a 197 NULL; see "Why it's worth a slot" for the inference.

## Why it's worth a slot
**Discriminator vs the two W_O-adjacent closed priors.** 197 sits next to two nulls that already closed the W_O neighborhood at 0.94M:

1. *171-dropconnect-wo* (closed 2026-06-15, Δ=+0.0478 wrong-sign): a *weight-level regularizer* on W_O. The intervention is training-time multiplicative noise on W_O weights; α is fixed and stochastic. 197 is *parameter-level sharing* of W_O; the intervention is inference-time structural collapse of W_O_b into W_O_shared, with α as a learnable structural parameter. Different mechanisms, different failure modes: 171's null says "W_O has no slack to absorb dropconnect noise at training time" (regularizer story); 197's question is "W_O has no slack to absorb block-collapse at inference time" (structural-collapse story). Even if both null, they kill different mechanistic stories and the discriminator between them is informative.

2. *closed.md "layer tying"* (full Universal-Transformer tying, null): a *stronger* regularizer than 197. Full tying ties QKVO + FFN; 197 ties only W_O. 197 is the *narrowest* possible tying lever, the one with the *weakest* implicit prior. The 197 bet: full tying's null came from one of three things — (i) QK-tie killed attention learning, (ii) FFN-tie killed depth-specific representation learning, (iii) W_O-tie killed the residual-stream write geometry. 197 isolates (iii). If 197 *wins*, the binding constraint of full tying was (i) or (ii), not (iii) — and that points the next tying experiment at FFN-tie, not W_O-tie. If 197 *nulls*, W_O-tie is not the binding constraint either, and the full-tying failure mode lives in QK or FFN. Either result is informative; a 197 NULL is not "the third W_O null in a row" because the closed priors test *different* W_O interventions with *different* failure modes.

**Why this is a parameter-shape lever, not a regularizer.** 171 was a regularizer (it made the loss landscape noisier, in expectation). 197 is a *structural* lever — it changes the dimension of the parameter space (one global W_O + 12 scalars, not 12 W_O's). The β-bounded-tying formulation is the standard way to make a structural change *testable* at step 0; the same pattern is used in adapter modules, soft prompt tuning, and LoRA. The mechanical claim is: attention blocks have an *output geometry* (a coordinate system in the residual stream), and tying that geometry across blocks is a structural prior. Whether that prior helps or hurts is the empirical question.

**Leverage read.** At 0.94M, +4,108 params is +0.4% overhead — sub-noise, so any signal is *signal* not *capacity*. The win case is "soft W_O tying wins by Δ<-0.01 with sub-1% param overhead"; the null case is "W_O tying is the third closed W_O lever at 0.94M, the W_O neighborhood is exhausted, future levers should target QK or FFN." A 197 NULL is informative enough to justify the slot because it bounds the *narrowest* tying failure mode from above, which is data the closed-tying-axis log can't supply on its own.

**Taste bar check.** Big-if-true: a W_O WIN is the cleanest discriminator in the tying family because it points the next tying experiment. Safe-but-tiny: the soft-blend formulation is small in scope (+12 scalars, +1 matrix) and the implementation is `<50 LoC` in `models/layers.py`. The bet is sharp (one mechanism, one prior, one discriminator), and the null is informative (bounds the failure mode from the most conservative side).

## Pass/fail bar at tiny1m3m (seed 42)

- **Cache reference** (from `autoresearch/baseline-cache.json`):
  - Champion val: **6.2403** (175-alibi; std 0.0088, noise_band 0.04) — best known at this tier.
  - Current baseline mean: **6.40 ± 0.04** (two-ctrl bracket per §2 of the protocol).
  - Box noise floor at 0.94M / 3M tokens: **±0.01** (per the §2 two-ctrl rule). Any band narrower than this will not resolve at this tier.
- **WIN**: `trt_val ≤ ctrl_val − 0.01` **AND** clears the two-ctrl rule (the two-ctrl bracket must be tighter than the Δ). Sub-1% param overhead means the win is signal, not capacity.
- **NULL**: `|trt_val − ctrl_val| < 0.01` — within the box noise floor. Mechanistic read: the W_O-tie story is not the binding constraint of the full-tying null; the binding constraint lives in QK-tie or FFN-tie. Log as a third tying-axis null with the attribution narrowed from {QK, FFN, W_O} to {QK, FFN}.
- **DRIFT**: `trt_val > ctrl_val + 0.01` — treatment is *worse* than control. The structural-collapse prior is *harmful*, not just inert. Stop the tying axis on W_O; redirect future tying experiments at FFN-tie (the next-narrowest lever).
- **No in-between outcomes.** The band is 0.01 (the noise floor); any result between WIN and NULL is logged as NULL with a note that the effect is sub-noise and the experiment is repeated only if the magnitude is within 2× the noise floor of a 1-seed-42 reference.
