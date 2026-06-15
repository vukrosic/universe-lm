---
id: 197-tied-wo-across-blocks
status: needs-review
round: 1
updated: 2026-06-15T08:29:46Z
transfer-risk: med
plain: Force every attention block to use the same output projection matrix W_O (init at the baseline's W_O so step-0 is byte-identical), like tying the final step of attention across depth — a cheap regularizer on what each block is allowed to write back to the residual stream.
---

# 197 — Tied W_O Across Blocks (Soft Blend, α_b Init 0)

## Source
- Dehghani et al., "Universal Transformers" (ICLR 2019, arXiv:1807.03819) — share *all* parameters across blocks; 197 shares only W_O and only via a learnable soft blend.
- Lan et al., "ALBERT" (arXiv:1909.11942, 2020) — shares attention and FFN parameters across blocks; validated at BERT-base/large/xlarge.
- In-repo priors: (a) `closed.md` "layer tying" axis (full Universal-Transformer tying closed null at this tier). (b) `171-dropconnect-wo` closed 2026-06-15, Δ=+0.0478 wrong-sign on a *different* W_O intervention.

## Mechanism
Standard attention: each block b has its own `W_O_b ∈ R^{d_model × d_model}` that projects the attention output back to the residual stream.

**197 commits to the soft blend only.** Every block keeps its own `W_O_b`; a single shared `W_O_shared` is added; an effective output projection is built per-block:

```
α_b ∈ R, init 0.0
W_O_eff_b = (1 − α_b) · W_O_b + α_b · W_O_shared
```

- **At init (α_b = 0):** `W_O_eff_b = W_O_b` exactly. Step-0 is byte-identical to baseline.
- **At α_b = 1:** block b degenerates to a copy of `W_O_shared`; with all 12 blocks at α_b = 1, this reduces to a *hard* tied W_O. That extreme is a *limit*, not a target — the sweet spot is somewhere in (0, 1), learnable per block, with no manual schedule.
- **Params added:** 1 shared `W_O` (4096) + 12 `α_b` scalars (12) = 4,108 params. **No per-block parameters are removed.** Treatment has +4,108 params vs control; param delta is *symmetric* — this is a parameter-shape lever, not a model-size lever.
- **Optimization detail:** α_b is a free parameter of the per-block `attention_block` module. Standard SGD/AdamW on α_b is fine; no special optimizer (the only constraint is that the gradient on α_b is a scalar, so any optimizer works). No LR warmup trick, no α-schedule — let the model learn whether each block wants to share.

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
