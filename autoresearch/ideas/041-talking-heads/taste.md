# Taste log — 041 talking-heads

## r2 — 2026-06-11 — verdict: accept
- **All five r1 concerns closed.** (1) Variant selected: post-softmax weight
  mix only — drops the paper's pre-softmax mix and the asymmetric variants,
  which is the right call since 042 occupies the pre-attention channel.
  (2) Head-budget concern engaged head-on: "a 4×4 H×H mix at identity init is
  the *full* rank linear transformation on the 4-head space — it's not
  'near-trivial' in a math sense, it's exactly the constraint we want to
  test." That's the right framing — full-rank-on-the-head-axis makes the
  H=4 ablation mechanistically informative, not a rounding error.
  (3) 042 portfolio conflict is now partitioned cleanly: 041 sits on the
  probability tensor (B, H, T, T) after softmax, 042 sits on the Q/K/V
  feature tensor (B, H, T, d_k) before the dot product — different
  tensors, different information pathways, identity init on both keeps
  each A/B parameter-clean. (4) Identity init is the load-bearing
  redeemer and the repitch leans on it correctly. (5) Null is sharpened:
  a H=4 GQA null on a 4×4 post-softmax mix tells us the *probability-side*
  cross-head term is redundant when concat-side mixing is unconstrained —
  a specific, 135M-actionable lesson, not a generic "head-mixing doesn't
  help."
- **Leverage is honest.** The repitch says "measurable but small" and frames
  the slot as informative on *which channel* cross-head mixing matters,
  not as a "we expect to win big." That's the right calibration at H=4 —
  the bet is on information value, not magnitude.
- **Transfer risk is correctly `med`, not `low`.** Mechanism is real and
  complements the existing W_O; gain magnitude will be smaller at H=4 than
  at the paper's H=12-24. The direction-of-effect bet is clean.
- **Niche fit is solid.** Identity-init, ~16 params, one einsum per block,
  zero FLOP overhead, fits the 200-LoC budget. The 135M recipe carries
  this slot forward only if the H=4 result hits; a clean null with the
  sign-magnitude specified frees parameters for stronger candidates
  (per the repitch's "e.g., second W_O, or head-axis MoE"). That's the
  right portfolio behavior.
- **Verdict: accept.** Sharp bet, sharp null, parameter-clean A/B, distinct
  channel from 042. Move to the definition gate.

## r1 — 2026-06-11 — verdict: revise
- **Head budget kills the lever.** tiny1m3m runs `n_heads=4` with GQA
  (`n_kv_heads=2`). Talking-heads inserts H×H mixes — that's a 4×4 matrix on
  the score and weight tensors. Paper gains live at H=12–24 in T5/BERT, where
  the mixing matrix has real capacity. A 4×4 mix is a near-trivial lever and a
  null result here teaches us almost nothing about 135M (H≈8–12). Re-pitch must
  name this head-count gap and argue why a 4×4 mix is still informative — or
  drop it.
- **The bet is vague — pick a variant.** The Shazeer paper has multiple
  knobs: pre-softmax score mixing, post-softmax weight mixing, both, and
  asymmetric (logit-only) variants with different head counts. "Two H×H
  matrices both initialized to identity" is one of many; the paper shows
  pre-softmax is the dominant lever. State which variant we run and why, with
  one sentence of "we expect X because Y" tied to the variant.
- **Portfolio conflict with 042-knocking-heads.** 042 is the same cross-head
  family with stronger paper evidence (1T tokens, 6.1B MoE pretraining) and
  `transfer-risk: low` vs this idea's `med`. Both can't justify a slot. Either
  sharpen 041 to test a *distinct* mechanism (e.g., post-softmax weight mix —
  041 — vs pre-attention feature mix — 042), or step aside for the sibling.
- **Identity init is the redeeming feature.** Both H×H matrices identity-init
  is clean and zero-overhead at step 0 — that's the niche-fit anchor; keep it
  and lean on it in the re-pitch.
- **Sharpen the null.** Rewrite "Why it's worth a slot" so the null result
  pays off concretely: e.g., "null at H=4 GQA tells us probability-tensor
  mixing is not a binding bottleneck when KV is already shared, so the 135M
  recipe can skip post-softmax mixing and only consider pre-softmax variants."
  Right now the null says "head-mixing isn't binding" which is too generic.
