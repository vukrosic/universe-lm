# Taste log — 193 μP base init

## r2 — 2026-06-15 — verdict: revise

- **The r1 corrections are applied in good faith.** Baseline is now `std=0.02`
  (correct), the step-0 byte-identity is exact (verified algebraically:
  `(W_μP @ x) * (1/50) = (50·W_baseline @ x) * (1/50) = W_baseline @ x`), and
  the scope is narrowed to single-axis (W_emb only, W_Q/K/V/O and FFN left at
  baseline `std=0.02`). The partition table is a useful framing. These are
  real improvements over r1 and the miner should be credited for engaging.

- **But the r2 mechanism has a sign error in the gradient claim (line 44).** The
  idea states: *"the gradient on the embedding weight is 50× larger in
  magnitude than baseline (because `∂L/∂W_emb = 50 · ∂L/∂W_emb_baseline`)"*.
  Chain rule: `logits = (W_μP @ x) * (1/50)`, so
  `∂L/∂W_μP = (1/50) · ∂L/∂logits @ xᵀ = (1/50) · ∂L/∂W_baseline`. The
  gradient is **1/50 of baseline, not 50×**. The idea confuses the gradient
  on `logit_scale_param` (which IS ~50× of the "natural" logit-scale
  gradient) with the gradient on `W_emb` (which is 1/50 of baseline). This
  matters: the r2 pitch's "the optimizer can re-fit the embedding's magnitude
  freely (the gradient is 50×)" claim is wrong, and the "implicit LR
  multiplier" argument on line 50 collapses with it. For AdamW the per-param
  update is `lr · m / sqrt(v)`; with gradient scaled by 1/50, `m` is 1/50 and
  `v` is 1/2500, so the effective update on `W_emb` is **the same as baseline**
  (the 1/50 in `m` and 1/50 in `sqrt(v)` cancel exactly). The r2 pitch's
  headline bet — "lets the optimizer re-fit the input magnitude freely" — is
  not what the lever does.

- **Stripped of the gradient claim, 193-r2 is essentially 184 with a different
  init for `logit_scale_param`.** Baseline: `W_emb ~ N(0, 0.02²)`,
  `logit_scale_param = 0` (`logit_scale = 1`). 193-r2: `W_emb ~ N(0, 1)`,
  `logit_scale_param = −ln 50` (`logit_scale = 1/50`). After AdamW
  normalization the embedding update is identical, and the only new learnable
  parameter is `logit_scale_param` — which is **literally the same parameter
  184 adds** (see `models/llm.py:1408` and `models/llm.py:2016`), just
  initialized to `−ln 50` instead of `0`. The optimizer can move
  `logit_scale_param` freely, so the steady-state is the same. The unique
  signal of 193-r2 vs 184 is the embedding's *initial random sample*
  (variance-1 vs variance-0.0004) — which is not a mechanism, it's a seed
  effect. With seed 42 fixed, the "unique signal" is just one specific random
  draw's luck.

- **The "joint axis" partition is structurally misleading.** The r2 table
  presents 184 / 194 / 193 as three orthogonal probes of the
  (input-magnitude × output-magnitude) plane. But the *optimizer's effective
  view* of the plane collapses to a single axis: the output-magnitude axis
  (parameterized by `logit_scale_param`), because AdamW normalizes the input
  gradient. So 184 probes the output axis; 194 probes the input axis with no
  output compensation (its loss-preserving claim is also hand-wavy — scaling
  the embedding changes the residual stream's magnitude, which changes the
  attention/FFN outputs, which changes the LM head's input, which changes the
  loss at step 0); 193-r2 probes the output axis with a different init and a
  different embedding draw. The "joint" framing is a relabeling, not a third
  axis.

- **Portfolio is now crowded with three magnitude-space levers.** 184
  (accepted, `needs-run`) tests the output axis. 194 (pending/implementing)
  tests the input axis. 193-r2 is a third, with the weakest bet of the three
  (no unique mechanism, wrong gradient claim, relies on a specific random
  draw). Even if the gradient claim were correct, the information value is
  borderline: a null at 0.94M on a 50×-inflated embedding could mean "μP
  doesn't help" or "the specific init wasn't lucky" — indistinguishable with
  one seed. A win is hard to interpret (is it μP or the lucky draw?). A
  drift is the most likely outcome and closes nothing cleanly.

- **The μP pedigree is real but doesn't transfer to a single-tier, one-seed
  test.** μP's headline property is zero-shot HP transfer across scales; we
  have one tier per the one-tier-only rule, so the headline is not testable
  here. The init-only form of μP (which is what 193-r2 is, since the LR
  multiplier argument fails) is a known modest-gain lever, and the paper
  itself notes the gain is small relative to the full joint parameterization.
  At 0.94M the lever has no unique signal that 184 doesn't already cover.

- **Niche fit is mixed.** Mechanism? Yes (init + output coupling).
  Identity/zero-init-able? Partially — `logit_scale_param` is learnable, but
  the embedding init is not absorbable by any current parameter, so the
  *initialization* (not the steady-state) is the lever, and AdamW washes the
  initialization out. Tiny1m3m-runnable? Yes (1 param, init-only). But
  "init lever that AdamW washes out" is the wrong shape for a 92-step A/B.

- **The 3-round cap is approaching.** This is r2; one more revise is allowed
  before the forced call. The miner should use r3 to either:
  1. **Fix the gradient claim and rebuild the bet around what the lever
     actually does.** If the AdamW update on W_emb is the same as baseline,
     the lever's only effect is the `logit_scale_param` init. Argue why a
     different init for that parameter (vs 184's `0`) is worth a slot. Or
     argue why the specific random draw of a variance-1 W_emb at seed 42 is
     expected to differ meaningfully from `std=0.02` at seed 42. (Ablation:
     compare two variance-1 W_emb inits at the same seed — the variance
     shouldn't matter, only the specific draw.)
  2. **Add explicit LR multipliers** so the lever is a proper μP probe
     (e.g., `lr_emb_mult = 50`, `lr_output_mult = 1/50`). Then the bet is
     "μP-LR multipliers help at 0.94M", which is a structurally different
     lever from 184 and has a real mechanism. LoC budget is still small
     (~40 LoC).
  3. **Drop in favor of 184** and let 194 carry the input-axis probe. The
     three-lever triangulation is not worth a slot if the third probe is
     weaker than the first two.

Routing: `needs-repitch` for the miner's last round. The pitch has improved
materially (byte-identity is real, single-axis is the right scope), but the
bet is still a relabeling of 184 and the gradient claim needs to be fixed or
the mechanism needs to be rebuilt around what the lever actually does.

## r1 — 2026-06-15 — verdict: revise

- **The "Mechanism" section has a factual error on the current init.** The idea
  states: *"For tiny1m3m (d_model=64): the standard init is W ~ N(0, 1/64) =
  N(0, 0.0156)"* and *"the standard init for the LM head is N(0, 1/d_model) =
  N(0, 0.0156)"*. Both are wrong. The actual baseline in `models/llm.py:1443` is
  `torch.nn.init.normal_(self.lm_head, mean=0.0, std=0.02)` and the global
  `_init_weights` (line 1533) is `normal_(module.weight, mean=0.0, std=0.02)` for
  every `nn.Linear`. This is GPT-2's fixed-init, *not* `1/sqrt(d_model)`. So
  every numeric comparison in the "Design sketch" / "Why it might lower val
  loss" sections (the "8× larger LM head" claim, the "sharper softmax"
  argument) is anchored on a false premise. The miner must re-do the magnitude
  accounting against the real `std=0.02` baseline before re-pitching.

- **μP is a *joint* parameterization, not an init-only lever — and the proposal
  ships only the init half.** μP's headline property (zero-shot HP transfer)
  requires *both* the base init scales *and* matched per-parameter LR
  multipliers (`lr_mult = 1/fan_in` for hidden weights, `lr_mult = d_model` for
  the LM head and embedding). Applying only the init half on a vanilla AdamW at
  the same LR is not μP — it is "wrong init, wrong LR" and is a known
  recipe for early-step explosion. The proposal adds `use_mup_init: bool` with
  no companion LR-multiplier parameterization, so what would actually be coded
  is a non-μP lever with a μP label. The 92-step horizon at 0.94M has no
  tolerance for a 50×-inflated LM head running on an un-tuned AdamW.

- **The proposed magnitudes are catastrophic at d_model=64.** Re-running the
  numbers against the real `std=0.02` baseline:
  - LM head: 0.02 → 1.0 = **50× inflation** (output logit magnitudes 50× larger;
    softmax saturates on the argmax token; gradient on the 32k-vocab
    denominator collapses to ~0 in a handful of steps).
  - W_emb: 0.02 → 1.0 = **50× inflation** at the input, multiplied through
    12 residual-additions, grows the residual stream by ~3.5×.
  - W_Q/K/V/O: 0.02 → 1/√64 ≈ 0.125 = **6.25× inflation** per attention matmul.
  - W_FFN_up / W_FFN_down: 0.02 → 1/√256 ≈ 0.0625 = **3.1× inflation**.

  The lever as proposed is a near-certain DRIFT at 92 steps. The "sharper
  softmax" intuition in the idea is real, but the proposal points the
  sharpness **off the cliff** rather than tuning it. The miner must either
  (a) pick a single μP-derived axis to perturb (e.g. just the LM head, or
  just the embedding), or (b) pair the init change with the matching
  μP-LR multipliers so the optimizer's step size is on the same scale as
  the init.

- **Step-0 byte-identity is broken, and the proposal hand-waves this.** The
  idea says *"the lever is not step-0 byte-identical … the spec allows
  non-bit-identical levers"*. The spec does, but the clean A/B boundary
  matters here: with 5×–50× magnitude changes the run is dominated by
  init-perturbation recovery, not by the lever's steady-state behaviour. A
  one-tier, one-seed, 92-step A/B cannot disentangle "lever helps" from
  "init needed 200 steps to settle" — and the 184-logit-scale lever
  (already accepted) tests the *same output-magnitude hypothesis* with
  exact step-0 byte-identity. If 193 re-enters the queue, it must either
  be byte-identical (perturb only the embedding, or only the FFN, in a way
  the optimizer can absorb in step 0) or it must commit to a much larger
  run that lets the init amortize.

- **Portfolio overlap with 184 and 194 — same axis, worse hygiene.** Three
  recent levers all touch "where does the residual stream's magnitude
  come from" / "how sharp is the output softmax":
  - 184-logit-scale (ACCEPTED, `needs-run`): learned global logit scale,
    exp(0)=1 init, exact byte-identity, +1 param, info-rich in all 3 outcomes.
  - 194-embed-sqrt-d (pending taste, same gate): scalar `1/√d_model` on the
    embedding, loss-preserving at step 0, 0 params, fresh "magnitude-only"
    axis that explicitly avoids the 159-emb-layernorm DRIFT.
  - 193-mup-init (this one): non-byte-identical 50×-inflated LM head + 6×
    inflated W_Q/K/V with no LR compensation.

  184 and 194 are the clean probes of the "output temperature" / "embedding
  magnitude" hypotheses. 193 as written is a third probe of *overlapping*
  hypotheses with the worst experimental hygiene. The unique test μP could
  offer is the *transfer* property — but we have only one tier per the
  one-tier-only rule, so μP's headline (zero-shot HP transfer) is not
  testable in this pipeline. The init-only form of μP is a known
  modest-gain lever (paper says so), and at 0.94M the lever has no unique
  signal that 184/194 don't already cover.

- **The μP hypothesis is real but the *pitch* needs to be sharper.** The
  defensible re-pitch is one of these two:

  1. **Single-axis μP probe, byte-identical.** Pick *one* μP axis — e.g. "μP
     embedding: `W_emb ~ N(0, 1)` (50× current), but compensate with a
     learned global logit scale initialized to 1/50, so the *effective*
     output logit magnitude at step 0 matches baseline byte-for-byte" — that
     is `exp(logit_scale_param=−ln(50))` at init, giving `logit_scale=1/50`.
     The 50×-inflated embedding is then *absorbed* by the learned logit
     scale (which is exactly what 184 added). This tests the *init-half*
     of μP in a way the optimizer can quickly undo, and pairs naturally
     with 184 so the A/B is "μP embedding vs GPT-2 embedding, with
     compensating output scale". The two-ctrl rule then has a clear
     interpretation. (Caveat: this overlaps so much with 184 + 194 that
     the info value is borderline — the miner should argue why this is
     not redundant.)

  2. **Drop μP, lean into the residual-stream hypothesis** and let 194
     carry the lever. 194 already does the clean magnitude-only
     embedding probe; μP is a less-controlled version of the same
     experiment and a strictly worse bet for one-tier, one-seed, 92-step.

  Either way: re-pitch must correct the `std=0.02` vs `1/d_model` factual
  error, name the LR-multiplier co-design or its absence, and address
  why this is a fresh test rather than a noisier duplicate of 184 / 194.

- **Information value is low in all three outcomes as written.** A WIN at
  tiny1m3m on a 50×-inflated LM head is hard to interpret — could be "μP
  init is better" or "any init with larger LM head is better and 184 will
  close that hypothesis for 1 param". A NULL closes nothing cleanly because
  the magnitude change is so large the optimizer never gets to the
  steady-state regime. A DRIFT is the most likely outcome and tells us
  only "this particular over-aggressive μP instantiation is bad", not
  "μP at 0.94M is bad".

- **Niche fit is mixed.** Mechanism? Yes (init is architectural). Identity/
  zero-init-able? No — the proposed magnitudes are *not* absorbable by
  the existing final-RMSNorm or any current learnable parameter, so step-0
  is materially different. Tiny1m3m-runnable? Yes (0 new params,
  one-shot init). But "init lever that materially changes step-0" is
  borderline in a pipeline that prizes clean A/B boundaries (cf. 184's
  exact byte-identity).

Routing: `needs-repitch` for the miner, with the corrections and
single-axis re-pitch options above.
