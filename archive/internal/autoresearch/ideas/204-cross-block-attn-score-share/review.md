# Review log — 204 cross-block-attn-score-share

## r1 — 2026-06-15 — verdict: approve

The pre-softmax attention-logit axis is a genuinely new leg of the cross-block
mixing family at 0.94M: 021 is residual-stream V (WIN), 164 is residual-stream
Q (NULL wrong-sign), 168 is post-attention AV (NULL), 186 is within-block V
time-axis (recoding-budget exhausted), 188 is W_K/W_V projection-level
(needs-review), 196 was residual-stream EMA (taste-reject), 150 was
cross-block attention *output* (rejected on divergence). 204 blends
*detached pre-softmax scores* (Q·K^T/√d_k, not softmax output) between
adjacent blocks via a per-block learnable scalar α=0-init — that is a
functionally distinct axis from every closed or active leg, and the
attribution read is sharp: WIN = attention-pattern propagation is a
missing depth lever; NULL = V-side residual carry is uniquely binding and
the other four axes are dead ends. Identity-init is clean (`sigmoid(-10) ≈
4.5e-5` ⇒ forward graph ≈ baseline at step 0). +12 α scalars is the
cheapest cost profile in the family (+0.001%). The Memorizing-Transformers
citation is honest (Sukhbaatar et al. ICLR 2022, arXiv:2203.08913 — a real
2022 paper — but the validation is cross-document memory, not
within-model cross-block score reuse; `transfer-risk: med` matches).
Source real, mechanism structural, scope tiny1m3m-only, family
attribution accurate, scale evidence present and honest, falsifiable in
<200 LoC. Approve.

### Findings (for the reviser; not blocking — name the section, name the fix)

- **A. Add a `## Pass bar` section with a concrete Δval number against a
  real control.** The idea names a *bet* (WIN vs NULL outcome meaning)
  but does not pin a Δval target. The taste review alludes to
  `Δ=−0.034` (021's analog) and `±0.04` noise band, but the spec itself
  must commit. Suggestion: `Δval ≤ −0.02 vs 4-ctrl cluster mean, train
  right-sign, no NaN through 92 steps, val_acc right-sign`. The plan
  gate will need this number — lock it in the spec.

- **B. Spell out the `detach()` placement and the *exact* tensor type of
  `prev_block_scores` in `## Design sketch`.** The sketch says
  "previous block's scores must be detached" but does not pin whether
  that tensor is the pre-softmax QK^T/√d_k logit (the intended) or the
  post-softmax attention distribution (a different lever, not what
  taste accepted). One sentence suffices: "`prev_block_scores` is
  `Q_{b-1} · K_{b-1}^T / √d_k` *before* softmax, with `.detach()` so
  gradients flow only through `α_b` and the current block's Q/K — never
  through the previous block's QK computation." Without this, a code
  drop that grabs `attn_b.detach()` instead would test a different
  mechanism silently.

- **C. Add a one-paragraph `## Why 204 is not 150-xlayer-feedback` diff
  in the spec.** 150 was rejected on cross-block attention-path
  divergence at 0.94M (raw 11.39, two recodes, cap hit). 150 fed
  cross-block attention *output* into the residual stream at
  d_model=64/12L; 204 blends detached *logits* with no residual-stream
  interference, no full QK-output carry, and a much weaker effective
  gate (`α·sigmoid(0) ≈ α·0.5` against a 0.5+0.5 blend, not 1.0+0.5 like
  150's path). The contrast is what makes 204 plausibly a *softer*
  intervention; spell it out so the runner doesn't re-litigate the 150
  failure mode.

- **D. Tighten the "bit-identical at step 0" claim.** `sigmoid(-10) =
  4.5398e-5`, not 0. The forward graph is *practically* baseline
  (max-abs-diff < 1e-4) but the literal "bit-identical" wording will
  get the plan gate dinged on the baseline-parity self-check. The plan
  must explicitly assert: `step-0 max-abs-diff across all 12 blocks <
  1e-4` (not < 1e-6 like a literal zero-init). One-sentence precision
  fix in the design sketch.

- **E. Specify the α param group in the design sketch.** With 12 α
  scalars at `α_raw = -10` init, the plan must say whether they ride
  in the Muon group (consistent with the only-learned-scalar pattern
  in 021's value-residual) or AdamW. Muon is the right call — α is a
  1-D scalar that benefits from the same LR scale as 1-D gain
  parameters, and putting it in AdamW at 0.024 peak LR is ~10× too
  hot. One sentence, no debate needed.
