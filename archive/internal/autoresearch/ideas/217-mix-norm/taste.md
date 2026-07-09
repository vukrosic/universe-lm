# Taste review — 217-mix-norm

## r1 — 2026-06-16 — verdict: revise
- **Norm-architecture portfolio at tiny1m3m is heavily saturated.** closed.md and recent
  close lines already log 10+ nulls in this family: 017-sub-LN, 111/116/130/142
  (depth-conditional scalar per block), 152 (per-head bias), 155 (per-head temp),
  159-emb-LN (DRIFT), 160-rms-gain-per-head, 162-q-only-norm, 165-k-only-norm,
  169-qk-norm-depth, 176-v-pre-av-norm, 181-cross-head-RMSNorm (DRIFT),
  183-pre-lm-head-RMSNorm, 190-per-layer-qk-norm, 210-qk-layernorm-alibi, plus
  the closed.md "Norm zoo" axes line (pnorm / manhattan / center / squash /
  clip / channelscale). 217-mix-norm adds yet another per-block norm knob to
  this saturated family. Even on WIN, the marginal information per slot is
  modest; on NULL, it is the ~11th-15th null in the family — low info value
  unless the WIN hypothesis is sharp enough to distinguish from coincidence.
- **Horizon-scaling is the binding risk.** 12 learnable α_l scalars over ~92
  update steps (post-warmup) ≈ 0.008 update-steps per param — the exact
  regime that closed 110/121/122/124/134 (per-block learnable scalars in the
  same update-step budget). Init at α_l = +4.6 (sigmoid ≈ 0.99, ≈ pure
  RMSNorm) is also *deeply* in the saturated regime of the sigmoid: |σ'(4.6)|
  ≈ 0.009, so even a strong logit pull produces a tiny shift in the mix
  weight. The pitch acknowledges the null risk but does not quantify it. Make
  the update-step economics explicit and pick one of two recovery options
  before re-pitching.
- **Two concrete init/control options to choose from.** (a) Keep α_l = +4.6
  init for byte-identity, BUT add a *separate* movement diagnostic: log mean
  |α_l_final − α_l_init| and val_acc on the ctrl seed; declare a 217-null
  uninformative if mean |Δα_l| < 0.1 nats regardless of val delta. (b) Accept
  a non-byte-identical init closer to 0 (sigmoid = 0.5, equal mix) and pair
  with a byte-identical control run; the α_l scalars then sit in the
  high-gradient regime from step 0, but the comparison is unpaired. Pick one
  and commit in the re-pitch — don't ship both options as alternatives.
- **Sharpen the bet, in one sentence.** As written the pitch reads "let the
  model learn which blocks prefer which centering" — passive exploration.
  Pre-commit to a directional hypothesis the run can confirm or kill, e.g.
  "early blocks (L1-4) want LayerNorm (mean-centering stabilizes the
  post-embedding activation), late blocks (L9-12) want RMSNorm (the residual
  stream is already calibrated by the time tokens get there)." Pre-committed
  direction makes a WIN readable; a passive null is uninformative. State it
  in the `plain:` frontmatter as a one-sentence claim.
- **Distinguish from 016-qk-norm (WIN) explicitly.** The pitch claims 217 is
  the "other free norm knob" vs the WIN 016 axis (pre-softmax attention
  normalization). That is true but the *binding* of 016 at 0.94M is a
  softmax-attention specific finding — pre-softmax QK normalization lowers
  logit magnitudes before softmax temperature kicks in. Pre-residual-stream
  norm mixture operates on the residual-stream activation, NOT on the
  attention path. State this explicitly so the run doesn't get conflated with
  016's mechanism if it wins (or nulls).
- **Transfer.** `transfer-risk: low` is plausible — the mechanism is
  purely architectural and the per-block mixture is a real (if novel) axis.
  At 135M the model has more layers per block and ~140× more gradient
  signal per α_l scalar, so a 0.94M-null doesn't necessarily kill transfer.
  But if 217-nulls at 0.94M you will need a separate argument to re-test at
  135M beyond "maybe it works bigger" — name what changes at 135M that lets
  the α_l scalars actually move.

**Why revise, not reject:** the lever is real (mixing two named norm families
at block level is a real architectural choice not in the closed portfolio
under that exact framing), the cost is cheap (~+12 params, one forward-pass
mix), and the path above makes the bet sharp enough to be informative
either way. The blocking issue is that the current pitch is too passive
to teach us anything new in a saturated family.
