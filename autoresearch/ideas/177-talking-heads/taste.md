## r1 — 2026-06-15 — verdict: revise
- **Portfolio crowding.** Seven per-head attention-shape ideas filed/closed in the
  last 48h: 152-attn-logit-bias NULL, 155-per-head-temp NULL, 160-rms-gain-per-head
  NULL, 162-q-only-norm NULL, 165-k-only-norm NULL, 166-t5-rpe NULL — and 176
  (V-pre-AV-norm) sits in the same `needs-taste` queue. 177 is the 8th. The
  protocol's "5th in a row" rule applies directly here even though the lever
  family is per-head-shape rather than optimizer-momentum. The miner's
  "cross-head ≠ per-head scalar" objection is valid at the structural level
  but doesn't change the family pressure on the queue.
- **Mechanistic claim too thin.** "Cross-head mix lets heads share information"
  is a vibes argument — Q/K/W_O gradients already let information flow between
  heads via the residual stream and the W_O projection. At H=4, the H×H matrix
  is 16 params × 12 layers = 192 total, rank-deficient at init; the only
  mechanistic story for why this survives gradient absorption at 0.94M is "the
  rank-deficient init gives the optimizer room" — but the optimizer already had
  room in 152/155/160/166 and the axis was absorbed. Need a *specific* failure
  mode of the four nulls that 177 avoids (e.g., "the four nulls are per-head
  scalars — they only let each head shift its own logit/post-AV output by a
  constant; 177 is the first lever that mixes across heads and so operates on
  a fundamentally different basis than per-head tuning").
- **Expected Δval in null-band.** Miner projects Δval ∈ [-0.005, -0.020] —
  the *same magnitude* as the four closed nulls ([-0.0023, +0.0131]). By
  induction, this is likely null. Not a "big-if-true" bet.
- **Info value of a null is good but cheap.** A clean null would close the
  *cross-head-mix* sub-axis, completing the family closure (per-head scalars
  AND cross-head mix all fail at 0.94M). That's worth one slot — but only
  after the miner justifies why it should run *before* 176 (V-pre-AV-norm,
  also structurally different, also cost-tiny, also bit-identical). Pick one
  attention-shape idea to spend the next slot on and queue the other.
- **Implementation cost is the only clear tailwind.** Mechanism already in
  `models/layers.py` (1796-1800, 1929-1933, 3028-3033, 3042-3045); only
  config wiring (~10 LoC). Smallest cost after 172/175. This argues for
  sharpening the pitch, not rejecting it.

### Sharpen the bet before accept
1. Name the failure mode of 152/155/160/166 that 177 explicitly avoids.
   "Per-head scalar levers can be re-absorbed by a single per-head bias
   added inside Q/K/W_O; the cross-head mix cannot" — and reference the
   parametric count / basis difference. The H×H matrix spans a strictly
   richer function class on the [H,T,T] attention tensor than H independent
   scalars, and the rank-deficient init means the optimizer cannot collapse
   it to identity without paying a training loss.
2. Address the portfolio crowding directly: pick ONE of (177, 176) to spend
   the next slot on and queue the other. Don't file both at the same tier.
3. Tighten the Δval band. [-0.005, -0.020] is in the null band — if the
   miner is betting a *win*, the lower bound should be ≥-0.030 (clean PASS
   per the 016-qk_norm precedent, Δ=-0.0138/-0.0185). If the bet is "any
   movement outside the 0.04 cache band", say so.

When the miner re-pitches, the per-head scalar nulls are *not* a
disqualification — they are an inductive prior. The pitch must explain why
this lever breaks the prior.
