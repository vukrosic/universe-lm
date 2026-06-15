# Taste log — 192 topk-attn

## r1 — 2026-06-15 — verdict: revise

The lever sits in a 5-deep sparse-attention family that has already produced one
cap-hit (173-entmax, axis abandoned), one closed soft-sparse (022-softpick), one
null windowed (182-per-head-window), and one massive DRIFT (177-talking-heads,
Δ=+0.9509, softmax modifications blow up at d_k=16). 192's pitch doesn't
defend against that family strongly enough to take a slot in this round.

- **Two parameterizations bundled, neither committed.** The spec opens with
  `k = T/4 = 512` as the default working point (hard sparsity, non-byte-
  identical at step 0), then immediately offers "Alternative parameterization"
  with a learnable `k_l` per block init at T (byte-identical at step 0,
  +12 scalar params). These are two different levers with two different
  step-0 stories, two different param counts (0 vs 12), and two different
  gradient regimes (boundary-discontinuous sort vs smooth per-block learn).
  A/B with the default `k=T/4` is a structural-sparsity test; A/B with
  learnable `k_l` is a learned-regularization test. The implementer will
  pick wrong. Pick ONE and commit.

- **The "fixed > learned at 0.94M" bet is not defended against 173's null.**
  173-entmax-15 closed at the recode cap; the axis was abandoned because
  learned sparse softmax could not find the right support at 0.94M. 192's
  claim is "fixed k is simpler than learned support, so it should win where
  173 didn't." That's plausible but the mechanism isn't named: at 0.94M /
  d_model=64 / 12L, what *specifically* about fixed k should bind that
  entmax-1.5's bisection-Lagrange support couldn't? Two testable answers:
  (a) the bisection on the Lagrange multiplier adds gradient noise near the
  support boundary that a fixed-k lever has no analog of (this is real,
  entmax-1.5's gradient through the support boundary is famously noisy);
  (b) fixed k is a tighter prior that reduces effective hypothesis class,
  trading flexibility for sample efficiency. Pick (a) or (b), name the
  mechanism, and predict a magnitude.

- **Default `k=T/4` is hostile at d_k=16.** At T=2048 with default k=512
  (75% sparsity), each query attends to exactly 512 of 2048 keys. Combined
  with d_k=16 and 6 heads (the config that's already proven hostile in
  177-talking-heads Δ=+0.9509), the forced information bottleneck is
  structurally risky. The closest existing data point: 154-rebased-attn
  WIN (record trt=2.9628 — note: that run hit a bug pre-correction;
  post-correct the magnitude is smaller, but still a WIN per closed.md),
  which says *locality* priors help; top-k is the opposite — it enforces
  hard global sparsity, not locality. If `k=T/4` is the default, the lever
  is competing with rebased, not complementing it. State which.

- **No magnitude. The bet is a vibe.** "A null at 0.94M closes the
  fixed-sparsity axis" is not a prediction. What's the predicted Δval?
  Three realistic outcomes given the family priors: (i) sub-noise null
  inside |Δ|<0.01 (most likely, given 173 cap-hit + 182 null + 022
  closed); (ii) DRIFT +0.01..+0.05 if the forced bottleneck breaks long-
  range signal; (iii) WIN in the [-0.01, -0.03] window only if rigid
  fixed-sparsity truly beats learned-support at d_k=16. Pick one as the
  primary prediction, with the mechanism that opens it.

- **Byte-identity claim contradicts the default.** The spec says
  step-0 must be byte-identical to baseline. With `k = T/4 = 512`, step-0
  is non-trivially different — the topk_idx scatter writes a binary mask
  over the scores, the softmax then renormalizes over k positions not T.
  At step 0, scores are random Gaussians; the top-k of a Gaussian over
  2048 positions are not the full distribution. So at step 0 the model
  is attending to a *different* set of keys than baseline. The pitch
  concedes this. The lever is in the same "non-zero-init" category as
  022-softpick / 173-entmax / 154-rebased. That's fine for the screen,
  but the slot must justify itself against that category's results, not
  the byte-identity zero-init family.

- **Transfer check is fine; the med tag holds.** Top-k is validated at
  100M-300M on vision (Touvron 2021) and sparsemax at 30M-100M on
  classification. The mechanism is scale-stable. transfer-risk: med is
  reasonable — the *forced sparsity ratio* matters at scale (e.g. k=128
  for T=8192 is 98% sparsity, much harsher than 75% at T=2048), and the
  sweet spot may shift. The 135M recipe could test k ∈ {256, 512, 1024}
  for T=4096 with the ratio kept constant. Don't need to lock this in
  now; just flag it for the 135M-stage definition.

**Concrete revision path for round 2 (one shot remaining):**

1. **Commit to ONE parameterization.** Either:
   (a) Hard-fixed top-k as a *structural sparsity lever*. Drop the
   learnable `k_l` alternative entirely. Default `k = T/4 = 512` for
   T=2048 (75% sparsity), 0 new params, step-0 non-identical (frame as
   "structural lever, same category as 173/022"), pass/fail at
   tiny1m3m/seed-42.
   (b) Learnable `k_l` per block, init at T (byte-identical step 0), 12
   scalar params, optimizer grows or shrinks. Default = T/4 only after
   step 0 (use a *schedule*: `k_l = T - (T - T/4) * σ(step)` with a sharp
   ramp around step 500 to mirror the entmax-1.5 → top-k transition the
   miner is implicitly comparing). Pick (a) if you want a clean null/info
   result; pick (b) if you want a byte-identical lever that gets a free
   bias-init.

2. **State a sharp bet with a mechanism and a magnitude.** Suggested
   form: "We predict Δval ∈ [-0.005, +0.005] (sub-noise null) because
   {mechanism}. If the lever wins by Δval < -0.01, the mechanism is
   {X}. If the lever drifts by Δval > +0.01, the mechanism is {Y}."
   Engage the 177-talking-heads DRIFT explicitly: top-k's hard sort
   differs from talking-heads' H×H soft mixing because {…}, so the
   d_k=16 hostility doesn't transfer.

3. **Defend against 173 with mechanism (a) or (b) above.** Without
   this, the slot is "5th sparse-attention lever that ignored the prior
   4" — that's a portfolio-crowding revise regardless of merit.

4. **Address 154-rebased.** Top-k is "rebase + hard sparsity"; rebased
   WIN says locality helps; top-k enforces global sparsity. If k=T/4
   the lever is *subtractive* relative to rebased, not additive. Spell
   out whether 192 is competing with or complementing rebased.

The lever has genuine novelty (fixed vs learned support, hard vs soft
zeroing, sort-discontinuity in gradient) and the closed family has only
touched the *learned*-support and *windowed* axes. A clean re-pitch that
commits to one parameterization, names a magnitude, and engages the
prior nulls is worth a slot. The current pitch is not.
