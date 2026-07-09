---
id: 037-lookahead
status: needs-plan
round: 1
updated: 2026-06-11T01:34:07Z
transfer-risk: med
---

# 037 — Lookahead (k steps forward, 1 step back)

## Source
Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back"
(arXiv:1907.08610, NeurIPS 2019). Generic wrapper, not optimizer-family.

## Mechanism
Wrap the existing Muon + AdamW(vocab) pair. Maintain a slow-weight buffer
`φ` alongside the fast weights `θ`. Every `k` inner steps, take `k` normal
inner steps, then snap: `φ ← φ + α(θ - φ)` and `θ ← φ`. Identity at `α=0`
(bit-identical baseline). Pure wrapper, ~30 LoC: an `OptimWrapper` that
holds the existing optimizer + a `state_dict` slot for `φ`, called from
`trainer.step()` only when `step_idx % k == 0`. Inner optimizers are
untouched — this composes with 015 Moonlight-Muon and 011 Cautious-Lion
on the same `θ`.

## Scale evidence
The paper's strongest signal is broad-task variance reduction (vision,
translation, GANs); the LLM-pretrain case is under-tested — no 100M+
headline. transfer-risk: med (fair; downgrading from any "Muon at 135M"
claim). The scale evidence is **weaker** than 015/011's Moonlight /
Cautious-Lion; the bet lives on the *mechanism* (see below), not the
empirical ladder.

## Why it's worth a slot — the sharpened bet

**Portfolio fit (addresses crowding).** The 9 optimizer-cluster ideas
(031-040) all *replace the inner step*. Lookahead is the only one that
*wraps* an inner step without changing it. The closest closed analog is
018-ademamix, which failed taste for being a *slow-EMA* inner-step
modifier that needs ≥100k steps to converge its EMA — Lookahead is
mechanically distinct: at every `k` inner steps it does a one-shot
pullback `φ ← φ + α(θ-φ)`. The slow weights are *not* an exponential
mean of the trajectory; they're an anchor that the fast weights snap to
every `k` steps. This is bounded, not asymptotic.

**Failure mode named (addresses vague-bet).** The bet is that tiny1m3m's
**late-stage AdamW trajectory on the embedding/head oscillates** around
its running mean — the LR schedule has decayed but the per-step gradient
noise on the sparse vocabulary rows hasn't. Moonlight-Muon (Δ-0.0138
closed-WIN) fixed the *magnitude* axis of the inner step; Lookahead
fixes the *temporal* axis (trajectory oscillation). Compositional
hypothesis: 015 × 037 should be additive on the embedding/head slot.
The 92-step budget is long enough to see 18 snap events at k=5 — not
a slow-EMA convergence, a per-event pullback that fires every time.

**Step-budget math (addresses 018-ademamix problem).** With `k=5, α=0.5`,
every 5 inner steps the fast weights snap halfway back to `φ`. After 18
snaps in 92 steps the trajectory has been *bounded* by `α·‖θ-φ‖_max`
for the entire run — not a slow buildup. By contrast 018's EMA half-life
of ~7k steps against a 92-step run means 99% init-weight at the end
(closed.md:36). The math: Lookahead's variance reduction is per-snap
*not* asymptotic, so 18 events is plenty to fire the mechanism.
Defensible defaults: `k=5, α=0.5` (the paper's "fast" setting).

**Informative-null framing (addresses the uninformative-null problem).**
Both outcomes teach us something:
- **WIN** (Δ ≤ −0.01): the inner Muon+AdamW trajectory is oscillating at
  92 steps; a wrapper fixes it without changing the inner optimizer →
  015 (Moonlight-Muon) and 037 are *compositional*, and we should plan
  015×037 as a follow-up.
- **NULL** (|Δ| < 0.01): the inner trajectory is *not* oscillating at
  this step count → wrappers add no value, and 015 is the right place
  to keep spending optimization budget (we already have a -0.0138 WIN
  there, no need to layer). The 011 Cautious-Lion -0.0312 WIN tells us
  sign-masking helps; a Lookahead null would tell us the issue is
  *step direction*, not *step magnitude or trajectory shape*.

**Observable.** Per-step val-loss Δ between successive eval points in
the last 20 steps. WIN case should show that std-dev *decreases* (the
trajectory is being pulled back to `φ`); null case should show it
unchanged vs ctrl. This is logged in the run evidence.

## Hypothesis
Δ in [−0.01, −0.03] val loss on tiny1m3m / seed 42 vs the bare
Muon+AdamW(vocab) baseline. Asymmetry: a clean null is informative too
(closes the wrapper class at this scale, see above).
