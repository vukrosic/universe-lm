# DRAFT — A single-seed screen-win over-estimates the honest effect by ≈ one noise SD

**Hypothesis.** At tiny1m3m, when a paired Δ is selected *because* it cleared the 0.02
screen band on one seed, its honest ≥3-seed paired mean is systematically *smaller* (less
favourable) by ≈ 0.013–0.015 val-loss — about one noise-floor SD ([[L001]]) — so a 1-seed
screen-win is not evidence of a real effect.

**Believed because.** Winner's curse / selection bias: the screen reports the seed (and the
candidate) that *looks* best, so its expectation conditional on being the max is biased
upward by ~O(σ). Two clean in-scope cases, both shrinking by ≈ σ:
- **175-alibi** re-pin: single seed **6.2403** → honest 3-seed mean **6.2539**
  (6.265 / 6.2556 / 6.2412). The single value was **0.0136** optimistic.
- **306-×2.0-LR**: seed-42 paired **Δ−0.0248** (a screen-win) → honest 3-seed paired
  **Δ−0.0104** (sub-band). The single value overstated the effect by **0.0144**.
Both shrinks ≈ the L001 SD (0.015). Counter-pressure that it is *not always* fatal:
323-mom0.90+×2LR's seed-42 component (Δ−0.0185) held up under 3 seeds — so a 1-seed win
*can* be real; the point is you cannot tell which from one seed.

**Test.** Already-run paired confirms supply the data: for each idea that produced a 1-seed
screen-win, compare its seed-42 paired Δ to its ≥3-seed paired mean. Tally the signed
shrink (honest − screen). Falsifiable on the sign and magnitude of that tally.

**Predicted.** Mean signed shrink ≈ +0.013–0.015 (honest less favourable), with most
1-seed |Δ| in the 0.02–0.03 band collapsing to sub-band under 3 seeds.

**Promotes to.** **L?** now (two consistent cases + a deductive mechanism). → **L!** once
≥3 documented inflation cases show a consistent shrink magnitude with a CI on the shrink
that excludes 0. The *mechanism* (winner's curse) is deductive; only the *magnitude at this
scale* is empirical.

**Falsifier.** A set of ≥3 single-seed screen-wins whose 3-seed confirms hold at the **same**
effect size (no systematic shrink, shrink-CI includes 0) → kills the inflation law; the
screen would then be an unbiased estimator and the ≥3-seed confirm gate redundant.

**Operational rule it already implies.** Never promote on 1 seed — the daemon's lucky-seed
guard (1-seed WIN → `needs-confirm`, never auto-promote) is this law made into policy.

**Evidence so far.** partial — two paired cases above (175 re-pin log; 306 confirm,
`_arq_confirm_306`-class). Needs a third independent case for L!.

**Blocked on.** One more 1-seed-win→3-seed-confirm pair logged with its shrink.
