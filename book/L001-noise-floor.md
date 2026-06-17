# L001 — The seed-noise floor at tiny scale

**Statement.** At tiny1m3m, the unpaired val-loss SD across seeds is ≈ 0.015; paired
differences below the 0.02 screen band are not distinguishable from seed noise without
multi-seed confirmation.

**Status.** L — strong. Measured; the constant is the basis of the screen band.

**Scope.** tiny1m3m: ≈ 1.3M params, 3M tokens, 732 steps, seq 512, Muon+AdamW. The
*number* (0.015) is scale-specific; the *existence* of a floor is general.

## Evidence
- 175-alibi re-pin across 3 seeds: 6.2650 / 6.2556 / 6.2412 → SD ≈ 0.012.
- Dozens of null brackets in `closed.md` cluster within ±0.02 of the champion with no
  consistent sign — the empirical width of "no effect."
- The screen band is set to 0.02 ≈ 1.3 × the unpaired SD; the promotion band 0.018 plus a
  3-seed paired confirm is what actually clears noise (see [[D001]]).

## Falsifier
A seed sweep at this scale showing unpaired SD ≫ 0.02 or ≪ 0.005 — either re-sets the band.

## Why it matters
Defines "clears the noise" for every other entry. The companion failure mode: a single-seed
screen-win over-states the effect by ≈ one SD (winner's curse) — drafted separately; this is
why no promotion is ever made on one seed.

Links: [[D001]], PIPELINE.md (noise constants).
