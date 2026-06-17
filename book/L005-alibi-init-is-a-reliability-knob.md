# L005 — ALiBi slope init is a reliability knob, not (mainly) a score knob

**Statement.** How the per-head ALiBi slopes are *initialized* (geometric vs uniform
seeding) primarily buys training **reliability** — roughly 3× lower run-to-run variance,
fewer bad seeds — and only a marginal mean-val improvement.

**Status.** L — strong for the *reliability* claim; **L? (tentative)** for any *mean-score*
claim (the mean Δ sits right at the band, carried by lucky seeds).

**Scope.** tiny1m3m.

## Evidence
- ALiBi deep-dive: geometric slope seeding cut run-to-run variance ≈ 3× vs uniform, with the
  mean roughly unchanged.
- Box-era, on the champion: 290-uniform-slope Δ −0.0190 and 291-geometric-slope Δ −0.0200 —
  right-sign but *at* the 0.02 screen bar, and 291 was a single-seed screen-win parked by the
  lucky-seed guard (never confirmed at the mean). So: reliability strong, mean score marginal.

## Falsifier
A ≥3-seed paired confirm showing slope-init moves the *mean* past 0.02 with CI excluding 0 —
would upgrade the mean claim from L? to a genuine score lever.

## Why it matters
Reframes a "tuning knob" as a *variance* control: use geometric seeding to stop bad-seed runs,
not to chase val. A clean example of an effect that is real (reliability) but not where you'd
naively bank it (score) — and of why single-seed wins on it must be distrusted ([[L001]]).

Links: [[L001]], [[L004]].
