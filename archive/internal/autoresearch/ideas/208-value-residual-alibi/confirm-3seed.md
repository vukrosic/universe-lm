# 208 value-residual-alibi — paired 3-seed CONFIRM (2026-06-15)

The single-seed screen judged 208 at **6.2594 vs the pinned champion 6.2403 →
Δ+0.019** and logged NULL. But +0.019 sat *inside* the within-session 2σ, so it
was **inconclusive, not refuted**. This is the paired confirm: both arms (champion
= `Tiny1M3MAlibiConfig`; treatment = champion + `use_value_residual=True`) at 3
seeds in the **same session on the same box**, so only within-session noise is in
play. Runner: `_arq_paired_vres_confirm.py` (driver `/root/run_paired.sh`, tmux `paired`).

## Raw results (val loss)

| seed | ctrl (champion) | trt (+value-residual) | paired Δ = trt − ctrl |
|---|---|---|---|
| 42  | 6.2650 | 6.2572 | **−0.0078** (trt better) |
| 123 | 6.2556 | 6.2512 | **−0.0044** (trt better) |
| 7   | 6.2412 | 6.2606 | **+0.0194** (trt worse) |

- ctrl 3-seed: mean **6.2539**, std 0.0120, median 6.2556
- trt  3-seed: mean **6.2563**, std 0.0048, median 6.2572
- **paired Δ mean = +0.0024 ± 0.0086 (SEM)**, 95% CI [−0.015, +0.020] — spans 0.

## Verdict: NULL (confirmed, not noise-masked)

value-residual (021) does **not** stack on the alibi champion at tiny1m3m. The
true effect is centered on **zero** (+0.0024), with the band spanning zero. Unlike
the screen's "+0.019," this is a *confident* null: even the best seed (−0.0078) is
nowhere near a win, and the per-seed scatter (σ≈0.015) is larger than the mean
effect. The known V-side win (021, Δ−0.034 vs the *bare* base) is already absorbed
by alibi — they are not additive at this tier.

## The bigger finding — the pinned champion val is optimistically biased

The champion is pinned at **6.2403**, but three fresh runs of *that exact config*
gave 6.2650 / 6.2556 / 6.2412 → honest 3-seed mean **6.2539**. The pin is **+0.0136
below its own 3-seed mean — below all three fresh control runs.** 6.2403 was a lucky
single seed.

Consequence: every treatment in the 208–216 batch was judged against a bar ~0.014
**too low**, which systematically made each stack look "+Δ worse" than it is. This
compounds the noise-band problem (see `autoresearch/PROMOTION.md`,
`tools/autoresearch/NOISE-AND-BAND.md`). **Recommended fix: re-pin the champion val
to its 3-seed mean (6.2539) so the screen bar is honest.** Not auto-applied — re-pin
changes every future verdict, flagged for a human call.
