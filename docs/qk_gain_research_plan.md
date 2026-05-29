# QK-Gain Research Plan

## What's done

Three-seed pilot at **25M tokens × 25M params**, sweep vs baseline:

| Seed | Baseline | QK-Gain | Delta |
|------|----------|---------|-------|
| 42   | 4.3820   | 4.3798  | +0.0022 (QG wins) |
| 43   | —        | —       | running |
| 44   | —        | —       | pending |

**Signal so far:** directionally positive but tiny — well below the 0.01 promotion margin. With one seed the noise band is too wide to interpret.

## Problem: val loss is noisy

Single-seed val loss at fixed token budget has high variance. Comparing raw final values between two variants with only 1-3 seeds is unreliable — the noise band overlaps.

## Fix: smooth val loss with exponential moving average

For each run, compute a smoothed val loss curve:

```
 smoothed_vl[t] = α * raw_vl[t] + (1-α) * smoothed_vl[t-1]
```

Use α=0.3 (fast adaptation). Take the **minimum of the smoothed curve** as the comparison point — not the raw final value. This strips out the oscillation noise and gives a cleaner signal for whether qk_gain actually converges lower than baseline.

Also report **gap at matched step** rather than gap at fixed token budget, since steps can misalign when batch size / seq length differs slightly across runs.

## Decision rules

- ≥2/3 seeds showing smoothed_min(qk_gain) < smoothed_min(baseline) by >0.005 → promote
- ≥2/3 seeds showing delta within ±0.003 → null result, drop qk_gain
- Any seed showing NaN / divergence → hard fail, stop that config

## Next steps

1. **Wait for seeds 43 and 44** — current priority
2. **Add smoothed metric to sweep.py** — update `final_metrics.json` with `smoothed_min_val_loss` computed from the history
3. **Run at 5M × 100M next** — cheaper, will give another 3-seed read before scaling to 70M
4. **If signal holds at 5M × 100M**, run 25M × 500M for confirmation
5. **If signal holds at 25M × 500M**, ship to main and open `experiment/qk-gain` PR

## Tunables

| Parameter | Current | Range to test |
|-----------|---------|---------------|
| `qk_gain_init` | 1.0 (default) | 0.5, 2.0, 4.0 |
| `gain_schedule` | fixed per head | learned, per-layer |
| α for smoothing | 0.3 | 0.1, 0.5 |

Start with α=0.3 as the standard. If it over-smooths (lags behind real improvement), drop to 0.1. If under-smooths (still noisy), raise to 0.5.