# RMSNorm ablations — results

Filled as runs land. Control = `Screen10M20MConfig` plain RMSNorm. **3-seed
mandatory** — norm results are seed-noisy; single-seed wins are noise. "Live" =
3-seed mean beats control by ≥0.01 and seeds don't straddle zero.

## Screen tier (Screen10M20M, 20M tokens, ~4880 steps)

| Name | seed 42 | seed 43 | seed 44 | mean | std | Δ vs ctrl | run dir | verdict |
|---|---|---|---|---|---|---|---|---|
| control (RMSNorm) | 4.7984 | — | — | — | — | 0 | `s_ctrl_full` | baseline |
| ReparamGain | — | — | — | — | — | — | — | pending |
| RMSBias | — | — | — | — | — | — | — | pending |
| GlobalTemp | — | — | — | — | — | — | — | pending |
| PartialNormMix | — | — | — | — | — | — | — | pending |
| PartialNormVector | — | — | — | — | — | — | — | pending |
| GroupRMS | — | — | — | — | — | — | — | pending |
| StopGradRMS | — | — | — | — | — | — | — | pending |
| AsymGain | — | — | — | — | — | — | — | pending |
| DynTanh | — | — | — | — | — | — | — | pending |
| CenterMix | — | — | — | — | — | — | — | pending |

(add Batch 1–4 rows as they run; norm-zoo sweep goes in its own table below)

## Norm-zoo sweep (existing norm_type options)

| norm_type | mean (3-seed) | std | Δ vs ctrl | note |
|---|---|---|---|---|
| (pnorm1.5, clipnorm3, channelscale, manhattan, …) | | | | |

## Tiny tier (Tiny1M3M — screening only)

| Name | val_loss | Δ vs tiny ctrl | run dir | note |
|---|---|---|---|---|
| (prior tiny1m norm4/5/6 results can be backfilled here for reference) | | | | |

## Key A/Bs

- PartialNormMix (scalar N4) vs PartialNormVector (per-channel N8): scalar or vector?
- CenterMix (N17) landing μ vs existing `layernorm`: does partial centering beat full?
- StopGradRMS (N10): does gradient-through-rms matter, or only the forward scaling?

## Notes / surprises

- (record seed spread — if std > the Δ, it's noise. step-0 mismatches, dead gradients.)
