# Query ablations — results

Filled by the implementing AI as runs land. Control = clean `Screen10M20MConfig`.
"Live" = 3-seed mean beats control by ≥0.01 and seeds don't straddle zero.

## Screen tier (Screen10M20M, 20M tokens, ~4880 steps)

| Name | seed 42 | seed 43 | seed 44 | mean | std | Δ vs ctrl | run dir | commit | verdict |
|---|---|---|---|---|---|---|---|---|---|
| control | 4.7984 | — | — | — | — | 0 | `s_ctrl_full` | `4be65bb6` | baseline |
| AlibiBias | — | — | — | — | — | — | — | — | pending |
| QTempToken | — | — | — | — | — | — | — | — | pending |
| CosineAttn | — | — | — | — | — | — | — | — | pending |
| QKBilinear | — | — | — | — | — | — | — | — | pending |
| TalkingHeadsQ | — | — | — | — | — | — | — | — | pending |
| PerHeadRopeBase | — | — | — | — | — | — | — | — | pending |
| PartialRotary | — | — | — | — | — | — | — | — | pending |

## Tiny tier (Tiny1M3M, screening only — not a claim)

| Name | val_loss | Δ vs tiny ctrl | run dir | note |
|---|---|---|---|---|
| (add tiny screens here) | | | | |

## Notes / surprises

- (record step-0 mismatches, gradient-flow gotchas, anything that broke the identity-init assumption)
