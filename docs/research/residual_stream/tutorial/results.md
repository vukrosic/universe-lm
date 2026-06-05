# Residual-stream ablations — results

Filled as runs land. Control = clean `Screen10M20MConfig`. "Live" = 3-seed mean
beats control by ≥0.01 and seeds don't straddle zero.

## Screen tier (Screen10M20M, 20M tokens, ~4880 steps)

| Name | seed 42 | seed 43 | seed 44 | mean | std | Δ vs ctrl | run dir | verdict |
|---|---|---|---|---|---|---|---|---|
| control | 4.7984 | — | — | — | — | 0 | `s_ctrl_full` | baseline |
| LayerScale (existing) | — | — | — | — | — | — | `s_layerscale` | (backfill from prior run) |
| ReZero | — | — | — | — | — | — | — | pending |
| ResidMix | — | — | — | — | — | — | — | pending |
| HighwayGate | — | — | — | — | — | — | — | pending |
| BranchGainHead | — | — | — | — | — | — | — | pending |
| DepthScaledInit | — | — | — | — | — | — | — | pending |
| FrozenLayerScale | — | — | — | — | — | — | — | pending |
| StochDepth | — | — | — | — | — | — | — | pending |
| BranchDropout | — | — | — | — | — | — | — | pending |

## Tiny tier (Tiny1M3M — screening only, not a claim)

| Name | val_loss | Δ vs tiny ctrl | run dir | note |
|---|---|---|---|---|
| (add tiny screens here) | | | | |

## Key A/Bs

- ReZero (scalar gate) vs LayerScale (per-channel gate): scalar or vector?
- DepthScaledInit / FrozenLayerScale (fixed) vs ReZero / ResidMix (learned): init or learning?

## Notes / surprises

- (record step-0 mismatches, dead-gradient gotchas, anything that broke identity-init)
