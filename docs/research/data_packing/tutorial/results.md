# Data / sequence-packing ablations — results

# Pending

Filled as runs land. Control = `Screen10M20MConfig` (val_loss 4.7984, `s_ctrl_full`).
"Live" = 3-seed mean beats control by ≥0.01 and seeds don't straddle zero.

## Tiny tier (Tiny1M3M — screening only)

| Name | val_loss | Δ vs tiny ctrl | run dir | note |
|---|---|---|---|---|
| (pending) | — | — | — | pending |

## Screen tier (Screen10M20M, 20M tokens, ~4880 steps)

| Name | seed 42 | seed 43 | seed 44 | mean | std | Δ vs ctrl | run dir | verdict |
|---|---|---|---|---|---|---|---|---|
| (pending) | — | — | — | — | — | — | — | pending |

## Key A/Bs

- D1 vs D2: is the data-axis win the boundary info, or the cross-doc masking?
- D3 sweep: is there a useful seq_length optimum (vs 2048 default) at this scale?

## Notes / surprises

- (record seed spread — if std > the Δ, it's noise. step-0 mismatches, dead gradients.)
