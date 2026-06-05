# Output-head ablations — results

Filled as runs land. Control = clean `Screen10M20MConfig`. **All val_loss is plain
cross-entropy** (aux terms are train-only, never in the reported number). "Live" =
3-seed mean beats control by ≥0.01 and seeds don't straddle zero.

## Screen tier (Screen10M20M, 20M tokens, ~4880 steps)

| Name | seed 42 | seed 43 | seed 44 | mean | std | Δ vs ctrl | run dir | verdict |
|---|---|---|---|---|---|---|---|---|
| control | 4.7984 | — | — | — | — | 0 | `s_ctrl_full` | baseline |
| ZLoss | — | — | — | — | — | — | — | pending |
| LabelSmooth | — | — | — | — | — | — | — | pending |
| ConfPenalty | — | — | — | — | — | — | — | pending |
| OutputTemp | — | — | — | — | — | — | — | pending |
| VocabBias | — | — | — | — | — | — | — | pending |
| LogitSoftcap (existing, swept) | — | — | — | — | — | — | — | pending |
| UntieHead (diagnostic) | — | — | — | — | — | — | — | pending |

## Tiny tier (Tiny1M3M — screening only, not a claim)

| Name | val_loss | Δ vs tiny ctrl | run dir | note |
|---|---|---|---|---|
| (add tiny screens here) | | | | |

## Key questions

- ZLoss: does pulling logit norm down stabilize / let LR run hotter, for a CE win?
- VocabBias: how much CE is just an unmodeled unigram prior at this scale?
- UntieHead: is weight-tying load-bearing, or a free param saving? (not budget-fair)

## Notes / surprises

- Confirm aux terms (OH1–OH3) never leak into eval CE — a "win" that's just a
  smaller reported loss from smoothing is a measurement bug, not a result.
- (record step-0 mismatches, dead-gradient gotchas)
