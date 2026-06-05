# Muon ablations — results

Filled as runs land. Control = `Screen10M20MConfig` default Muon. Report **3-seed
mean + std** AND **wall-clock** at the per-arm best LR. "Live" = mean beats control by
≥0.01 and seeds don't straddle zero. A loss-neutral arm that's faster is still a win.

## Screen tier (Screen10M20M, 20M tokens, ~4880 steps)

| Name | best muon_lr | s42 | s43 | s44 | mean | std | Δ vs ctrl | min | run dir | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| control (Muon, 5 steps) | 0.024 | 4.7984 | — | — | — | — | 0 | — | `s_ctrl_full` | baseline |
| NSSteps=1 | — | — | — | — | — | — | — | — | — | speed probe |
| NSSteps=2 | — | — | — | — | — | — | — | — | — | speed probe |
| NSSteps=3 | — | — | — | — | — | — | — | — | — | speed probe |
| NoOrtho (mom-SGD) | — | — | — | — | — | — | — | — | — | headline A/B |
| NSCoeffs (quintic) | — | — | — | — | — | — | — | — | — | pending |
| NoShapeScale | — | — | — | — | — | — | — | — | — | pending |
| SpectralScale | — | — | — | — | — | — | — | — | — | pending |
| MomentumSweep | — | — | — | — | — | — | — | — | — | pending |
| EmbedToMuon | — | — | — | — | — | — | — | — | — | pending |
| LRRatioSweep | — | — | — | — | — | — | — | — | — | pending |
| LazyOrtho (N=?) | — | — | — | — | — | — | — | — | — | speed probe |

(`min` = wall-clock minutes from metrics.json `total_time_minutes`.)

## Tiny tier (Tiny1M3M — screening only)

| Name | val_loss | min | Δ vs tiny ctrl | run dir |
|---|---|---|---|---|
| (add tiny screens here) | | | | |

## Headline questions

- **NoOrtho:** strip the polar-express — how much of Muon's edge is the
  orthogonalization vs just momentum + the LR/scaling? The whole "why Muon" story.
- **NSSteps:** can we run 1–3 iters instead of 5 with no loss? direct compute saving.
- **NoShapeScale / LRRatio:** is the current tuning (0.024/0.006, fanout/fanin) actually optimal?

## Notes / surprises

- Always note whether a regression is real or just LR mistuning (did you sweep?).
- (record divergences, NaNs in low-step orthogonalization, fp32-vs-bf16 deltas)
