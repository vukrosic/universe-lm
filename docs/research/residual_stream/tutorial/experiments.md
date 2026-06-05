# Residual-stream ablations — experiment manifest

Run status for [../plan.md](../plan.md). `status` ∈ {TODO, wired, tiny-done,
screen-running, screen-done, dropped}. Control = `Screen10M20MConfig` (4.7984, `s_ctrl_full`).

## Already in repo (reference / comparison only — not new work)

| Flag | What | status |
|---|---|---|
| `use_layerscale` | per-channel branch scale | done (`s_layerscale`) |
| `use_embed_residual` | x0 per-dim mix | done (`s_embresid`) |

## Batch 1 — cheap core

| # | Name | Config class | Flag | step-0==base | params/block | status |
|---|---|---|---|---|---|---|
| R1 | ReZero | `Screen10M20MReZeroConfig` | `use_rezero` | yes (α=0) | 2 | TODO |
| R2 | ResidMix | `Screen10M20MResidMixConfig` | `use_resid_mix` | yes (a=b=1) | 4 | TODO |
| R3 | HighwayGate | `Screen10M20MHighwayGateConfig` | `use_highway_gate` | ~yes (σ≈1) | 2 | TODO |
| R4 | BranchGainHead | `Screen10M20MBranchGainHeadConfig` | `use_branch_gain_head` | yes (g=1) | n_heads | TODO |

## Batch 2 — init / schedule

| # | Name | Config class | Flag | step-0==base | status |
|---|---|---|---|---|---|
| R5 | DepthScaledInit | `Screen10M20MDepthScaledInitConfig` | `use_depth_scaled_init` | no — own control | TODO |
| R6 | FrozenLayerScale | `Screen10M20MFrozenLayerScaleConfig` | `frozen_layerscale` | no — own control | TODO |
| R7 | StochDepth | `Screen10M20MStochDepthConfig` | `stoch_depth_p` | yes (eval) | TODO |
| R8 | BranchDropout | `Screen10M20MBranchDropoutConfig` | `branch_dropout_p` | yes (eval) | TODO |

## Batch 3 — gated on Batch 1

| # | Name | Flag | A/B against | status |
|---|---|---|---|---|
| R9 | ReZeroPerChannel | `use_rezero_channel` | `use_layerscale` | TODO |
| R10 | ResidMixVector | `use_resid_mix_vector` | R2 ResidMix | TODO |

## Per-experiment checklist (tick before screen-done)

- [ ] flag guarded `if self.use_<x>:`, baseline path untouched
- [ ] step-0 val_loss matches clean control (or own control if non-identity)
- [ ] new params confirmed getting gradient (AdamW)
- [ ] tiny run (kill if washing)
- [ ] screen 3-seed (42/43/44), mean + std in results.md
- [ ] metrics.json committed, evidence index regenerated
