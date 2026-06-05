# Attention output (W_O) — experiment manifest

Run status for [../plan.md](../plan.md). `status` ∈ {TODO, wired, tiny-done,
screen-running, screen-done, dropped}. Control = clean `Screen10M20MConfig`
(4.7984, `s_ctrl_full`).

## Batch 1 — cross-head mixing on the output (headline C1 lever)

| # | Name | Config class | Flag | step-0==base | status |
|---|---|---|---|---|---|
| O1 | TalkingHeadsOut | `Screen10M20MTalkingHeadsOutConfig` | `use_talking_heads_out` | yes (M=I) | TODO |
| O2 | OutputHeadGate | `Screen10M20MOutputHeadGateConfig` | `use_output_head_gate` | yes (g=1) | TODO |

## Batch 2 — post-softmax nonlinearity / norm

| # | Name | Config class | Knob | step-0==base | status |
|---|---|---|---|---|---|
| O3 | OutputRMS | `Screen10M20MOutputRMSConfig` | `output_rms_norm` | yes | TODO |
| O4 | OutputTanh | `Screen10M20MOutputTanhConfig` | `output_tanh_alpha` | yes (α=1) | TODO |

## Batch 3 — value-side variants

| # | Name | Config class | Knob | step-0==base | status |
|---|---|---|---|---|---|
| O5 | OutputSoftplus | `Screen10M20MOutputSoftplusConfig` | `output_softplus` | no — own ctrl | TODO |
| O6 | OutputBias | `Screen10M20MOutputBiasConfig` | `output_bias` | yes (b=0) | TODO |

## Per-experiment checklist (tick before screen-done)

- [ ] step-0 val_loss matches control (or own control if non-identity)
- [ ] new params getting gradient (Muon for 2D, AdamW for 1D)
- [ ] tiny run (kill if washing)
- [ ] screen **3-seed** (42/43/44), mean + std in results.md
- [ ] metrics.json committed, evidence index regenerated
