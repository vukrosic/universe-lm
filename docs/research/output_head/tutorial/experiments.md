# Output-head ablations — experiment manifest

Run status for [../plan.md](../plan.md). `status` ∈ {TODO, wired, tiny-done,
screen-running, screen-done, dropped}. Control = `Screen10M20MConfig` (4.7984, `s_ctrl_full`).

⚠️ Reported val_loss is **plain CE**. Train-only aux terms (OH1–OH3) must NOT leak
into eval — see plan's reporting rule.

## Already in repo (reference / comparison only)

| Existing | What | status |
|---|---|---|
| `logit_softcap` (#71) | Gemma logit cap | done (`s_vqgain_swa_highrope_softcap`, single-seed) |
| `output_adapter` | rank-32 head adapter | done (`s_outadapter`) |

## Batch 1 — loss-side terms (train-only)

| # | Name | Config class | Knob | edit site | step-0==base | status |
|---|---|---|---|---|---|---|
| OH1 | ZLoss | `Screen10M20MZLossConfig` | `z_loss_lambda` | trainer.py | yes (λ=0) | TODO |
| OH2 | LabelSmooth | `Screen10M20MLabelSmoothConfig` | `label_smoothing` | trainer.py | yes (ε=0) | TODO |
| OH3 | ConfPenalty | `Screen10M20MConfPenaltyConfig` | `conf_penalty_beta` | trainer.py | yes (β=0) | TODO |

## Batch 2 — logit ops (flow into eval)

| # | Name | Config class | Flag | edit site | step-0==base | params | status |
|---|---|---|---|---|---|---|---|
| OH4 | OutputTemp | `Screen10M20MOutputTempConfig` | `use_output_temp` | llm.py | yes (τ=1) | 1 | TODO |
| OH5 | VocabBias | `Screen10M20MVocabBiasConfig` | `use_vocab_bias` | llm.py | yes (b=0) | vocab | TODO |

## Batch 3 — head structure (gated)

| # | Name | Knob | note | status |
|---|---|---|---|---|
| OH6 | LogitSoftcapSweep | `logit_softcap` (existing) | sweep c∈{10,15,30}, 3-seed | TODO |
| OH7 | UntieHead | `tie_output_head=False` | costs params, diagnostic only | TODO |
| OH8 | NonlinearDecode | → tied-output-mlp plan | don't duplicate | see other plan |

## Per-experiment checklist (tick before screen-done)

- [ ] knob guarded; baseline path untouched
- [ ] aux terms train-only; eval CE stays plain (OH1–OH3)
- [ ] step-0 val_loss matches clean control
- [ ] new params confirmed getting gradient (AdamW)
- [ ] tiny run (kill if washing)
- [ ] screen 3-seed (42/43/44), plain-CE mean + std in results.md
- [ ] metrics.json committed, evidence index regenerated
