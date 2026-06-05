# Data / sequence-packing ablations — experiment manifest

Run status for [../plan.md](../plan.md). `status` ∈ {TODO, wired, tiny-done,
screen-running, screen-done, dropped}. Control = `Screen10M20MConfig`
(val_loss **4.7984**, `s_ctrl_full`). **3-seed mandatory** on the screen tier.

## Batch 1 — packing

| # | Name | Config class | knob | step-0==base | status |
|---|---|---|---|---|---|
| D1 | DocPack | `Tiny1M3MDocPackConfig` | `use_doc_pack=True` | yes (loader flag off by default) | TODO |
| D2 | NoCrossDoc | (pending — follow-up) | (pending) | (pending) | TODO |

## Batch 2 — length / curriculum

| # | Name | Config class | knob | step-0==base | status |
|---|---|---|---|---|---|
| D3 | SeqLenSweep | (pending) | (pending) | yes | TODO |
| D4 | ShortSeqFirst | (pending) | (pending) | yes | TODO |

## Per-experiment checklist (tick before screen-done)

- [ ] step-0 val_loss matches control (loader lever ⇒ trivial)
- [ ] tiny run (kill if washing)
- [ ] screen **3-seed** (42/43/44), mean + std in results.md
- [ ] metrics.json committed, evidence index regenerated
