# Output-head ablations — tutorial (SCAFFOLD — fill after experiments)

> **Status: not written.** Placeholder. Do not publish until
> [results.md](results.md) has a clear story. Follow the house style of
> [../../../tutorials/qk_gain/README.md](../../../tutorials/qk_gain/README.md):
> problem → one mechanism → fair baseline → the number → what it means.

## Title

<!-- hook, e.g. "The whole model funnels through one matmul and one loss" -->

## The problem we need to solve

<!-- the head maps the residual to vocab logits and CE scores them raw.
     is the raw logit / plain CE the right thing to optimize? -->

## The mechanism

<!-- the winning / most instructive lever (z-loss? vocab bias?). formula in ```text```. -->

## The fair baseline

<!-- clean Screen10M20M = 4.7984. CRITICAL: reported val_loss is plain CE;
     aux terms are train-only. identity-init so step-0 == baseline. -->

## The result

<!-- 3-seed plain-CE mean ± std vs control from results.md; figure in images/ -->

## What it means

<!-- stabilizer vs prior vs regularizer — which knob on the output actually pays,
     and the measurement trap (don't report a smoothed loss as a win) -->

## Reproduce

```bash
python train_llm.py --config <winning-config-key> --seed 42
```

## Assets to produce

- [ ] figure(s) in [images/](images/)
- [ ] English PDF (match qk_gain packaging)
