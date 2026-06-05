# Query ablations — tutorial (SCAFFOLD — fill after experiments)

> **Status: not written.** This is a placeholder. Do not publish until
> [results.md](results.md) has a clear story (a live win or an instructive null).
> Follow the house style of
> [../../../tutorials/qk_gain/README.md](../../../tutorials/qk_gain/README.md):
> problem → one mechanism → fair baseline → the number → what it means.

## Title

<!-- one concrete hook, e.g. "Does the query need more than one scalar?" -->

## The problem we need to solve

<!-- what the query path already does, why one scalar (q_gain) might not be enough -->

## The mechanism

<!-- the winning lever (or the most instructive one). diagram in ```text```. -->

## The fair baseline

<!-- clean Screen10M20M control = 4.7984; identity-init so step-0 == baseline -->

## The result

<!-- 3-seed mean ± std vs control, pulled from results.md. figure in images/ -->

## What it means

<!-- attribution: which axis (positional / similarity / capacity) actually moved loss -->

## Reproduce

```bash
python train_llm.py --config <winning-config-key> --seed 42
```

## Assets to produce

- [ ] figure(s) in [images/](images/)
- [ ] English PDF (match qk_gain packaging)
- [ ] X / video companion (optional)
