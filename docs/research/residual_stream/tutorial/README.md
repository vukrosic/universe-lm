# Residual-stream ablations — tutorial (SCAFFOLD — fill after experiments)

> **Status: not written.** Placeholder. Do not publish until
> [results.md](results.md) has a clear story. Follow the house style of
> [../../../tutorials/qk_gain/README.md](../../../tutorials/qk_gain/README.md):
> problem → one mechanism → fair baseline → the number → what it means.

## Title

<!-- hook, e.g. "Does the residual stream want a scalar or a vector gate?" -->

## The problem we need to solve

<!-- the block adds each sublayer straight back: x = x + f(x). is that the right add? -->

## The mechanism

<!-- the winning / most instructive lever. diagram of x = a·x + g⊙f(x) in ```text``` -->

## The fair baseline

<!-- clean Screen10M20M control = 4.7984; identity-init so step-0 == baseline -->

## The result

<!-- 3-seed mean ± std vs control from results.md; figure in images/ -->

## What it means

<!-- the real question: scalar vs vector gate, and init vs learning -->

## Reproduce

```bash
python train_llm.py --config <winning-config-key> --seed 42
```

## Assets to produce

- [ ] figure(s) in [images/](images/)
- [ ] English PDF (match qk_gain packaging)
