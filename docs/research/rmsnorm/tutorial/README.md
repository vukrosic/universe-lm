# RMSNorm ablations — tutorial (SCAFFOLD — fill after experiments)

> **Status: not written.** Placeholder. Do not publish until
> [results.md](results.md) has a clear 3-seed story. This folder **extends** the
> published normalization tutorial — link back to it and follow its house style:
> [../../../tutorials/normalization/README.md](../../../tutorials/normalization/README.md)
> ("a normalization trick is only useful if it survives a fair baseline").

## Title

<!-- hook, e.g. "Seven ways to change RMSNorm, and which one survived 3 seeds" -->

## The problem we need to solve

<!-- y = g·x/rms(x) is the simplest possible norm. what's it leaving on the table? -->

## The mechanism

<!-- the winning / most instructive tweak. formula in ```text```. -->

## The fair baseline

<!-- clean RMSNorm = 4.7984; norm is seed-noisy so 3-seed mean ± std is mandatory.
     identity-init so step-0 == baseline. -->

## The result

<!-- 3-seed mean ± std vs control from results.md; show the seed spread, figure in images/ -->

## What it means

<!-- scalar vs vector, centering amount, gradient-through-rms — which axis pays,
     and the lesson that single-seed norm wins are noise -->

## Reproduce

```bash
python train_llm.py --config <winning-config-key> --seed 42
```

## Assets to produce

- [ ] figure(s) in [images/](images/) — include the seed-spread plot
- [ ] English PDF (match qk_gain / normalization packaging)
