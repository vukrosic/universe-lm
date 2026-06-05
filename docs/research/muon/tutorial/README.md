# Muon ablations — tutorial (SCAFFOLD — fill after experiments)

> **Status: not written.** Placeholder. Do not publish until
> [results.md](results.md) has a clear 3-seed + wall-clock story. House style of
> [../../../tutorials/qk_gain/README.md](../../../tutorials/qk_gain/README.md):
> problem → one mechanism → fair baseline → the number → what it means.

## Title

<!-- hook, e.g. "How much of Muon is the orthogonalization?" -->

## The problem we need to solve

<!-- Muon orthogonalizes the momentum before each step. is that expensive iteration
     load-bearing, or can we cut it? and is the routing/scaling tuned? -->

## The mechanism

<!-- the part that mattered (or didn't): orthogonalization / scaling / routing.
     show the update rule in ```text```. -->

## The fair baseline

<!-- default Muon = 4.7984. CRITICAL: optimizer changes need an LR re-sweep —
     compare best-LR vs best-LR, not at a fixed LR. 3-seed. -->

## The result

<!-- 3-seed mean ± std AND wall-clock vs default Muon, from results.md; figure in images/ -->

## What it means

<!-- which sub-part carries Muon's edge, and the compute lever (fewer NS steps?)
     that a small-compute lab should actually adopt -->

## Reproduce

```bash
python train_llm.py --config <winning-config-key> --muon_lr <best> --seed 42
```

## Assets to produce

- [ ] figure(s) in [images/](images/) — include a loss-vs-wallclock plot
- [ ] English PDF (match qk_gain packaging)
