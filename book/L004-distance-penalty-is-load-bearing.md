# L004 — A growing distance penalty is load-bearing

**Statement.** Adding a growing attention-distance penalty (ALiBi) versus no positional
penalty improves val loss by ≈ 0.155 paired — the single largest lever measured in this lab.

**Status.** L — strong. The dominant effect in the size hierarchy.

**Scope.** tiny1m3m, seq 512. The *magnitude* is scale- and context-length-specific; a
longer context or a model that recovers position another way (e.g. strong RoPE) may shrink it.

## Evidence
- ALiBi deep-dive: paired ALiBi-vs-none Δ ≈ −0.155 — roughly 10× the L001 noise floor and
  an order of magnitude larger than any single architecture lever found since.
- Everything downstream (DeepNet-α, poly-ALiBi, the combo, the optimizer stack) is a
  sub-0.04 refinement *on top of* this base — the record board is built on the ALiBi recipe.

## Falsifier
A regime (longer context, larger scale, or a competing positional scheme) where removing the
growing penalty costs < 0.02 — would bound this law to short-context tiny scale.

## Why it matters
Anchors the effect-size hierarchy: get the positional prior right *first*; it dwarfs every
shape/curvature/optimizer refinement ([[L002]], [[L003]], [[L007]]). This is the load-bearing
wall of the whole champion lineage.

Links: [[L002]], [[L003]], [[L007]].
