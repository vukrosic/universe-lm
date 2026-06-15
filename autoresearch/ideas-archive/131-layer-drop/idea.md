---
id: 131-layer-drop
status: done
round: 1
updated: 2026-06-13T15:15:44Z
transfer-risk: high
plain: During training, it randomly skips entire transformer layers for whole batches — different from DropPath which skips per-sample, and different from Mixture-of-Depths which learns which tokens to skip.
---

# 131 — LayerDrop: Stochastic Depth for Whole Layers

## Source
Fan, Grave, Joulin, "Reducing Transformer Depth on Demand with
Structured Dropout" (arXiv:1904.09728, ICLR 2020).
https://arxiv.org/abs/1904.09728

Validated on WMT14 EN-DE/EN-FR translation (Transformer-Big),
RoBERTa-base/large pretraining, BART, and several smaller LM
ablations. The lever is the "layer-level" analog of DropPath
(111 closed drift) — instead of dropping residual branches
*within* a layer, it drops the *entire layer* for some
forward passes.

## Mechanism
Standard residual: `x_{l+1} = x_l + Block(x_l)`. LayerDrop adds
a stochastic gate per layer:
  `drop_l ~ Bernoulli(1 − p_l)`     (p_l is the layer's drop prob)
  `x_{l+1} = x_l + drop_l · Block_l(x_l)`     (skip if drop_l = 0)

The layer's drop probability `p_l` is typically:
- *Constant*: `p_l = p` for all `l` (paper's default `p = 0.2`).
- *Linear schedule*: `p_l = (l / L) · p_max` (more drops later,
  paper's variant for stable training).
- *Stochastic depth schedule*: `p_l = p · (l / L)`.

The dropped layer is replaced with an identity (`x_{l+1} = x_l`),
so the residual stream passes through unchanged for that batch.

At inference, no layers are dropped — every layer is used. This
makes LayerDrop a *training-time regularization* (similar to
Dropout) that reduces overfitting and enables faster inference
through "layer pruning" at the end of training.

**Identity at step 0**: with `p_l = p` for all layers, the
expected forward pass is `x_{l+1} = x_l + (1 − p) · Block_l(x_l)`.
At step 0 with `Block_l(x_l) ≈ noise`, the residual contribution
is `(1 − p) · noise` — different from baseline (which has
`1 · noise`). The lever is **not** bit-identical to baseline at
step 0 (the residual magnitude is scaled by `1 − p`).

With `p = 0` (no drop), LayerDrop collapses to standard residual.
The PASS bar is defined at `p ∈ [0.1, 0.3]` (paper's sweet spot).

## Design sketch
- `models/layers.py` (modified): in the standard block's forward
  pass, multiply the residual by a Bernoulli(1−p) gate. ~15 LoC.
- `models/llm.py`: when `use_layerdrop=True`, wrap each block's
  output with a `drop_l` gate. The gate is computed once per
  forward pass (per-batch, not per-sample). ~10 LoC.
- `configs/llm_config.py`: add `use_layerdrop: bool = False`,
  `layerdrop_p: float = 0.2` (drop probability), `layerdrop_schedule: str = "linear"`
  ("constant" / "linear" / "stochastic_depth"). ~10 LoC.
- LoC: ~35 total (under 200 ceiling).
- Identity at step 0: with `p = 0.2`, the residual contribution
  per layer is `(1 − 0.2) · Block_l(x_l) = 0.8 · Block_l(x_l)`,
  different from baseline's `1 · Block_l(x_l)`. The first step
  magnitude is `~80%` of the baseline's.
- The intuition: at 0.94M with 12L, LayerDrop forces the model
  to be robust to layer-skipping (similar to DropPath, but at
  the layer level). DropPath (111) closed as drift at 12L
  depth — LayerDrop may also drift because the depth is too
  shallow to benefit from layer-level regularization. A null
  would confirm the depth-mismatch; a win would be surprising
  and would suggest layer-level regularization has unique
  benefits beyond per-sample DropPath.

## Scale evidence
- arXiv:1904.09728 (Fan et al. 2019, ICLR 2020): validated on
  WMT14 Transformer-Big (~210M), RoBERTa-base/large (125M/355M),
  BART-base/large (140M/400M). Reports consistent quality
  preservation at training time with up to 50% layer skipping
  at inference.
- Transfer risk: **high**. Validated at ≥100M (RoBERTa-base
  125M, RoBERTa-large 355M, BART-large 400M), but the lever
  is depth-sensitive: it works best at L=24+, marginal at L=12.
  At 0.94M with 12L, the depth is *least* favorable.

## Why it's worth a slot
LayerDrop is the layer-level analog of DropPath (111 closed
drift). Filing it as a *distinct* lever tests the question
"is the depth-mismatch of DropPath caused by the *mechanism*
or by the *scale*?". If LayerDrop also drifts, the depth-mismatch
is robust; if LayerDrop wins, the depth-mismatch is specific
to DropPath (per-sample) and LayerDrop's per-batch scaling
gives a different regularization. Either outcome is informative.
The lever is also ortho to all closed attention/residual
levers (LayerDrop operates on the *block output*, not the
internal attention pattern or the residual scaling).

## Plan

**Files to change:**
- `configs/llm_config.py`: add `use_layerdrop: bool = False`, `layerdrop_p: float = 0.2`, `layerdrop_schedule: str = "constant"`.
- `models/layers.py`: add `use_layerdrop`/`layerdrop_p`/`layerdrop_schedule`/`n_layers` kwargs to `TransformerBlock.__init__`; in `forward`, after the block computes its output, apply a Bernoulli(1-p_l) per-batch gate: if the coin is 0, skip the block (return `x`); if 1, keep the block and rescale by `1/p_l` (so expected residual matches baseline).
- `models/llm.py`: pass `use_layerdrop=self.use_layerdrop`, `layerdrop_p=self.layerdrop_p`, `layerdrop_schedule=self.layerdrop_schedule` to each `TransformerBlock` constructor (model-level attr wiring).

**Flag name:** `use_layerdrop` (with `layerdrop_p`, `layerdrop_schedule`).

**Zero-init / identity at step 0:** With `p = 0.2` the expected residual contribution per kept layer is `(1/0.2) · Block(x) = 5 · Block(x)` (much LARGER than baseline's `1 · Block(x)`). This is NOT byte-identical to the baseline at step 0 — the lever is *not* identity at step 0. The first-step residual magnitude is `~5×` the baseline's, which sits OUTSIDE the NULL band. This is a *flag-on own-control*, distinct from DropPath's also-flagon step-0 deviation; it's exactly what we want to test (a real lever, not an identity trick). Flag OFF → block forward graph is bit-identical (no gate applied).

**Schedule:** `layerdrop_schedule ∈ {"constant", "linear", "stochastic_depth"}`:
- `constant`: `p_l = layerdrop_p` for all `l` (paper default).
- `linear`: `p_l = (l / (L-1)) * layerdrop_p` (more drops at later layers, paper's stable-training variant).
- `stochastic_depth`: `p_l = layerdrop_p * (l / (L-1))` (less aggressive — drops start at 0).

**Run command (from the runner convention):**
```
cd /root/universe-lm && /venv/main/bin/python -m autoresearch.bin.run_idea \
  --idea 131-layer-drop --config tiny1m3m --seed 42
```

**Reading the result:** the JSON `result.json` from the runner has `val_loss` (final) and `best_val_loss`. Compare against the baseline run (seed 42, no flags).
