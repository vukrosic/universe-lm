---
id: 127-grad-centralization
status: done
round: 1
updated: 2026-06-13T14:52:43Z
transfer-risk: med
plain: It subtracts the mean from each gradient matrix before sending it to the optimizer — a one-line transformation that consistently gives modest wins across many network types.
---

# 127 — Gradient Centralization (GC)

## Source
Yong, Fortuin, Morariu, Salzmann, Ni, "Transforming Gradient
Centralization to Momentum" (arXiv:2010.01461, October 2020) and
the original "Gradient Centralization: A New Optimization Technique
for Deep Neural Networks" (arXiv:2004.01461, ICONIP 2020).
https://arxiv.org/abs/2004.01461

Validated on CIFAR-10/100 ResNet, ImageNet ResNet-50/VGG-16/16,
ShuffleNetV2, MobileNetV2, and several transformer/seq2seq
benchmarks. The lever is one of the cleanest "free wins" in the
deep learning optimization literature — a single transformation
on the gradient before it enters the optimizer.

## Mechanism
Standard optimization: `w ← w − lr · optimizer(g_t)`. GC modifies
`g_t` before the optimizer sees it:
  `μ = mean(g_t, dim=axis)`     (mean across the *output* axis)
  `g_t_centralized = g_t − μ`   (subtract the mean)

For a 2-D weight matrix `W ∈ R^{n×m}`, the mean `μ ∈ R^m` is
subtracted along the output axis (axis=1), giving each output
neuron zero-mean input gradient. For a 4-D conv weight, GC is
applied per-filter (subtracting the mean across the spatial
dimensions).

The intuition: zero-mean gradients prevent the *output bias
shift* that occurs when a single neuron's gradients are biased.
In practice, this is a regularization trick that constrains the
gradient to lie on a sphere (subtract the mean ⇒ unit centroid),
which often correlates with better generalization.

Mathematically, GC can be expressed as a *linear operator* on
the gradient:
  `g_centralized = (I − (1/m)·1·1^T) · g`     (subtract mean)

**Identity at step 0**: with `g_0` being the gradient at initialization,
GC transforms `g_0 → g_0 − mean(g_0)`. The optimizer then sees
the centered gradient instead of the raw gradient. This is **not**
bit-identical to AdamW at step 0 (the optimizer sees a different
gradient), but the *expected magnitude* is preserved (the mean
subtraction only removes the constant component, not the variance).
The first step is `O(1/n)` different from AdamW's first step (where
`n` is the number of elements per output axis).

## Design sketch
- `optimizers/grad_centralization.py` (new): `GCAdamW` — wrapper
  that adds a GC transform before AdamW's step. The transform
  is a single `g -= g.mean(dim=axis)` per param (per-axis mean
  subtraction). ~30 LoC.
- `training/trainer.py`: when `use_gc=True`, wrap the AdamW
  optimizer with GC. The 2-D slot's Muon path can also apply
  GC to its gradient (paper recommends applying GC before any
  optimizer). ~10 LoC.
- `configs/llm_config.py`: add `use_gc: bool = False`,
  `gc_dim: int = 1` (axis to mean-subtract). ~5 LoC.
- LoC: ~45 total (under 200 ceiling).
- Identity at step 0: with `dim=1`, the first step has
  `g_centralized = g_0 − mean(g_0, dim=1)`. The mean-subtracted
  gradient has zero mean per output neuron. Different from
  AdamW's raw-gradient first step by `O(1/n)` in magnitude.
- The intuition: at 0.94M, the per-output-neuron gradient mean
  can drift non-trivially between steps (the gradient at one
  output is correlated with adjacent outputs). GC's centering
  removes this drift. A null would say "at 0.94M the gradient
  mean drift is benign"; a win would say "centering the gradient
  before AdamW gives a more uniform step direction across
  neurons".

## Scale evidence
- arXiv:2004.01461 (Yong et al. 2020, ICONIP): CIFAR-10/100
  ResNet + ShuffleNetV2 + MobileNetV2, all show +0.3-1.0% top-1.
- arXiv:2010.01461 (follow-up): extends to "Momentum Centralization"
  which centers the *momentum* instead of the gradient; gives
  larger gains.
- Reproductions in fastai, PyTorch Lightning, and timm.
- Transfer risk: **med**. Validated at image-classification
  scale (≤100M params in main experiments, though the paper
  also reports ViT-B/16 which is 86M). The mechanism is
  scale-free (any deep net has gradient drift across neurons).
  At 0.94M, the per-neuron gradient drift should be similar
  to the larger-scale reports.

## Why it's worth a slot
GC is one of the cleanest "free wins" in the optimization
literature — a one-line gradient transform with consistent
+0.3-1% gains across many network architectures. It is
ortho to every closed optimizer lever (031-040, 001-006)
because it operates on the *gradient before the optimizer
sees it*, not on the optimizer's update rule. The lever is
*compositional* with every other optimizer — GC + AdamW,
GC + Muon, GC + SAM, etc. are all valid combinations. A win
would say "centering the gradient is load-bearing even at
0.94M and should be applied to every optimizer we use"; a
null would say "at 0.94M the gradient centering is benign
and the additional compute is wasted". The lever's simplicity
is its main strength — even a tiny win compounds with every
other lever.

## Plan
- **Flag**: `use_gc: bool = False` (default off → baseline path
  bit-identical). Auxiliary: `gc_axis: int = 1` (mean-subtract
  along dim 1, the output axis for `W ∈ R^{n×m}`). For 4-D conv
  weights we apply per-filter mean subtraction (axis over the
  spatial dims).
- **Files**:
  - `optimizers/grad_centralization.py` (new): `GCAdamW` —
    thin subclass of `torch.optim.AdamW`. Pre-step hook
    `_centralize_grads` mean-subtracts each `param.grad` along
    the chosen axis (per-row for 2-D weights, per-filter for
    4-D conv), then calls `super().step()`. The per-parameter
    `(m, v)` state is untouched; only the gradient input is
    centered.
  - `configs/llm_config.py`: add `use_gc`, `gc_axis`, plus a
    `Tiny1M3MGCConfig` preset (inherits `Tiny1M3MConfig`,
    flips `use_gc=True`).
  - `optimizers/__init__.py`: export `GCAdamW` (parallel to
    `MARSAdamW`, `DAdaptAdamW`, `CAME`).
  - `training/trainer.py`: in `setup_muon_optimizer`, when
    `config.use_gc=True`, route the AdamW params through
    `GCAdamW` instead of `torch.optim.AdamW` (or whichever
    AdamW replacement is active — GC composes by replacing
    `AdamW` at the base of the chain, NOT by wrapping the
    replacement). The 2-D Muon path is unchanged.
  - `train_llm.py`: add `--use_gc` and `--gc_axis` CLI flags.
- **Identity at step 0**: GC modifies `g_t` to
  `g_t − mean(g_t, dim=axis)`. The *forward graph* is unchanged,
  so `val_loss` at step 0 (computed before any optimizer step)
  is bit-identical to baseline. The first optimizer step itself
  is *not* bit-identical (the centered gradient has zero mean
  per output neuron, removing the constant component that
  AdamW's first step otherwise sees) — this is the lever's
  signature, not a bug. With `use_gc=False` (default) the
  trainer's AdamW path is bit-identical to baseline (the
  `GCAdamW` class is never instantiated).
- **Run command**:
  ```
  python train_llm.py --config tiny1m --use_gc true \
    --seed 42 --output_dir runs/127-gc-tiny1m3m
  ```
  Ctrl is `python train_llm.py --config tiny1m --seed 42
  --output_dir runs/127-ctrl-tiny1m3m` (no flag → baseline
  path). Final val_loss is read from `metrics.json` →
  `final_metrics.val_loss`. Tiny1M3MConfig gives 0.94M params
  / 3M tokens / 92 steps (one seed: 42), matching the
  autoresearch test specification.
- **PASS bar**: `Δ ≤ −0.005` vs the tiny1m3m ctrl val_loss
  (paper reports +0.3–1.0% top-1 in image-classification;
  tiny1m3m's val_loss scale puts 0.5% of the ~6.43 baseline
  at ~0.03, so PASS −0.005 is conservative). NULL band
  `|Δ| < 0.005`. DRIFT `> +0.005` (centering the gradient
  adds an O(B·T·d) reduction per param per step, which at
  0.94M should be cheap but the lever's value at this scale
  is the question).
