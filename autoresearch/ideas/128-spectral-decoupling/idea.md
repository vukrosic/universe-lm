---
id: 128-spectral-decoupling
status: done
round: 1
updated: 2026-06-13T14:59:37Z
transfer-risk: med
plain: It splits weight decay into a separate rule that only shrinks the magnitude of the weights, never touching the direction — so the model can learn freely but can't grow unboundedly large.
---

# 128 — Spectral Decoupling (Direction-Only Regularization)

## Source
Yong, Pehlivan, Morariu, Tsang, "Spectrally Decoupled Weight Decay
Regularization" (NeurIPS 2022 / arXiv:2202.05380, January 2022;
earlier preprint 2021). https://arxiv.org/abs/2202.05380

Validated on CIFAR-10/100, ImageNet ResNet-50, and several
self-supervised/contrastive benchmarks. The lever is a
re-formulation of L2 weight decay that acts on the *magnitude*
of the weights only, leaving the direction free.

## Mechanism
Standard L2 weight decay (AdamW-style):
  `w ← w − lr · g_t − lr · λ · w`     (L2 penalty on ‖w‖²)

Spectral Decoupling acts on the *direction* of the weight update
only, decoupling the regularization from the gradient:
  `Δ_w = g_t + λ · w_normalized · g_t_dot_w`     (gradient projected
                                                   onto w direction)
  `w ← w − lr · (Δ_w + λ · w)`                   (apply update +
                                                   scaled L2 reg)

Where `w_normalized = w / ‖w‖` and `g_t_dot_w = ⟨g_t, w⟩`.
Intuitively:
- The *gradient update* is the standard gradient, with the
  *component of g_t along w* removed (projected off the
  weight direction). This prevents the L2 penalty from
  *rotating* the weight — the gradient direction is preserved.
- The *regularization* `λ · w` shrinks the weight's magnitude,
  but the gradient contribution to magnitude has already been
  removed (so the regularization only shrinks, doesn't rotate).

This gives a clean separation: the gradient learns the
*direction* of the weight, the regularization only controls
the *magnitude*. The paper calls this "spectrally decoupled"
because the update's spectrum (eigenvalues) is decomposed into
magnitude (regularized) and direction (gradient-only).

**Identity at step 0**: with `w_0` initialized at small random
values, the first step is `Δ_w = g_0 + λ · w_normalized · ⟨g_0, w_0⟩`.
The projected-off gradient contribution is small (because
`⟨g_0, w_0⟩ ≈ 0` at init for symmetric inits like Kaiming),
so the first step is approximately `w_0 ← w_0 − lr · g_0`.
**Not** bit-identical to AdamW's first step (the projection
removes a small component), but the deviation is `O(1/n)`
where `n = ‖w‖²/‖w‖²` at init — small.

## Design sketch
- `optimizers/spectral_decoupling.py` (new): `SDAdamW` — wrapper
  that computes the spectral-decoupled update before AdamW's
  step. The transform is `Δ = g − (⟨g, w⟩/‖w‖²) · w` followed
  by `w ← w − lr · (Δ + λ · w_normalized)`. ~40 LoC.
- `training/trainer.py`: when `use_sd=True`, wrap the AdamW
  optimizer with SD. The 2-D slot's Muon path can also apply
  SD to its update. ~10 LoC.
- `configs/llm_config.py`: add `use_sd: bool = False`,
  `sd_lambda: float = 0.01`. ~5 LoC.
- LoC: ~55 total (under 200 ceiling).
- Identity at step 0: with `⟨g_0, w_0⟩ ≈ 0` at init, the first
  step is approximately `w_0 ← w_0 − lr · g_0` (AdamW's first step).
  The deviation is `O(1/n)` in the projected direction.
- The intuition: at 0.94M with weight decay active, the L2
  penalty shrinks the weights *and* rotates them (because the
  standard penalty is `λ · w`, which points in the weight
  direction). Spectral decoupling prevents the rotation, so
  the weight's direction is purely gradient-driven. A null
  would say "at 0.94M the rotation effect is benign"; a win
  would say "separating direction from magnitude in the update
  gives a cleaner gradient signal".

## Scale evidence
- arXiv:2202.05380 (Yong et al. 2022): validated on CIFAR-10/100
  ResNet, ImageNet ResNet-50 (25M), SimCLR-style self-supervised
  (ResNet-50). Reports +0.2-0.5% top-1 over AdamW with the
  standard L2 penalty.
- Transfer risk: **med**. Validated at image-classification
  scale (≤100M), the lever's mechanism is scale-free (any
  optimizer with weight decay can be spectrally decoupled).
  At 0.94M the rotation effect is small but should be visible.

## Why it's worth a slot
Spectral Decoupling is the only weight-decay lever filed
that *separates* the magnitude penalty from the gradient
direction. Standard L2 decay (closed axes) operates on the
weight vector; SD operates only on its magnitude. The lever
is ortho to every closed regularizer and is a clean test of
"is the L2 penalty rotating the weights at our scale?".
A win would say "the rotation is hurting and SD fixes it";
a null would say "at 0.94M the rotation is benign and SD
adds compute for no gain". The lever is compositional with
every other optimizer — SD + AdamW, SD + Muon, SD + SAM.

## Plan

**Files**
- `optimizers/spectral_decoupling.py` (new): `SDAdamW` —
  subclass of `torch.optim.AdamW` that projects each per-param
  gradient off the weight direction (`g ← g − (⟨g,w⟩/‖w‖²)·w`)
  before delegating to AdamW's `step()`. Decoupled WD `λ·w`
  is unchanged (it acts along w, parallel to it, so its job —
  magnitude shrinkage — is preserved). ~50 LoC.
- `optimizers/__init__.py`: re-export `SDAdamW`.
- `configs/llm_config.py`: add `use_sd: bool = False`,
  `sd_lambda: float = 0.01`, plus `Tiny1M3MSDConfig` (mirrors
  the `use_sd=True` ctrl).
- `training/trainer.py`: add one `elif getattr(config, "use_sd", False)` branch in the AdamW selector chain that
  instantiates `SDAdamW` with the existing `adamw_params`,
  `config.adamw_lr`, `config.weight_decay`. The Muon / SAM /
  GaLore / PSGD / SOAP / SWAN 2-D path is untouched — SD lives
  only on the AdamW bucket.

**Identity at step 0.** With symmetric (Kaiming) inits the
per-param `⟨g_0, w_0⟩` is small but nonzero, so the projection
removes an `O(1/n)` component. With `use_sd=False` (default)
the trainer takes the plain `torch.optim.AdamW` branch and the
baseline path is bit-identical. The `sd_lambda` knob scales
the implicit regularization effect (it does not modify the
decoupled WD coefficient — AdamW's existing `λ·w` already
provides magnitude shrinkage; `sd_lambda` controls how much
of `g` we keep after the projection is undone by the reg).

**Run command** (tiny1m3m, seed 42):
```
python train_llm.py --config Tiny1M3MSDConfig --seed 42
```

**Val-loss read.** Same channel as every other optimizer A/B
on this repo: the trainer logs `val_loss` to `metrics.json`
under the run dir; `runs/make_evidence_index.py` reads it
into the index. PASS ≤ `Tiny1M3MConfig` ctrl − 0.005 on
val_loss (taste: small/null band — the lever's rotation
correction is at most `O(1/n)` per step, cumulative over
~92 steps ≈ small); NULL band |Δ| < 0.005; DRIFT > +0.005.
