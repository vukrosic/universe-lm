---
id: 137-adamp
status: done
round: 1
updated: 2026-06-13T20:50:47Z
transfer-risk: med
plain: Before each AdamW step, it removes the part of the momentum that points in the weight's own direction — so the optimizer changes the weight's magnitude without rotating it.
---

# 137 — AdamP: Adam with Projection-Based Update

## Source
He, Liu, Mao, Chen, Zhang, "AdamP: Slowing Down the Slowdown for
Momentum Accelerators on Scale-Invariant Weights" (arXiv:2006.08217,
NeurIPS 2020). https://arxiv.org/abs/2006.08217

Validated on CIFAR-10/100 ResNet, ImageNet ResNet-50/101,
DETR-style object detection, and several ViT ablations. The
lever is a *projection* trick — before applying the AdamW update,
project it onto the orthogonal complement of the weight's
direction, so the update doesn't rotate the weight (only
scales its magnitude).

## Mechanism
Standard AdamW update: `Δ = lr · m̂ / (√v̂ + ε)`. AdamP projects
`Δ` onto the orthogonal complement of `w`:
  `δ = Δ · w / ‖w‖²`     (component of Δ along w)
  `Δ_proj = Δ − δ · w_normalized`     (subtract the parallel component)
  `w ← w − Δ_proj + (λ · ‖w‖) · w_normalized`     (apply projected update
                                                    + L2 reg on magnitude)

Where `λ · ‖w‖` is the L2 weight decay (paper's "L2 regularization
on the magnitude of w"). The key insight: `Δ_proj` rotates the
weight without changing its magnitude, while the L2 reg shrinks
the magnitude without rotating. The two are *spectrally decoupled*
(similar to 128 Spectral Decoupling, but achieved via projection
rather than operator decomposition).

**Identity at step 0**: with `w_0` initialized at small random
values, the parallel component `δ` is small (because `w_0` and
`Δ_0` are nearly orthogonal at init for standard inits). So
`Δ_proj ≈ Δ_0` and the first step is approximately AdamW's first
step. **Approximately bit-identical** to AdamW at step 0 (within
`O(1/√d)` error from the projection).

## Design sketch
- `optimizers/adamp.py` (new): `AdamP` — `torch.optim.Optimizer`
  subclass with the projection + magnitude-only L2 reg. State
  per param: `exp_avg` (m), `exp_avg_sq` (v), standard AdamW state.
  ~90 LoC.
- `training/trainer.py`: when `use_adamp=True`, route AdamW-eligible
  params through `AdamP`. The 2-D slot still uses Muon. ~10 LoC.
- `configs/llm_config.py`: add `use_adamp: bool = False`,
  `adamp_lr: float = 0.006`, `adamp_beta1: float = 0.9`,
  `adamp_beta2: float = 0.999`, `adamp_eps: float = 1e-8`,
  `adamp_lambda: float = 0.01` (L2 reg magnitude). ~10 LoC.
- LoC: ~110 total (under 200 ceiling).
- Identity at step 0: approximately bit-identical to AdamW at
  step 0 (within `O(1/√d)` from the projection).
- The intuition: at 0.94M, the AdamW update rotates the weight
  by `O(‖Δ‖/‖w‖)` per step. After many steps, the cumulative
  rotation can drift the weight away from its optimal direction.
  AdamP's projection prevents the rotation by removing the
  parallel component. A null would say "at 0.94M the rotation
  is benign"; a win would say "the rotation accumulates and the
  projection removes it, giving cleaner gradient signal".

## Scale evidence
- arXiv:2006.08217 (He et al. 2020, NeurIPS 2020): CIFAR-10/100
  ResNet, ImageNet ResNet-50/101, DETR (~40M), several ViT
  ablations. Reports +0.3-1.0% top-1 over AdamW with the same
  LR and weight decay.
- Transfer risk: **med**. Validated at image-classification
  scale (≤100M for the headline experiments, though the paper
  also reports ResNet-101 which is 44M). The mechanism is
  scale-free (projection is well-defined at any scale). At
  0.94M the rotation is small per-step but accumulates over
  92 steps.

## Why it's worth a slot
AdamP is the projection-based analog of Spectral Decoupling
(128 filed). Both address the *weight-rotation* side effect of
L2 reg + gradient descent, but with different mechanisms:
Spectral Decoupling uses an operator decomposition, AdamP uses
an explicit projection. The two are *complementary* levers
and filing both gives us an A/B on "which regularization
decomposition wins at 0.94M?". AdamP is also ortho to every
closed optimizer lever (031-040, 001-006) — projection is a
novel mechanism. A win would say "weight rotation is hurting
and projection fixes it"; a null would say "at 0.94M the
rotation is benign and the projection adds compute without gain".

## Plan
- **New file**: `optimizers/adamp.py` (~90 LoC). Implements
  `AdamP(Optimizer)` with the per-param projection + magnitude-only
  L2 reg. State per param: `exp_avg` (m), `exp_avg_sq` (v), standard
  AdamW state.
- **Edit** `optimizers/__init__.py`: export `AdamP` (1 line).
- **Edit** `configs/llm_config.py`: add `use_adamp: bool = False`,
  `adamp_lr: float = 0.006`, `adamp_beta1: float = 0.9`,
  `adamp_beta2: float = 0.999`, `adamp_eps: float = 1e-8`,
  `adamp_lambda: float = 0.01`, `adamp_wd_apply_separately: bool = True`.
  Total ~15 LoC. Plus a `Tiny1M3MAdamPConfig` A/B preset.
- **Edit** `training/trainer.py`: add `elif getattr(config,
  "use_adamp", False):` branch in the AdamW-eligible optimizer
  selection block. Routes the same `adamw_params` through
  `AdamP`. ~25 LoC of wiring + comment.
- **LoC**: ~130 total (under 200 ceiling).
- **Identity at step 0**: the projection term `(Δ · w / ‖w‖²) · w`
  is small for symmetric inits (`O(1/√d)` in `‖w‖` norm). With
  `adamp_lambda=0.0` the magnitude-shrinking L2 reg is removed
  and the AdamP update is `Δ_proj = Δ − (Δ·w/‖w‖²)·w`. For
  standard Kaiming/Xavier inits the projection removes
  `O(1/√fan_in)` of the update — the first optimizer step is
  *approximately* AdamW's first step but with a small `O(1/√d)`
  correction. This is the lever's signature, not a bug.
- **Run command**:
  `/venv/main/bin/python train_llm.py --config configs/llm_config.py:Tiny1M3MAdamPConfig --seed 42`
- **Result read**: from `runs/tiny1m3m_adamp_*/metrics.json`
  `final_val_loss` (or last `eval_milestones` entry).
- **A/B**: `Tiny1M3MAdamPConfig` vs `Tiny1M3MConfig` (val 6.4306).
  PASS ≤ 6.4206 (Δ ≤ −0.01). NULL band |Δ| < 0.01.
  DRIFT > +0.01.
