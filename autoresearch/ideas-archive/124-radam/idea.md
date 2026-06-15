---
id: 124-radam
status: done
round: 1
updated: 2026-06-13T14:45:54Z
transfer-risk: med
plain: It fixes Adam's hidden learning-rate warmup trick — instead of relying on a hand-tuned schedule, the optimizer auto-detects when it's safe to use the full step size.
---

# 124 — RAdam (Rectified Adam): Variance-Aware Adaptive LR

## Source
Liu, Jiang, He, Chen, Liu, Gao, Han, "On the Variance of the Adaptive
Learning Rate and Beyond" (arXiv:1908.03265, ICLR 2020).
https://arxiv.org/abs/1908.03265

Validated on ResNet/CIFAR-10, ImageNet, Transformer-XL/Dialog,
BERT fine-tuning, ViT. The lever is a *correction* to Adam's
bias-correction term that removes the need for a manual LR warmup
schedule — Adam's effective LR ramps up implicitly as `1/(1 − β2^t)`
which can be unstable early in training.

## Mechanism
Adam's update: `update = m̂_t / (√v̂_t + ε)` where `m̂_t = m_t / (1 − β1^t)`
and `v̂_t = v_t / (1 − β2^t)`. The denominator `1 − β2^t` can be very
small for early `t` (e.g. `t = 10, β2 = 0.999` ⇒ `1 − β2^t = 0.01`),
which causes Adam's effective LR to *spike* in the first 100 steps.
The standard fix is a manual warmup schedule — multiply the LR by
`min(1, t / warmup_steps)`.

RAdam derives a *closed-form* correction that accounts for the
variance of the denominator `1 − β2^t`:
  `ρ_t = (ρ_∞ − 4 · t² · (1 − β2)² · (β2^t − 1)² / (t · (1 − β2)² · (1 + β2)²)) / (1 − β2^t)`
  `if ρ_t > 4: update = m̂_t · √(ρ_t) / (√v̂_t + ε)`     (variance-bounded)
  `else: update = m̂_t`                                   (SGD-only, fallback)

The intuition: when `t` is small enough that the variance of `1/(1 − β2^t)`
is high, RAdam uses SGD (no `v̂_t`) for the early steps; when `t` is
large enough for the variance to settle, it switches to the full Adam update.

**Identity at step 0**: at `t = 1`, the denominator is `1 − β2 ≈ 0.001`
which gives `ρ_1 ≪ 4`, so RAdam uses the *SGD-fallback* path:
`update = m̂_1 = (1 − β1) · g_0`. This is **not bit-identical**
to AdamW's first step (which uses the full Adam-normalized update),
but the magnitude is comparable. The first step is `O(β1)` smaller
than AdamW (because `m̂_1 = (1−β1)·g_0`, no `v̂` denominator).

At `t ≈ 100`, `ρ_t > 4` and RAdam switches to the full Adam path.
The transition is automatic and occurs roughly when `1 − β2^t > 1/4`.

## Design sketch
- `optimizers/radam.py` (new): `RAdam` — `torch.optim.Optimizer`
  subclass with the variance-bounded correction. State per param:
  `exp_avg` (m), `exp_avg_sq` (v), `step_count`. ~80 LoC.
- `training/trainer.py`: when `use_radam=True`, route AdamW-eligible
  params through `RAdam`. The 2-D slot still uses Muon. ~10 LoC.
- `configs/llm_config.py`: add `use_radam: bool = False`,
  `radam_lr: float = 0.006`, `radam_beta1: float = 0.9`,
  `radam_beta2: float = 0.999`, `radam_eps: float = 1e-8`. ~10 LoC.
- LoC: ~100 total (under 200 ceiling).
- Identity at step 0: RAdam uses SGD-fallback at step 0, so the
  first step is `(1 − β1)·g_0` (no `v̂` denominator). Different
  from AdamW's first step (which has full `v̂`).
- The intuition: at 0.94M with 92 steps, the LR warmup is ~10 steps
  out of 92 (~10% of training). RAdam's variance-bounded correction
  *automatically* handles the early-step instability without a
  manual warmup. A null would say "the implicit Adam LR spike is
  benign at 0.94M and the manual warmup (or its absence) is fine";
  a win would say "the manual warmup is suboptimal and RAdam's
  closed-form correction gives a better early-step LR".

## Scale evidence
- arXiv:1908.03265 (Liu et al. 2019/2020): validated on CIFAR-10
  ResNet, ImageNet ResNet-50, Transformer-XL dialog model (LM),
  BERT fine-tuning. Reports parity-to-better vs AdamW with manual
  warmup, with the warmup *removed* from the schedule.
- Subsequent reproductions (HuggingFace `transformers` includes
  RAdam as a drop-in): consistent parity on LM fine-tuning.
- Transfer risk: **med**. Validated at ≥100M (BERT fine-tuning
  on 110M-base, ViT-B/16 86M, Transformer-XL is <100M but
  well-published), the mechanism is scale-free. The early-step
  variance is *most* pronounced at small run windows (92 steps
  is exactly where the warmup-vs-no-warmup distinction matters
  most).

## Why it's worth a slot
RAdam is the only Adam variant that *removes the warmup knob*
without changing the LR schedule (other Adam variants in the
closed wave 031-040 changed the LR or the moment shape; RAdam
changes the *early-step policy*). It is the cleanest test of
"does our `warmup_steps` config (whatever it is) help or hurt
at 0.94M?". A win would say "the manual warmup is suboptimal
and RAdam's closed-form correction is better"; a null would
say "at 0.94M the warmup is benign and RAdam's correction is
a no-op". The lever is ortho to every closed optimizer (the
031-040 wave tested LR/moment shape changes; RAdam tests the
*early-step transition* between SGD and Adam).

## Plan

### Files to add/change
- `optimizers/radam.py` (new, ~85 LoC): `RAdam` — `torch.optim.Optimizer`
  subclass with the variance-bounded correction.
  Per-parameter state: `exp_avg` (m), `exp_avg_sq` (v), `step`.
  Implements the `ρ_t` formula from Liu et al. 2019 §3.2:
  - if `ρ_t > 4`: `update = m̂_t · √(ρ_t) / (√v̂_t + ε)` (variance-bounded)
  - else: `update = m̂_t` (SGD-only fallback)
- `optimizers/__init__.py`: export `RAdam`.
- `configs/llm_config.py`:
  - add `use_radam: bool = False`, `radam_lr: float = 0.006`,
    `radam_beta1: float = 0.9`, `radam_beta2: float = 0.999`,
    `radam_eps: float = 1e-8` (~10 LoC).
  - add `Tiny1M3MRAdamConfig(Tiny1M3MConfig)` — the A/B treatment.
- `training/trainer.py`: route `adamw_params` through `RAdam` when
  `use_radam=True`. The 2-D Muon path is unchanged (RAdam is an AdamW
  replacement, like 114-MARS / 119-SAM / 120-DAdapt / 121-Prodigy /
  123-CAME). Insert as another `elif getattr(config, "use_radam", False)`
  branch in the AdamW-path selection cascade.

### Config flag name
`use_radam: bool = False` (default off → plain AdamW path, baseline
bit-identical).

### Identity at step 0
RAdam uses the SGD-fallback path at step 0 (`ρ_1 < 4`), so the first
step is `(1 − β1) · g_0` with **no `v̂_t` denominator**. This is NOT
bit-identical to AdamW's first step (which uses the full Adam-normalized
update) — but the magnitude is comparable (O(β1) smaller). The lever's
inherent first-step cost is the bet, not a bug. With
`use_radam=False` (default) the class is never instantiated and the
baseline AdamW path is bit-identical.

### Run command
```
/venv/main/bin/python -m configs.llm_config --config_class Tiny1M3MRAdamConfig \
  --output_dir runs/124-radam/tiny1m3m_seed42 \
  --seed 42 --train_tokens 3000000
```

### How final val_loss is read
`val_loss` is the last entry in `metrics.json` (`final_metrics.val_loss`),
written by `train_model(...)` at the final milestone. Compare against
the tiny1m3m ctrl (`Tiny1M3MConfig`, val 6.4306).

### Pass / NULL / DRIFT criteria
- PASS ≤ ctrl − 0.005 (small/null band — the lever is on the
  early-step LR transition, which only matters in the warmup window
  at tiny1m3m's 92-step budget).
- NULL band |Δ| < 0.005.
- DRIFT > +0.005.
