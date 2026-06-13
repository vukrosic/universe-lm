---
id: 126-adashift
status: done
round: 1
updated: 2026-06-13T14:50:33Z
transfer-risk: high
plain: It decorrelates Adam's running averages by using a deliberately delayed copy of the gradient — the idea is that today's update should ignore what it just saw and react to slightly older info instead.
---

# 126 — AdaShift: Decorrelated Adam via Delayed Gradients

## Source
Zhou, Yang, Wang, Wang, "AdaShift: Decorrelation and Convergence of
Adaptive Learning Rate Methods" (arXiv:1810.00143, NeurIPS 2019
workshop). https://arxiv.org/abs/1810.00143

Validated on CIFAR-10 ResNet, ImageNet ResNet-50, and small-scale
LM (LSTM char-LM). The lever is a *temporal* decorrelation trick —
the second moment uses a *delayed* gradient to avoid the strong
auto-correlation between `g_t²` and the EMA `v_t`.

## Mechanism
Adam's update: `v_t = β2 · v_{t−1} + (1 − β2) · g_t²`. AdaShift uses
a *shifted* gradient:
  `v_t = β2 · v_{t−1} + (1 − β2) · g_{t−n}²`
  `update = m_t / (√v_t + ε)`
  `w ← w − lr · update`

Where `n` is the *delay* (paper default `n = 3`). The intuition:
Adam's `v_t` is highly auto-correlated with `m_t` (both use `g_t`),
which creates a *bias* in the adaptive step size — `v_t` increases
exactly when `m_t` increases, so the Adam normalization cancels
out some of the actual signal. Using `g_{t−n}²` decorrelates `v_t`
from `m_t`, so the normalization captures the *recent* gradient
size without being driven by the *current* gradient.

**Identity at step 0**: with `n = 3`, the first step has `v_1` using
`g_{−2}² = 0` (no past gradient). The paper handles this with
a warmstart `v_0 = g_0²` so the first step uses `v_1 = β2 · g_0² + (1−β2) · 0 = β2 · g_0²`.
Different from AdamW's first step (which uses `v_1 = (1−β2)·g_0²`),
but same magnitude order.

With `n = 0`, AdaShift collapses to AdamW. The PASS bar is defined
at the smallest non-trivial `n` (paper default `n = 3`).

## Design sketch
- `optimizers/adashift.py` (new): `AdaShift` — `torch.optim.Optimizer`
  subclass with the delayed-gradient update. State per param:
  `exp_avg` (m), `exp_avg_sq` (v), and a small queue of past `n`
  gradients. ~80 LoC.
- `training/trainer.py`: when `use_adashift=True`, route AdamW-eligible
  params through `AdaShift`. The 2-D slot still uses Muon. ~10 LoC.
- `configs/llm_config.py`: add `use_adashift: bool = False`,
  `adashift_lr: float = 0.006`, `adashift_n: int = 3` (delay),
  `adashift_beta1: float = 0.9`, `adashift_beta2: float = 0.999`. ~10 LoC.
- LoC: ~100 total (under 200 ceiling).
- Identity at step 0: with `n = 3`, the first step uses `v_1 = β2 · g_0²`
  (warm-start). The displacement is `O(β2)` different from AdamW's
  first step.
- The intuition: at 0.94M with 92 steps, Adam's auto-correlation
  between `m_t` and `v_t` is moderate (small gradients are noisy).
  AdaShift's decorrelation removes the bias. A null would say
  "the auto-correlation is benign at 0.94M"; a win would say
  "the decorrelation reveals gradient-direction signal that Adam
  was suppressing".

## Scale evidence
- arXiv:1810.00143 (Zhou et al. 2019): validated on CIFAR-10
  ResNet (small scale), ImageNet ResNet-50 (medium scale),
  char-LM LSTM (small scale). Reported gains are small (0.1-0.3%
  on ImageNet, modest on char-LM) but consistent.
- Transfer risk: **high**. Validated at small scale only (CIFAR,
  char-LM, ImageNet ResNet-50 is 25M which is on the boundary
  of "medium"). The 92-step tiny1m3m context is *most* relevant
  to the small-scale experiments. The lever is also reported to
  be brittle to delay choice (n=1,2,3,4 all have to be tuned).

## Why it's worth a slot
AdaShift is the only filed optimizer lever that targets *Adam's
internal auto-correlation*. Every other closed optimizer lever
(031-040, 001-006) changes the *shape* of the update (sign,
moment, LR); AdaShift changes the *temporal alignment* between
m_t and v_t. The slot is a clean test of "is Adam's v_t-m_t
auto-correlation load-bearing at tiny scale?". A win would
say "the auto-correlation is biasing the step direction and
AdaShift's decorrelation helps"; a null would say "at 0.94M
the auto-correlation is benign and AdaShift adds delay without
gaining information". The high transfer-risk is honest — this
lever's published evidence is at small scale.

## Plan

**Files touched**
- `optimizers/adashift.py` (new, ~120 logic LoC): `AdaShift`
  optimizer subclassing `torch.optim.Optimizer`. Per-parameter
  state: `exp_avg` (m), `exp_avg_sq` (v, lazy warm-start),
  `grad_queue` (list of past `n` fp32 grad clones, bounded to
  length n), and `step`. The delayed gradient `g_{t-n}²` is
  read from `grad_queue[0]` when the queue is full, else 0.
- `optimizers/__init__.py` (+1 import, +1 in `__all__`):
  exports `AdaShift`.
- `configs/llm_config.py`: adds 6 config fields on `LLMConfig`
  (`use_adashift: bool = False`, `adashift_lr: float = 0.006`,
  `adashift_beta1: float = 0.9`, `adashift_beta2: float = 0.999`,
  `adashift_eps: float = 1e-8`, `adashift_n: int = 3`) and the
  `Tiny1M3MAdaShiftConfig` preset that flips the flag on with
  paper defaults.
- `training/trainer.py`: in `setup_muon_optimizer`, after the
  RAdam branch and before the cautious-AdamW branch, adds an
  `elif getattr(config, "use_adashift", False):` that constructs
  `AdaShift(adamw_params, lr=..., betas=..., eps=...,
  weight_decay=config.weight_decay, n=...)`. The 2-D Muon path
  is unchanged.
- `train_llm.py`: adds `--use_adashift`, `--adashift_lr`,
  `--adashift_n` CLI flags and their `config.X = args.X` lines.

**Config flag**: `use_adashift` (default `False`). With it off
the trainer uses plain `torch.optim.AdamW` on the
1-D / embedding / norm / head path and the Muon path on the
2-D slot — baseline bit-identical at step 0.

**Step-0 behavior with the flag ON**: warm-start `v_0 = g_0²`
⇒ `v_1 = β2·g_0²` (paper convention). AdamW's first step uses
`v_1 = (1−β2)·g_0²`. The displacement is O(β2) different from
AdamW, same magnitude order. This is the lever, not a bug, and
is explicitly documented in the idea. With `n = 0` AdaShift
collapses to AdamW (no decorrelation).

**Run command** (tiny1m3m, seed 42, off-the-shelf preset):
```
/venv/main/bin/python /root/universe-lm/train_llm.py \
  --config_class configs.llm_config.Tiny1M3MAdaShiftConfig \
  --seed 42 --output_dir /root/universe-lm/runs/126-adashift-trt
```
The matching ctrl run uses `--config_class
configs.llm_config.Tiny1M3MConfig`. Final val_loss read from
`/root/universe-lm/runs/126-adashift-trt/metrics.json`
(`final_metrics.val_loss`).
