---
id: 135-adan
status: done
round: 1
updated: 2026-06-13T15:15:45Z
transfer-risk: low
plain: It uses an N-step lookback gradient (not just one-step momentum) to compute both the momentum and the variance — capturing longer-range signal than Adam's two-EMA design.
---

# 135 — Adan: Adaptive Nesterov Momentum with N-Step Lookback

## Source
Xie, Zhou, Lin, Li, Yan, Wang, Wang, "Adan: Adaptive Nesterov
Momentum Algorithm for Faster Optimizing Deep Models" (arXiv:2208.06677,
TPAMI 2022 / ICLR 2023 workshop). https://arxiv.org/abs/2208.06677

Validated on ImageNet ResNet-50, ViT-B/16/L/16, MAE pretraining,
Cascade Mask R-CNN, and a 7B-class language model (CogVLM-style).
The lever is the cleanest *N-step adaptive* optimizer published
in 2022 — combines Nesterov momentum with N-step gradient
lookback for the variance estimate.

## Mechanism
Adam uses 1-step momentum + 1-step second moment:
  `m_t = β1 · m_{t−1} + (1 − β1) · g_t`
  `v_t = β2 · v_{t−1} + (1 − β2) · g_t²`

Adan uses 1-step momentum + N-step second moment:
  `m_t = β1 · m_{t−1} + (1 − β1) · g_t`
  `v_t = β2 · v_{t−1} + (1 − β2) · (1/N) · Σ_{i=0}^{N-1} g_{t−i}²`
  `g_lookahead = g_t + β_lookahead · (g_t − g_{t−1})`     (Nesterov-style)
  `update = m_t / (√v_t + ε)`
  `w ← w − lr · update`

Where `N` is the lookback window (paper default `N = 4`). The
Nesterov-style `g_lookahead` adds an *extrapolated* gradient
that uses the recent gradient direction as a correction.

The intuition: Adam's `v_t` only sees `g_t` at step `t`, which
makes the variance estimate *noisy* when the gradient oscillates.
Adan's N-step lookback smooths the variance estimate over the
last N gradients, giving a more stable second moment.

**Identity at step 0**: with `m_0 = 0, v_0 = 0`, the first step
has `m_0 = (1 − β1) · g_0`, `v_0 = (1 − β2) · g_0²` (assuming
N=1 lookback). Different from AdamW's first step (which has
the Adam normalization `m̂/√v̂`). With `N = 0`, Adan collapses
to Nesterov-SGD (no `v` denominator).

The lever is **not** bit-identical to AdamW at step 0; the
deviation is `O(N)` in the `v_t` estimate.

## Design sketch
- `optimizers/adan.py` (new): `Adan` — `torch.optim.Optimizer`
  subclass with the N-step second moment and Nesterov-style
  lookahead. State per param: `exp_avg` (m), `exp_avg_sq` (v),
  `prev_grad` (g_{t−1} for Nesterov), `grad_queue` (last N
  gradients for variance). ~120 LoC.
- `training/trainer.py`: when `use_adan=True`, route AdamW-eligible
  params through `Adan`. The 2-D slot still uses Muon. ~10 LoC.
- `configs/llm_config.py`: add `use_adan: bool = False`,
  `adan_lr: float = 0.006`, `adan_beta1: float = 0.9`,
  `adan_beta2: float = 0.999`, `adan_lookahead_beta: float = 0.5`,
  `adan_n_lookback: int = 4`. ~10 LoC.
- LoC: ~140 total (under 200 ceiling).
- Identity at step 0: with N=4 lookback and `g_{t−1}` not yet
  defined, the first step has `v_0 = (1 − β2) · g_0²` (only one
  gradient in the queue). The first step magnitude is `O(1/N)`
  different from AdamW's first step.
- The intuition: at 0.94M with 92 steps, the gradient oscillation
  is moderate (small models have smoother gradients). Adan's
  N-step variance smoothing should help by averaging over the
  N recent gradients. A null would say "at 0.94M the gradients
  are smooth enough that N=1 is fine"; a win would say "the
  N-step smoothing reveals gradient signal that the N=1
  estimate misses".

## Scale evidence
- arXiv:2208.06677 (Xie et al. 2022, TPAMI): ImageNet ResNet-50
  (~25M), ViT-B/16/L/16 (~86M-307M), MAE pretraining
  (ViT-L ~307M), Cascade Mask R-CNN (~200M). Reports +0.5-1.5%
  top-1 on ImageNet, +0.3-0.8 on ViT-B/16.
- CogVLM (Wang et al. 2023): uses Adan for 7B-class LM training,
  reports faster convergence than AdamW.
- Transfer risk: **low**. Validated at ≥100M (ViT-L 307M,
  CogVLM 7B), the mechanism is scale-free. The N-step variance
  smoothing is well-defined at any scale.

## Why it's worth a slot
Adan is the cleanest *N-step* adaptive optimizer published
in 2022 — distinct from AdamW (1-step), from MARS (114 filed,
1-step momentum with variance correction), and from CAME
(123 filed, confidence-rescaled 1-step variance). Adan
tests "is the *N-step variance* a load-bearing lever at
0.94M?". A win would say "the N-step smoothing captures
longer-range gradient signal that 1-step misses"; a null
would say "at 0.94M the 1-step variance is sufficient and
the N-step lookback adds memory without gain". The lever
is ortho to every closed optimizer and is the most recent
high-quality pre-Moonlight lever with consistent ≥100M wins.

## Plan
- **New file** `optimizers/adan.py` (~130 LoC): `Adan(Optimizer)`.
  Per-param state: `exp_avg` (m), `exp_avg_sq` (v), `prev_grad`
  (g_{t−1} for the Nesterov extrapolated gradient), `grad_queue`
  (last N fp32 grad clones for the variance lookback). Step:
  `g = grad.float()`; build lookahead `g_la = g + β_la·(g − prev_grad)`
  (falls back to `g` on the first step); `m ← β1·m + (1−β1)·g_la`;
  `v ← β2·v + (1−β2)·mean(queue[i]²)` with the new `g²` appended to
  the queue and truncated to length N; `update = m / (√v + ε)`; decoupled
  weight decay. Cast `update` back to `p.dtype` before the in-place
  step. No bias correction (matches the paper's Algorithm 1).
- **`training/trainer.py`**: add a new `elif getattr(config, "use_adan", False)`
  branch in the AdamW-routing block (right after `use_adashift`, before
  `use_sd`) that imports `Adan` and instantiates it on `adamw_params`
  with `adan_lr`, `adan_beta1`, `adan_beta2`, `adan_eps`,
  `adan_lookahead_beta`, `adan_n_lookback`. ~12 LoC including the
  comment.
- **`configs/llm_config.py`**:
  - Add to `LLMConfig`: `use_adan: bool = False`, `adan_lr: float = 0.006`,
    `adan_beta1: float = 0.9`, `adan_beta2: float = 0.999`,
    `adan_eps: float = 1e-8`, `adan_lookahead_beta: float = 0.5`,
    `adan_n_lookback: int = 4`. ~10 LoC including the docstring.
  - Add `Tiny1M3MAdanConfig(Tiny1M3MConfig)` that flips `use_adan=True`.
    ~5 LoC.
- **`optimizers/__init__.py`**: add `from .adan import Adan` to the
  import list + append `'Adan'` to `__all__`. ~2 LoC.
- **A/B script** `_arq_135-adan.py` mirroring the came/radam/adashift
  pattern.
- **Identity at step 0**: with `m_0 = 0, v_0 = 0, prev_grad = None,
  grad_queue = []`, the first step has `m_1 = (1−β1)·g_0` (no Nesterov
  lookahead — `prev_grad` undefined so we skip the term), the queue
  receives `g_0²` and the variance term is `mean([g_0²]) = g_0²`, so
  `v_1 = (1−β2)·g_0²`. `update_0 = m_1 / (√v_1 + ε) ≈ g_0 / (|g_0| + ε)
  ≈ sign(g_0)`. NOT bit-identical to AdamW's first step (which uses
  the bias-corrected Adam normalization), but the magnitudes are
  similar — this is the lever's signature. The N=4 lookback ramps in
  over the first 4 steps. With `use_adan=False` (default) the `Adan`
  class is never instantiated — baseline path bit-identical.
- **Run command** (one seed: 42, tiny1m3m, 0.94M params, 3M tokens):
  ```
  /venv/main/bin/python _arq_135-adan.py
  ```
  Val loss is read from the trailing `val loss` line of stdout (same
  convention as the other `_arq_*.py` scripts).
- Total LoC: ~160 (under 200 ceiling).
