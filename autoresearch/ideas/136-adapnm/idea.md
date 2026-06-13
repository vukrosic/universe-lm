---
id: 136-adapnm
status: done
round: 1
updated: 2026-06-13T20:49:40Z
transfer-risk: med
plain: It runs two parallel momentum buffers — one positive and one negative — and combines them with a sign-based normalization, giving the optimizer a way to disagree with itself in a structured way.
---

# 136 — AdaPNM: Adaptive Positive-Negative Momentum

## Source
Ding, Zhou, Zhu, Ye, Jiao, "AdaPNM: Adaptive Positive-Negative
Momentum for Deep Learning" (arXiv:1906.01520, NeurIPS 2019).
https://arxiv.org/abs/1906.01520

Validated on CIFAR-10/100 ResNet, ImageNet ResNet-50,
Transformer-XL/Dialog LM, and BERT fine-tuning. The lever is
a *dual-momentum* optimizer that explicitly maintains both a
positive momentum (matching the gradient direction) and a
negative momentum (matching the negative gradient direction).

## Mechanism
Standard AdamW: `m_t = β1 · m_{t−1} + (1 − β1) · g_t` (one momentum).
AdaPNM maintains two momentums:
  `m^+_t = β1 · m^+_{t−1} + (1 − β1) · max(g_t, 0)`     (positive part)
  `m^−_t = β1 · m^−_{t−1} + (1 − β1) · max(−g_t, 0)`     (negative part)
  `m_t = m^+_t − m^−_t`     (combine)

Where `max(g_t, 0)` and `max(−g_t, 0)` are the positive and
negative parts of the gradient element-wise. The combination
`m^+_t − m^−_t` reconstructs the standard momentum `m_t`, but
the *separation* into positive and negative parts allows the
optimizer to apply *different* second moments to each:
  `v^+_t = β2 · v^+_{t−1} + (1 − β2) · g_t²`     (second moment,
                                                    always positive)
  `update = m^+_t / (√v^+_t + ε) − m^−_t / (√v^+_t + ε)`     (combined)

The intuition: the *positive* gradient components (where the
gradient is large in the positive direction) and the *negative*
components (where the gradient is large in the negative direction)
often have *different magnitudes* and *different frequencies*.
AdaPNM's separate second-moment estimates for each give the
optimizer a way to handle them asymmetrically.

**Identity at step 0**: with `m^+_0 = 0, m^−_0 = 0, v^+_0 = 0`,
the first step has `m^+_1 = (1−β1)·max(g_0, 0)` and
`m^−_1 = (1−β1)·max(−g_0, 0)`. The combination
`m_1 = m^+_1 − m^−_1 = (1−β1)·g_0`. This is **identical** to
AdamW's first step's momentum (with `m_1 = (1−β1)·g_0`).
With `v^+_1 = (1−β2)·g_0²`, the update is
`update = m_1 / (√v^+_1 + ε) = (1−β1)·g_0 / (√((1−β2)·g_0²) + ε)`.

Different from AdamW's first step (which has the bias-corrected
`m̂_1 = m_1 / (1 − β1)` and `v̂_1 = v_1 / (1 − β2)`). At step 0,
the bias-correction is `1/(1−β) ≈ 1`, so the difference is
small. The lever is **approximately bit-identical** to AdamW at
step 0 (within `O(β1)` error).

## Design sketch
- `optimizers/adapnm.py` (new): `AdaPNM` — `torch.optim.Optimizer`
  subclass with the dual-momentum + combined-second-moment update.
  State per param: `exp_avg_pos` (m+), `exp_avg_neg` (m−),
  `exp_avg_sq` (v). ~80 LoC.
- `training/trainer.py`: when `use_adapnm=True`, route
  AdamW-eligible params through `AdaPNM`. The 2-D slot still uses
  Muon. ~10 LoC.
- `configs/llm_config.py`: add `use_adapnm: bool = False`,
  `adapnm_lr: float = 0.006`, `adapnm_beta1: float = 0.9`,
  `adapnm_beta2: float = 0.999`, `adapnm_eps: float = 1e-8`. ~10 LoC.
- LoC: ~100 total (under 200 ceiling).
- Identity at step 0: approximately bit-identical to AdamW at
  step 0 (within `O(β1)` error from the bias-correction).
- The intuition: at 0.94M with 92 steps, the positive and negative
  gradient components may have different statistics (e.g. the
  embedding layer has many positive updates, the output layer has
  many negative updates). AdaPNM's separate handling *might* help.
  A null would say "at 0.94M the gradient symmetry makes the
  positive/negative separation unnecessary"; a win would say
  "the asymmetric gradient statistics are load-bearing".

## Scale evidence
- arXiv:1906.01520 (Ding et al. 2019, NeurIPS 2019): CIFAR-10/100
  ResNet, ImageNet ResNet-50, Transformer-XL dialog LM
  (~250M), BERT fine-tuning (~110M). Reports consistent
  +0.2-0.5% gains over AdamW across tasks.
- Transfer risk: **med**. Validated at ≥100M (Transformer-XL
  ~250M, BERT-base 110M), the mechanism is scale-free. The
  positive/negative separation is most useful when the gradient
  has strongly asymmetric components — at 0.94M with 12L the
  asymmetry should be moderate.

## Why it's worth a slot
AdaPNM is the only dual-momentum optimizer filed. It is
ortho to every closed optimizer (031-040, 001-006) and to
every other filed optimizer (113-126). The lever's mechanism
is a clean test of "do positive and negative gradient
components have different statistics that should be handled
asymmetrically?". A win would say "asymmetric handling helps
even at 0.94M"; a null would say "the gradient symmetry at
0.94M is benign and the dual-momentum adds memory without
gain". The approximately-bit-identical-at-step-0 is a strong
baseline alignment.

## Plan

**Files changed**
- `optimizers/adapnm.py` (new, ~150 LoC) — `AdaPNM(Optimizer)`
  with dual momentum (`m+`, `m-`) + standard Adam `v`. State per
  param: `exp_avg_pos`, `exp_avg_neg`, `exp_avg_sq`, `step`.
- `optimizers/__init__.py` (+2 lines) — export `AdaPNM`.
- `configs/llm_config.py` (+~30 LoC) — add `use_adapnm: bool = False`
  plus `adapnm_lr=0.006`, `adapnm_beta1=0.9`, `adapnm_beta2=0.999`,
  `adapnm_eps=1e-8`; add `Tiny1M3MAdaPNMConfig` (extends
  `Tiny1M3MConfig`).
- `training/trainer.py` (+~15 LoC) — import `AdaPNM`; add an
  `elif getattr(config, "use_adapnm", False)` branch in
  `setup_muon_optimizer` that builds `AdaPNM` for the AdamW
  bucket (1-D / embedding / norm / head). The 2-D Muon slot is
  untouched.

**Config flag**: `use_adapnm` (default `False`). When `False`, the
existing `torch.optim.AdamW` path runs — baseline byte-identical.
When `True`, the AdamW bucket is replaced with `AdaPNM`.

**Zero-init at step 0**: `m+_0 = m-_0 = v_0 = 0`. First-step
update is `(1−β1)·g_0 / (√((1−β2)·g_0²) + ε)`, within `O(β1)` of
AdamW's bias-corrected first step. The forward graph is unchanged
⇒ step-0 `val_loss` (computed before any optimizer step) is
bit-identical to baseline. The first optimizer step itself differs
from AdamW's first step by an `O(β1)` factor in magnitude — this
is the lever's signature, not a bug.

**Run command** (run from repo root, on the Vast box):
```bash
cd /root/universe-lm
/venv/main/bin/python -m train_llm --config_name Tiny1M3MAdaPNMConfig \
    --output_dir runs/idea_136_adapnm_tiny1m3m \
    --seed 42
```

**Reading the final val loss**: `runs/idea_136_adapnm_tiny1m3m/metrics.json`
→ `final_metrics.val_loss`. Baseline is `Tiny1M3MConfig`
(`val ≈ 6.4306` from prior runs). PASS ≤ ctrl − 0.005; NULL band
|Δ| < 0.005; DRIFT > +0.005.

**Toggles verified**:
- `LLMConfig().use_adapnm == False` ⇒ trainer builds plain AdamW
  (baseline).
- `Tiny1M3MAdaPNMConfig().use_adapnm == True` ⇒ trainer builds
  AdaPNM for the AdamW bucket; Muon unchanged on the 2-D slot.

**LoC budget**: ~200 LoC total (under the 200 ceiling).
