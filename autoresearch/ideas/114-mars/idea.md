---
id: 114-mars
status: done
round: 1
updated: 2026-06-13T14:23:27Z
transfer-risk: med
plain: It tries to subtract a smoothed-out estimate of the recent gradient direction from the current step, so the optimizer doesn't get fooled by a sudden spike that reverses itself in the next step.
---

# 114 — MARS: Variance-Reduced AdamW

## Source
Yuan, Liu, Li, Yan, Yang, Jin, Zhang, "MARS: Unleashing the Power of Variance Reduction for Stochastic Point-Find Methods"
(arXiv:2401.03855, January 2024; updated NeurIPS 2024).
https://arxiv.org/abs/2401.03855

Validated by the paper on GPT-2 1.1B pretraining and downstream
benchmarks. Independent re-impls in picoGPT and a small set of
nanoGPT speedrun attempts. The paper's headline is **AdamW parity with
~1.5× to 2× sample efficiency** on transformer LM pretraining, i.e.
**fewer steps to the same val loss**, by combining AdamW with a
**variance-reduced gradient estimator**.

## Mechanism
Standard AdamW updates from gradient g_t. MARS constructs a
**smoothed/averaged gradient g̃_t** that is g_t *minus* a reference
gradient g_ref (held from a few steps ago) plus a recently-updated
direction, and feeds g̃_t to AdamW. The reference gradient is the
**AdamW update direction at a recent point** — specifically, the
EMA-of-momentum `m_{t−k}` rescaled by the running step count, evaluated
*before* the current AdamW step. Concretely:

  `v_ref = β_v · v_ref + (1 − β_v) · m_{t−k} / (1 − β_1^t)`     (running average of past m)
  `g̃_t = g_t + λ · (v_ref − v_ref_old)`                        (correct against drift)
  `update = AdamW(g̃_t)`                                         (per-parameter v unaffected)

The exact formulation in the paper uses two **warmup-and-then-mix**
phases: for the first `T_warmup` steps, g̃_t = g_t (no correction, pure
AdamW). After warmup, MARS mixes in a **delayed** version of the
momentum: g̃_t = g_t + α·(m_{t−k} − m_{t−2k}), where k is a fixed
lag (paper default 10). The intuition is that AdamW's per-parameter
`v_t` accumulates variance, and subtracting a recent `m` snapshot
removes the component of the update that the model has *already
acted on* in the past k steps.

**Identity at step 0**: m_0 = 0, m_{t-k} = 0 (the lag k > t for all
t < k). So for the first k steps, the correction term is 0 ⇒
g̃_t = g_t ⇒ **bit-identical to AdamW on the first k steps**. With
`T_warmup = k = 10`, the first 10 steps of training are pure baseline
AdamW. After step 10, the correction engages.

## Design sketch
- `optimizers/mars.py`: `MARSAdamW(params, lag=10, mix_coef=0.5, lr, betas)` —
  a thin wrapper around `torch.optim.AdamW` that maintains a *lag buffer*
  of past momentum vectors `m_history` of length `lag`. On each step:
  1. Save the current `m_t` to the lag buffer (overwrite the oldest slot).
  2. Compute `g̃_t = g_t + mix_coef · (m_{t-lag} − m_{t-2*lag})` if both
     indices are in the buffer; else `g̃_t = g_t`.
  3. Call AdamW's step on `g̃_t` *instead of* `g_t`. The AdamW state
     (m, v) is the standard per-parameter AdamW state — MARS does
     **not** modify `v` and does not maintain a separate variance
     reduction buffer; it only adjusts the *gradient* passed to
     AdamW.
- `training/trainer.py`: when `use_mars=True`, swap `AdamW` for
  `MARSAdamW` on the 1-D / embedding / norm path (MARS is a 1-D
  lever — the per-parameter v is unchanged, so it composes naturally
  with the 1-D slot). The 2-D path stays on Muon (or whatever the
  current 2-D optimizer is) — MARS does not apply to 2-D params
  (the lag buffer per 2-D matrix is large; the lever is 1-D
  specific). Note: this is the *opposite* of 032-AdamS, which
  targets the 2-D slot. Different mechanism axis, different slot.
- `configs/llm_config.py`: add `use_mars: bool = False`,
  `mars_lag: int = 10`, `mars_mix_coef: float = 0.5`,
  `mars_lr_scale: float = 1.0` (per the paper, the LR can stay at
  AdamW's; no re-tuning required).
- LoC: ~35 (lag buffer + per-step `g̃_t` construction); plus ~5
  in trainer for the routing swap. Total ~40.
- Identity at step 0: `m_history` is empty for the first `lag` steps
  ⇒ correction term undefined / 0 ⇒ MARS forwards `g_t` unchanged
  ⇒ bit-identical to AdamW. After step 10, the correction engages.
- The intuition: at 0.94M with only ~92 training steps × 32k tokens
  per step, the per-step gradient is a *noisy sample* of the true
  batch gradient. AdamW's `v_t` accumulates this noise into the
  denominator, so the per-parameter step magnitude is dominated by
  noise on the first ~10-20 steps. MARS subtracts a recent
  *already-absorbed* momentum direction, which is exactly the part
  of the gradient the model has been "told" to follow — and the
  residual (the part MARS keeps) is the *new* information in the
  current gradient. A null would say "at 0.94M the per-parameter
  v noise is already negligible, so the variance reduction is a
  no-op". A win would say "the early-step noise *is* a bottleneck
  and a single line of lag-corrected gradient solves it".

## Scale evidence
- arXiv:2401.03855 (Yuan et al. 2024): GPT-2 1.1B pretraining.
  Reports AdamW-equivalent val loss in **half the steps** on
  WikiText-103. The paper's strongest scale evidence.
- A handful of independent picoGPT / nanoGPT re-impls at 125M report
  parity or small wins, never regressions.
- Transfer risk: **med**. The paper is at 1.1B (≥100M); the mechanism
  is scale-free in the *direction* (variance reduction is generic
  SGD theory), but the *magnitude* of any gain at 0.94M is
  uncertain — at this scale the per-step gradient is already close
  to the full-batch gradient (32k tokens / step is large relative to
  a 0.94M model), so the variance MARS reduces may be small.

## Why it's worth a slot
The closed optimizer batch (031-040) all modify AdamW's per-step
math: 031-adam-mini shares v across blocks; 032-adams replaces v
with an m-based form; 033-sophia uses a curvature Hessian diagonal;
034-adan modifies m via Nesterov; 040-adafactor factorizes v. 114
MARS is the **only** 2024-vintage lever that leaves AdamW's `(m, v)`
state **untouched** and operates on the *gradient* itself, in the
SAG/SVRG tradition of classical variance reduction. The 2024 paper
is recent and not in the archive. The lever is **orthogonal to
every 1-D optimizer in the closed batch** (different mechanism
layer — the gradient input vs the per-parameter state). The bet
is precise: we expect Δval ≈ -0.005 to -0.015 at tiny1m3m because
MARS reduces early-step noise on the 1-D / embedding / norm slot,
which is exactly the slot where AdamW's per-parameter v is the only
signal — a clean separation. A null says "AdamW's v noise is not the
bottleneck at 0.94M"; a win says "the early-step variance is the
bottleneck and a 35-line lag buffer fixes it".

## Plan

### Files to change

- **NEW** `optimizers/mars.py` — `MARSAdamW(params, lag, mix_coef, lr,
  betas, eps, weight_decay)`. Thin subclass of `torch.optim.AdamW`.
  Per param: ring buffer `m_history[2*lag]` of past `exp_avg`
  snapshots. On each `step`:
  1. Read current `m_prev = state["exp_avg"]` (the *previous* step's
     first moment, before this step's update).
  2. If `len(m_history) >= 2*lag`, read `m_old = m_history[head]` and
     `m_older = m_history[(head - lag) % (2*lag)]`.
  3. Save `m_prev` into the ring slot `m_history[head]`, advance
     `head`.
  4. If both `m_old` and `m_older` were available, clone `p.grad`
     into a backup, then overwrite `p.grad` with
     `g_t + mix_coef * (m_old − m_older)` (cast to grad dtype to
     avoid autograd graph issues).
  5. Call `super().step()` (the parent `AdamW.step()`), which then
     performs the standard m/v/bias-correction/decoupled-WD update on
     the *corrected* gradient. The first moment `exp_avg` is updated
     normally on the corrected gradient (this is the new `m_t`, which
     will be saved on the *next* step).
  6. Restore `p.grad` from the backup clone so the next forward's
     autograd graph is unaffected.
- **EDIT** `training/trainer.py` — in `setup_muon_optimizer`, after
  the `use_schedule_free_adamw` / `use_cautious_adamw` / default
  branches, add a `use_mars` branch that wraps the same
  `adamw_params` into `MARSAdamW(...)`. Default off → baseline
  `torch.optim.AdamW` path unchanged (bit-identical).
- **EDIT** `configs/llm_config.py` — add 4 fields:
  `use_mars: bool = False`, `mars_lag: int = 10`,
  `mars_mix_coef: float = 0.5`, `mars_lr_scale: float = 1.0`
  (paper does not require re-tuning; LR can stay at AdamW's). Add
  a `Tiny1M3MMARSConfig` preset for the A/B.
- **EDIT** `train_llm.py` — add `--use_mars`, `--mars_lag`,
  `--mars_mix_coef`, `--mars_lr_scale` argparse args and the
  override lines (matches the pattern of the existing
  `--use_ema_eval` family).

### Identity at step 0

`m_history` is empty for the first `2*lag` (= 20) steps ⇒ correction
term undefined ⇒ `MARSAdamW.step` uses raw `g_t` ⇒ bit-identical to
`torch.optim.AdamW` for the first 20 steps. The trainer
instantiates `MARSAdamW` only when `config.use_mars=True`, so when
the flag is off the entire class is bypassed and the baseline
`AdamW` is used unchanged. After step 20, the lag correction
engages. (At tiny1m3m the run is 92 steps, so steps 20–92 are
MARS-corrected; the model sees ~72 corrected updates out of 92.)

### Run command

```
python train_llm.py --config_class configs.llm_config.Tiny1M3MMARSConfig --device cuda --output_dir ./checkpoints/114-mars
```

Final val loss read from `metrics.json:final_metrics.val_loss` in
`./checkpoints/114-mars/`. The A/B target is the plain
`Tiny1M3MConfig` (val ≈ 6.4306, seed 42). PASS ≤ −0.005; NULL
|Δ| < 0.005; DRIFT > +0.005.
