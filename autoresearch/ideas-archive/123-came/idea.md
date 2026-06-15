---
id: 123-came
status: done
round: 1
updated: 2026-06-13T20:34:34Z
transfer-risk: med
plain: It's like AdamW but it down-weights the steps that it's least confident about — so noisy gradients don't get the same loud update as the consistent, repeated ones.
---

# 123 — CAME: Confidence-guided Adaptive Memory Efficient Optimization

## Source
Luo, Xu, Xu, Liu, Wang, Wang, Wang, Zhang, "CAME: Confidence-guided
Adaptive Memory Efficient Optimization" (arXiv:2307.02085, NeurIPS
2023). https://arxiv.org/abs/2307.02085

Validated on BERT-base pretraining, T5-base, GPT-2 small/medium
(125M-350M), and on a 3B-parameter Chinese LLM (Cerebras-GPT-style).
The lever is a memory-efficient Adam variant with a *confidence
rescaling* that addresses the noise sensitivity of Adam's second moment.

## Mechanism
Adam's update: `update = m_t / (√v_t + ε)` where `v_t` is the EMA of
`g_t²`. CAME replaces this with a *corrected* confidence-aware version:

  `m_t = β1 · m_{t−1} + (1 − β1) · g_t`          (momentum)
  `v_t = β2 · v_{t−1} + (1 − β2) · g_t²`         (Adam 2nd moment)
  `res_t = (m_t − g_t) / (√v_t + ε)`             (residual: how far
                                                  momentum drifted
                                                  from current grad)
  `confidence_t = max(res_t, 0) + ε`              (clip negative
                                                  residuals to 0)
  `update = m_t / (√v_t + ε) · confidence_t / (m_t + ε)`     (rescale)
  `w ← w − lr · update`

The intuition: when the gradient direction `g_t` agrees with the
momentum direction `m_t`, the residual `res_t` is small and
`confidence ≈ 1` (no rescaling). When they disagree (noisy gradient
that doesn't agree with the running estimate), the residual is
large and the update is *down-weighted*. This is similar to Lion's
"sign agreement" idea but uses the *magnitude* of disagreement
instead of a binary sign check.

Memory: CAME also supports a *non-negative factorization* of `v_t`
(paper section 3.2) similar to Adafactor's `R_t · C_t` factorization,
giving O(d) memory instead of O(d²). At 0.94M this is moot (full
`v_t` fits trivially).

**Identity at step 0**: with `m_0 = 0, v_0 = 0`, the first step
has `m_0 = (1−β1)·g_0`, `v_0 = (1−β2)·g_0²`, residual `res_0 =
(m_0 − g_0)/(√v_0 + ε) = (−β1·g_0)/(√(1−β2)·|g_0| + ε) ≈ −β1/√(1−β2)`.
This is *negative* (momentum has only seen one gradient, so it
*disagrees* with `g_0` by construction), so `confidence_0 = max(res_0, 0) + ε = ε`.
The first step has `update ≈ 0`, so **no update at step 0**.

The paper handles this by warmstarting `m_0 = g_0` (one-step
momentum init) so the first step's residual is `≈ 0` and
`confidence ≈ 1`. With warm-start, the first step is approximately
equal to AdamW's first step.

## Design sketch
- `optimizers/came.py` (new): `CAME` — `torch.optim.Optimizer`
  subclass implementing the confidence-rescaled update. State per
  param: `exp_avg` (m), `exp_avg_sq` (v). ~70 LoC.
- `training/trainer.py`: when `use_came=True`, route the
  AdamW-eligible params through `CAME`. The 2-D slot still uses
  Muon. ~10 LoC.
- `configs/llm_config.py`: add `use_came: bool = False`,
  `came_lr: float = 0.006`, `came_beta1: float = 0.9`,
  `came_beta2: float = 0.999`, `came_eps: float = 1e-8`. ~10 LoC.
- LoC: ~90 total (under 200 ceiling).
- Identity at step 0: with `m_0 = g_0` warm-start, the first step
  has `res_0 ≈ 0, confidence_0 ≈ 1`, so the update is approximately
  equal to AdamW's first step. Bit-identical to AdamW at step 0
  *with* the warm-start; without warm-start, the first step has
  near-zero magnitude (a bug, not a feature).
- The intuition: at 0.94M with 92 steps, AdamW's second-moment
  estimate `v_t` is biased by the few samples seen so far — early
  gradients can dominate `v_t` and produce noisy steps. CAME's
  confidence rescaling down-weights updates whose momentum disagrees
  with the current gradient, which is exactly the "early gradient
  disagrees with current" case. A null would say "at 0.94M the
  small-sample bias is negligible"; a win would say "the
  confidence rescaling reduces noise-induced variance in the
  update direction".

## Scale evidence
- arXiv:2307.02085 (Luo et al. 2023): validates on BERT-base
  pretraining (GLUE), T5-base, GPT-2 small/medium (~125M-350M),
  and on a 3B Chinese LLM. Reports parity-to-better vs AdamW at
  matched compute, with stronger gains on the smaller models.
- Transfer risk: **med**. Validated at ≥100M (BERT-base 110M,
  T5-base 220M, GPT-2 small 125M, GPT-2 medium 350M, plus 3B),
  the mechanism is scale-free. The memory-efficiency side is
  moot at 0.94M but the confidence-rescaling side is *exactly*
  the lever we're testing.

## Why it's worth a slot
CAME is the only Adam-variant filed that addresses the *small-sample
bias of v_t* explicitly (other Adam-variants in the closed wave
031-040 changed the *shape* of v_t or the LR schedule; CAME changes
the *update's magnitude* based on momentum agreement). It is the
cleanest test of "is AdamW's second moment too noisy at tiny scale
for confidence-rescaling to help?" — a question our 92-step run
window *uniquely* surfaces (longer runs don't see the noise because
v_t averages out). A win would say "small-sample noise in v_t is
load-bearing at 0.94M"; a null would say "at 0.94M AdamW's noise
is benign and the confidence rescaling adds compute for no gain".

## Plan

**Re-code (round 1, after 2026-06-13 GPU blowup):** the v1
implementation diverged (val 10.81 → 6.79e7 at step 25 → 1.06e8 →
46267 final). Root cause: the CAME update `m̂ / denom · conf /
|m̂|` is **unbounded** when `denom ≈ ε` (i.e., `v̂ ≈ 0`): a
single step with tiny `v̂` and non-trivial `m̂` produces
`update ≈ m̂ / ε² ≈ 1e16` magnitude. Also no NaN/Inf guard — one
non-finite grad poisons `m, v` forever. Also `came_lr=0.006` is too
aggressive at tiny1m3m's small-sample `v` noise regime. Fix:
1. NaN/Inf guard on `grad` and existing `m, v` buffers before
   applying update → skip the parameter (don't update `m, v, p`)
   on a non-finite input. Defensive only; should never fire on a
   healthy training trajectory.
2. Magnitude clip on `update` (default `update_clip=10.0`) → bounds
   any single step's per-element displacement to ±10·lr. The
   confidence factor can otherwise rescale `m̂/denom` beyond ±1.
3. `Tiny1M3MCAMEConfig.came_lr = 0.001` (was 0.006) — 6× lower than
   paper default; the paper scales up to 3B where `v̂` noise
   averages out. At 0.94M / 92-step we have small-sample `v̂` and
   need the more conservative LR.

**Files (round 1):**
- `optimizers/came.py` — add `update_clip` constructor arg + per-group
  field; add finiteness guard before update; clamp `update` to
  `[-update_clip, update_clip]` before applying. ~12 LoC added
  (147 → ~160 total).
- `configs/llm_config.py`:
  - Add `came_update_clip: float = 10.0` to `LLMConfig` (~3 LoC).
  - Lower `Tiny1M3MCAMEConfig.came_lr` from `0.006` to `0.001`
    (~1 LoC).
- `training/trainer.py` — pass `update_clip=getattr(config,
  "came_update_clip", 10.0)` to the `CAME(...)` constructor (~2 LoC).
- `autoresearch/ideas/123-came/idea.md` — update this Plan section.

**Total LoC added: ~18 (well under the 200 ceiling).**

**Step-0 byte-identical contract:** the NaN/Inf guard is a no-op on
finite grads (the normal case at step 0); the update-magnitude
clip threshold of 10.0 is far above the natural step-0 update
magnitude (~1e-6), so it never fires at step 0. With
`use_came=False` (default) plain `torch.optim.AdamW` is used —
baseline path bit-identical.

The step-0 identity argument from the v1 plan still holds: at step 1
with bias correction, `m̂_1 = g_0`, `v̂_1 = g_0²`, `res_1 = 0`,
`conf_1 = ε`, so `update ≈ sign(g_0) · ε · |g_0| / (|g_0|+ε)² ≈
1e-6 · sign(g_0)` (vanishingly small, well under the new clip).
No guard or clip fires ⇒ output bit-identical to v1's step 0.

**Run command:**
```bash
python train_llm.py --config_class configs.llm_config.Tiny1M3MCAMEConfig \
    --output_dir runs/123-came --seed 42
```

**Read final val_loss:** `metrics.json` → `final_metrics.val_loss`
(written to `runs/123-came/metrics.json` and `plots/metrics_*.json`).
A/B vs the tiny1m3m ctrl (`Tiny1M3MConfig`, val 6.4306). PASS bar:
val_loss ≤ ctrl − 0.005 (small/null band — the lever is per-parameter
update *magnitude* on a 92-step trajectory where AdamW's `v` noise
floor is already partially damped). NULL band |Δ| < 0.005.
DRIFT > +0.005.
