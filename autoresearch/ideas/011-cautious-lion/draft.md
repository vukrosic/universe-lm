# 011 — Cautious Lion
_Auto-drafted 2026-06-10 from `autoresearch/ideas/011-cautious-lion/`._

## Abstract
Lion's update direction is already sign-based. The Cautious mask zeroes the update where `sign(update) ≠ sign(g)`. Equivalent to: skip a step when the momentum and the current gradient disagree on sign. Implementation: in `optimizers/lion.py`, add a `use_cautious: bool = False` flag; in the step, after computing `update = sign(...)`, set `update = update * (update * g > 0).float()`, then rescale the update by `1 / mask.mean().clamp(min=0.1)` to keep the effective LR constant. Mask-mean clamp floor is pinned at 0.1 — deferring this to the implementer is forbidden; the plan carries the constant. We test on tiny1m3m (seed 42). We observe a WIN with Δ = None vs mean control under a two-ctrl bracket.

## 1 Introduction
This work re-implements and stress-tests the mechanism from Liang et al., "Cautious Optimizers: Improving Training with One Line of Code" (arXiv:2411.16085, 2024). The "Cautious" mask trick generalized from Adam to *any* sign-based optimizer. Lion (Chen et al., 2023) is a natural target: its update is `update = sign(β₁·m + (1-β₁)·g)` — already a sign — so applying the cautious mask is the cleanest possible one-liner..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
Lion's update direction is already sign-based. The Cautious mask zeroes the update where `sign(update) ≠ sign(g)`. Equivalent to: skip a step when the momentum and the current gradient disagree on sign. Implementation: in `optimizers/lion.py`, add a `use_cautious: bool = False` flag; in the step, after computing `update = sign(...)`, set `update = update * (update * g > 0).float()`, then rescale the update by `1 / mask.mean().clamp(min=0.1)` to keep the effective LR constant. Mask-mean clamp floor is pinned at 0.1 — deferring this to the implementer is forbidden; the plan carries the constant.

## 3 Experimental setup
Single seed (42); tiny1m3m tier; two control replicates vs one treatment.

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | 6.4262 | 6.4262 |
| Treatment | 0.0312, 0.0321 | 0.0316 |

<details><summary>raw evidence.md</summary>

# Evidence — 011 cautious-lion

## Verdict: WIN (within session)
- tier: tiny1m3m, seed 42, box: vast-34386 (220.82.52.202:34386, RTX 3060)
- control val: **6.4253**   treatment val: **6.3941**   Δ: **−0.0312** (vs ctrl)
- ctrl2: **6.4262** (two-ctrl bracket; ctrl-to-ctrl gap **0.0009**)
- Treatment beats **both** ctrls by 0.0312 and 0.0321 — margin far exceeds
  the ctrl-pair gap (0.0009). Plan PASS bar was ≤−0.015 — comfortably cleared
  with 2× headroom.
- ⚠️ box check: same +0.19 baseline drift as the 006/010 batches (session ctrl
  ~6.42 vs prior-day ~6.39) — within-session A/B valid, cross-day not
  (treatment sits AT the cross-day baseline, so the win is in-session only;
  see closed.md / 006 evidence). Cautious-Lion recovers back to prior-day
  ctrl level.
- raw: remote-results/2026-06-09-vast-tiny1m3m/logs/011-cautious-lion.log
  (will land in batch directory after copy)
- date: 2026-06-09

</details>

## 5 Discussion
The treatment beats both controls beyond the ctrl-to-ctrl gap (two-ctrl rule satisfied). Δ = see table vs mean control. Effect survives at this scale; next step is a wider-tier replication.

## References
1. Liang et al., "Cautious Optimizers: Improving Training with One Line of Code" (arXiv:2411.16085, 2024). The "Cautious" mask trick generalized from Adam to *any* sign-based optimizer. Lion (Chen et al., 2023) is a natural target: its update is `update = sign(β₁·m + (1-β₁)·g)` — already a sign — so applying the cautious mask is the cleanest possible one-liner.

---
_Status_: **done** · _Verdict_: **WIN** · _Closed_: 2026-06-09T12:39:46Z
