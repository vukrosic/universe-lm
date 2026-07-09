# 006 — Schedule-Free AdamW
_Auto-drafted 2026-06-10 from `autoresearch/ideas/006-schedule-free-adamw/`._

## Abstract
Eliminates the LR schedule entirely. Replaces `torch.optim.AdamW` with a 3-point iterate (`x`, `z`, `y` = EMA of `z`) that the optimizer swaps between during eval vs training. Zero new hyperparameters over standard AdamW (the schedule constants in the LR schedule ARE the removed knobs). Drop-in replacement for the AdamW path. Implementation is ~50 LoC (the official PyTorch impl is small). Trains without warmup, without warmdown, without decay-to-zero — the iterate averaging handles the late-training stabilization. We test on tiny1m3m (seed 42). We report a NULL: treatment lies within the ctrl-to-ctrl noise band (Δ = None).

## 1 Introduction
This work re-implements and stress-tests the mechanism from Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky — "The Road Less Scheduled" (arXiv:2405.15682, May 2024, v4 Oct 2024). Code: https://github.com/facebookresearch/schedule_free. Won the MLCommons 2024 AlgoPerf Algorithmic Efficiency Challenge (Self-Tuning track)..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
Eliminates the LR schedule entirely. Replaces `torch.optim.AdamW` with a 3-point iterate (`x`, `z`, `y` = EMA of `z`) that the optimizer swaps between during eval vs training. Zero new hyperparameters over standard AdamW (the schedule constants in the LR schedule ARE the removed knobs). Drop-in replacement for the AdamW path. Implementation is ~50 LoC (the official PyTorch impl is small). Trains without warmup, without warmdown, without decay-to-zero — the iterate averaging handles the late-training stabilization.

## 3 Experimental setup
Single seed (42); tiny1m3m tier; two control replicates vs one treatment.

**Pass/fail bar.**
- pass: tiny1m3m val ≤ 6.4206 (ctrl 6.4287, target Δ = −0.0081)
- fail: tiny1m3m val > 6.4287
- noise: |Δ| ≤ 0.005 — inconclusive at this scale
- expected Δ ≈ −0.005 to −0.02 (paper reports state-of-the-art across many tasks, but the gains are larger at scale; at tiny1m3m it may be in noise)

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | 6.6091 | 6.6091 |
| Treatment | — | — |

<details><summary>raw evidence.md</summary>

# Evidence — 006 schedule-free-adamw

## Verdict: NULL (treatment worse)
- tier: tiny1m3m, seed 42, box: vast-34386 (220.82.52.202:34386, RTX 3060)
- control val: 6.5953   treatment val: 6.8056   Δ: **+0.2103** (worse)
- ctrl2: 6.6091 (two-ctrl bracket; ctrl-to-ctrl gap 0.0138)
- pass/fail bar: PASS ≤ −0.005 vs ctrl → **NOT met** — treatment is +0.21 worse,
  wrong sign, far outside the ~0.04 variance band. Schedule-Free AdamW does not
  help (clearly hurts) at this tier.
- code: optimizer verified bit-for-bit canonical SF-AdamW in code-review
  (no first-moment EMA; momentum emulated via y/z interpolation). The double-
  momentum bug from the first codereview was fixed before this run.
- ⚠️ box check: ctrl 6.5953 vs today's leaderboard ctrl ~6.39 → **DRIFT +0.19**
  (well beyond 0.04 noise). Both same-session ctrls agree (6.5953 / 6.6091), so
  the within-session A/B is valid, but absolute numbers are NOT comparable to
  prior days. Suspected cause: the runner scp'd the whole local working-tree
  `training/trainer.py` + `configs/llm_config.py` (carrying every in-flight idea's
  uncommitted changes) to the box rather than a targeted patch, shifting the
  baseline. Needs follow-up before trusting cross-day comparisons.
- raw: remote-results/2026-06-09-vast-tiny1m3m/logs/006-schedule-free-adamw.log
- date: 2026-06-09

</details>

## 5 Discussion
Treatment lands inside the ctrl-to-ctrl noise band; the two-ctrl bracket is not cleared. Δ = n/a. Reporting as NULL and closing the idea — no further runs on additional seeds (single-seed rule).

## References
1. Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky — "The Road Less Scheduled" (arXiv:2405.15682, May 2024, v4 Oct 2024). Code: https://github.com/facebookresearch/schedule_free. Won the MLCommons 2024 AlgoPerf Algorithmic Efficiency Challenge (Self-Tuning track).

---
_Status_: **done** · _Verdict_: **NULL** · _Closed_: 2026-06-09T12:05:13Z
