# 001 — Cautious Muon
_Auto-drafted 2026-06-10 from `autoresearch/ideas/001-cautious-muon/`._

## Abstract
One-line sign-mask on the orthogonalized Muon update: zero out components whose sign disagrees with the current gradient. Suppresses stale-momentum artifacts. Bit-identical to baseline when `use_cautious_muon=False`. Applies only to the Muon path; AdamW is unchanged. We test on tiny1m3m (seed 42). We report a NULL: treatment lies within the ctrl-to-ctrl noise band (Δ = 0.006).

## 1 Introduction
This work re-implements and stress-tests the mechanism from Liang et al. 2024, "Cautious Optimizers: Improving Training with One Line of Code" (arXiv 2411.16085)..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
One-line sign-mask on the orthogonalized Muon update: zero out components whose sign disagrees with the current gradient. Suppresses stale-momentum artifacts. Bit-identical to baseline when `use_cautious_muon=False`. Applies only to the Muon path; AdamW is unchanged.

**LR compensation.** The paper's procedure is per-step rescale to pre-mask norm (divide update by mask fraction). A constant bump is a project choice. Our default: `muon_lr` 0.024 → 0.025 (+4%) when `use_cautious_muon=True`. Caller is free to tune or omit.

## 3 Experimental setup
On Kaggle T4, seed 42, tiny1m3m + `use_cautious_muon=True`, `muon_lr=0.025`
(2026-06-08). [[evidence]] — lands after the run finishes.
(Pipeline status lives in the frontmatter above.)

**Pass/fail bar.**
- pass: tiny1m3m val ≤ 6.4206 (ctrl 6.4287, target Δ = −0.0081)
- fail: tiny1m3m val > 6.4287 (worse than control — close the idea)
- noise: |Δ| ≤ 0.005 — below the 2-min tiny1m3m resolution; **inconclusive, not a result** (single-seed rule — do not re-run on another seed)
- expected Δ ≈ −0.005 to −0.02; anything inside ±0.005 is below the noise floor

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | 6.3875, 6.4050, 6.4322, 6.4009 | 6.4064 |
| Treatment | 6.4125, 6.4156 | 6.4140 |

<details><summary>raw evidence.md</summary>

# Evidence — 001 cautious-muon

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast 220.82.52.202:34386 (RTX 3060)
- treatment val: 6.4125 (r1), 6.4156 (r2)  — n=2
- control: ctrl/ctrl2 over two batches = 6.3875 / 6.4050 / 6.4322 / 6.4009 (mean 6.4064)
- Δ vs mean ctrl: +0.006 (wrong sign — treatment is slightly *worse*)
- pass/fail bar: needed ≤ ~-0.01 vs ctrl → not met
- box check: ctrl spread 6.3875–6.4322 (~0.045, flash-kernel nondeterminism); within expected
- raw: remote-results/2026-06-09-vast-tiny1m3m/arq-r1 + arq-r2 (results + logs)
- date: 2026-06-09

Note: an earlier orphan sweep saw cautious-Muon at Δ-0.0249/-0.0094; with a proper
two-ctrl bracket it lands inside noise (and slightly negative). Clean null.

</details>

## 5 Discussion
Treatment lands inside the ctrl-to-ctrl noise band; the two-ctrl bracket is not cleared. Δ = 0.006. Reporting as NULL and closing the idea — no further runs on additional seeds (single-seed rule).

## References
1. Liang et al. 2024, "Cautious Optimizers: Improving Training with One Line of Code" (arXiv 2411.16085).

---
_Status_: **done** · _Verdict_: **NULL** · _Closed_: 2026-06-09T09:36:28Z
