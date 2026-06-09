---
id: 001-cautious-muon
status: done
round: 1
updated: 2026-06-09T09:36:28Z
---

# 001 — Cautious Muon

## Source
Liang et al. 2024, "Cautious Optimizers: Improving Training with One Line of Code" (arXiv 2411.16085).

## Mechanism
One-line sign-mask on the orthogonalized Muon update: zero out components whose sign disagrees with the current gradient. Suppresses stale-momentum artifacts. Bit-identical to baseline when `use_cautious_muon=False`. Applies only to the Muon path; AdamW is unchanged.

## Pass / fail bar
- pass: tiny1m3m val ≤ 6.4206 (ctrl 6.4287, target Δ = −0.0081)
- fail: tiny1m3m val > 6.4287 (worse than control — close the idea)
- noise: |Δ| ≤ 0.005 — below the 2-min tiny1m3m resolution; **inconclusive, not a result** (single-seed rule — do not re-run on another seed)
- expected Δ ≈ −0.005 to −0.02; anything inside ±0.005 is below the noise floor

## LR compensation (project-specific, not from the paper)
The paper's procedure is per-step rescale to pre-mask norm (divide update by mask fraction). A constant bump is a project choice. Our default: `muon_lr` 0.024 → 0.025 (+4%) when `use_cautious_muon=True`. Caller is free to tune or omit.

## Seed-sensitivity caveat
Liang et al. report the largest gains in the small-batch / short-run regime. tiny1m3m IS that regime, so this is the right place to test. A follow-up at screen20m is NOT in that regime and may show a much smaller or null effect — treat any screen20m result with caution.

## Run notes
On Kaggle T4, seed 42, tiny1m3m + `use_cautious_muon=True`, `muon_lr=0.025`
(2026-06-08). [[evidence]] — lands after the run finishes.
(Pipeline status lives in the frontmatter above.)
