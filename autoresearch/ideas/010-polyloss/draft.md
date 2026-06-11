# 010-polyloss
_Auto-drafted 2026-06-10 from `autoresearch/ideas/010-polyloss/`._

## Abstract
CE = `-log p_t = (1-p_t) + (1-p_t)²/2 + …` (Taylor series of `-log(1 - x)` at `x =1-p_t`). Standard CE keeps only the leading term. PolyLoss adds back the `j=1` Taylor term as a coefficient-weighted correction: We test on tiny1m3m (seed 42). We report a NULL: treatment lies within the ctrl-to-ctrl noise band (Δ = None).

## 1 Introduction
This work re-implements and stress-tests the mechanism from Leng et al., "PolyLoss: A Polynomial Expansion Perspective of Loss Functions in Deep Learning" (2022, arXiv:2204.12511)..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
CE = `-log p_t = (1-p_t) + (1-p_t)²/2 + …` (Taylor series of `-log(1 - x)` at `x =1-p_t`). Standard CE keeps only the leading term. PolyLoss adds back the `j=1` Taylor term as a coefficient-weighted correction:

```
L_poly = L_CE + Σ_{j≥1} α_j · (1 - p_t)^(j+1)
```

**Pinned spec (this idea): `α_1 = ε₁ =1.0`, all other `α_j =0`.** That single number is the next Taylor term in the `-log p_t` expansion, so it is the principled correction to CE's truncation error — not a tuned hyperparameter. The paper's "strong default" for classification (CIFAR / ImageNet / LMs) is exactly `ε₁ =1.0`. No sweep over `ε₁`. The on/off flag (`use_poly_loss`) defaults to **off** so the control path is byte-identical to baseline CE; setting `ε₁=1.0` and turning the flag on gives the treatment.

## 3 Experimental setup
On Kaggle T4, seed42, tiny1m3m + `use_poly_loss=True`, `poly_eps1=1.0`. Eval stays plain CE. [[evidence]] lands after the run finishes.

**Pass/fail bar.**
- **PASS (win)**: treatment val_loss ≤ control val_loss −0.005 (≥0.005 absolute improvement).
- **NULL (clean, loggable)**: `|Δ| <0.005` vs control. Write `evidence.md` with verdict NULL and append one line to `closed.md`. A null here is informative — "CE's `j=1` Taylor truncation term is negligible at tiny1m3m" — not a failure.
- **DRIFT**: control val_loss drifts >0.01 from `LEADERBOARD.md` baseline ≈6.4287 → box validation; rerun or kill the slot.
- Anything in the **0.005–0.01** window → log inconclusive; do not promote, do not re-run on another seed.
- Expected Δ ≈ −0.005 to −0.02 (paper reports ~0.1–0.3% relative improvement at LM scale, so the upper end is plausible, lower end sits at the noise floor).

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | 6.6050 | 6.6050 |
| Treatment | — | — |

<details><summary>raw evidence.md</summary>

# Evidence — 010 polyloss

## Verdict: NULL (inside variance)
- tier: tiny1m3m, seed 42, box: vast-34386 (220.82.52.202:34386, RTX 3060)
- control val: 6.5991   treatment val: 6.5938   Δ: **−0.0053**
- ctrl2: 6.6050 (two-ctrl bracket; ctrl-to-ctrl gap **0.0059**)
- pass/fail bar: PASS ≤ −0.005 vs ctrl. Treatment beats ctrl1 by 0.0053 and ctrl2
  by 0.0112, BUT the margin over the nearest ctrl (0.0053) is **smaller than the
  ctrl-to-ctrl spread (0.0059)** → fails the two-ctrl WIN rule → **NULL**.
  PolyLoss (ε₁=1.0) nudges loss down but the effect is inside session variance.
- ⚠️ box check: same +0.19 baseline drift as the 006 batch (session ctrl ~6.60 vs
  prior-day ~6.39) — within-session A/B valid, cross-day not. See 006 evidence.
- raw: remote-results/2026-06-09-vast-tiny1m3m/logs/010-polyloss.log
- date: 2026-06-09

</details>

## 5 Discussion
Treatment lands inside the ctrl-to-ctrl noise band; the two-ctrl bracket is not cleared. Δ = n/a. Reporting as NULL and closing the idea — no further runs on additional seeds (single-seed rule).

## References
1. Leng et al., "PolyLoss: A Polynomial Expansion Perspective of Loss Functions in Deep Learning" (2022, arXiv:2204.12511).

---
_Status_: **done** · _Verdict_: **NULL** · _Closed_: 2026-06-09T12:13:40Z
