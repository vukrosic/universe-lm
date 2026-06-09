---
id:010-polyloss
status: needs-review
round: 2
updated: 2026-06-09T09:41:26Z
---

#010 — PolyLoss (polynomial expansion of cross-entropy)

## Source
Leng et al., "PolyLoss: A Polynomial Expansion Perspective of Loss Functions in Deep Learning" (2022, arXiv:2204.12511).

## Mechanism
CE = `-log p_t = (1-p_t) + (1-p_t)²/2 + …` (Taylor series of `-log(1 - x)` at `x =1-p_t`). Standard CE keeps only the leading term. PolyLoss adds back the `j=1` Taylor term as a coefficient-weighted correction:

```
L_poly = L_CE + Σ_{j≥1} α_j · (1 - p_t)^(j+1)
```

**Pinned spec (this idea): `α_1 = ε₁ =1.0`, all other `α_j =0`.** That single number is the next Taylor term in the `-log p_t` expansion, so it is the principled correction to CE's truncation error — not a tuned hyperparameter. The paper's "strong default" for classification (CIFAR / ImageNet / LMs) is exactly `ε₁ =1.0`. No sweep over `ε₁`. The on/off flag (`use_poly_loss`) defaults to **off** so the control path is byte-identical to baseline CE; setting `ε₁=1.0` and turning the flag on gives the treatment.

## Pass / fail bar
- **PASS (win)**: treatment val_loss ≤ control val_loss −0.005 (≥0.005 absolute improvement).
- **NULL (clean, loggable)**: `|Δ| <0.005` vs control. Write `evidence.md` with verdict NULL and append one line to `closed.md`. A null here is informative — "CE's `j=1` Taylor truncation term is negligible at tiny1m3m" — not a failure.
- **DRIFT**: control val_loss drifts >0.01 from `LEADERBOARD.md` baseline ≈6.4287 → box validation; rerun or kill the slot.
- Anything in the **0.005–0.01** window → log inconclusive; do not promote, do not re-run on another seed.
- Expected Δ ≈ −0.005 to −0.02 (paper reports ~0.1–0.3% relative improvement at LM scale, so the upper end is plausible, lower end sits at the noise floor).

## Reporting rule — train-only
The PolyLoss term is a **loss-shape change**, not a logit op, so it follows the same reporting rule as the existing loss-side aux family in `configs/output_head_ablations.py` (ZLoss / LabelSmooth / ConfPenalty): apply the correction in the **train** path only; **leave `training/evaluation.py:53` on plain CE** so the reported `val_loss` is directly comparable to the leaderboard. Concretely:
- `training/trainer.py:372-377` (forward loss) — add `+ ε₁ · (1 - p_t)` behind the `use_poly_loss` guard.
- `training/trainer.py:403-408` (AMP / alternative forward loss) — same guard.
- `training/evaluation.py:53` — **no change** (plain CE).

## Mask handling — must mirror the existing CE path
`trainer.py:368-370` builds `shift_labels = y[:,1:]` with the last token set to `-100` (and the non-AMP path does the same). The PolyLoss term `ε₁ · (1 - p_t)` must be masked with the same `ignore_index=-100` positions — otherwise the correction leaks into the last token of every sequence and silently biases the gradient. Spec rule:

```
mask = (shift_labels != -100).float()
p_t = softmax(logits)[..., shift_labels.clamp(min=0)]   # index safely; -100 ->0 harmlessly
poly_term = ε₁ * (1 - p_t) * mask
```

Equivalent form: compute `p_t` only at positions where `shift_labels != -100`, or multiply the per-position correction by `(shift_labels != -100).float()`. Either form is acceptable; the spec's hard requirement is that the `-100` positions contribute zero to the loss and zero to the gradient.

## Diff sites (LoC budget)
- `training/trainer.py:372-377` — +3-5 LoC behind `if getattr(config, "use_poly_loss", False):`. Base path byte-identical.
- `training/trainer.py:403-408` — +3-5 LoC, same guard.
- `training/evaluation.py:53` — **0 LoC** (plain CE stays).
- `configs/output_head_ablations.py` — +2 LoC: append `Tiny1M3MPolyLossConfig(LLMConfig)` with `use_poly_loss: bool = False, poly_eps1: float =1.0` next to the LabelSmooth anchor. Flag wires through the existing runner arg-pass path.
- Total ≈8–12 LoC, well under the200-LoC cap.

## Disambiguation vs LabelSmooth (frame honestly)
PolyLoss and LabelSmooth are not mathematically identical — PolyLoss's `(1 - p_t)` is a function of the model's own gold probability, LabelSmooth is a fixed Dirichlet prior over vocab. But at tiny1m3m the val-loss delta vs plain CE is unlikely to be large enough to separate the two empirically. The result should be framed as **"the `j=1` polynomial correction is X better (or not) than plain CE"**, not **"PolyLoss is X better than LabelSmooth."** If the result is NULL, the writeup acknowledges that a single-tier sweep at this scale cannot separate the two — the slot still informs whether the `j=1` term matters, even if it cannot adjudicate between two compatible mechanisms.

## Why it's worth a slot
Orthogonal to007-sigmoid-loss's bounded-gradient story: PolyLoss is the principled **loss-side** correction to CE's Taylor truncation error, not a gradient-side bound. They probe different parts of the loss surface, so running both is informative even if007 is closed.3-5 LoC, no model-shape change, no compute cost. Identity-safe at step0 (flag is the switch, `ε₁=0` ≡ baseline behavior). Transferable across scale (loss-only). Tier: **tiny1m3m / seed42.**

## Run notes
On Kaggle T4, seed42, tiny1m3m + `use_poly_loss=True`, `poly_eps1=1.0`. Eval stays plain CE. [[evidence]] lands after the run finishes.
