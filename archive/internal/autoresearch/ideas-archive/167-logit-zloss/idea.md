---
id: 167-logit-zloss
status: done
round: 1
updated: 2026-06-14T09:37:02Z
transfer-risk: med
plain: Add a small auxiliary penalty that prevents the output logits from growing too large — start the penalty weight at zero so the baseline is unchanged.
---

# 167 — Output Logit Z-Loss (PaLM-Style LogSumExp Penalty)

## Source
- Chowdhery et al. "PaLM: Scaling Language Modeling with Pathways" (arXiv:2204.02311, 2022) — introduced z-loss as `log(Z)^2` where `Z = sum_v exp(logits_v)`, added to the loss with weight `1e-4`. Used in PaLM 540B and many subsequent LMs (LLaMA 2, OLMo 2, Gemma).
- Wortsman et al. "StableLM" and "Gemma" reports — discuss z-loss as a stability / gradient-clipping alternative for very large batch training.
- Closed loss-shape axes: 066-label-smoothing, 067-confidence-penalty, 068-unlikelihood, 069-focal-loss, 070-mtp-head. None of these penalize logit *magnitude* directly.

Distinct from label smoothing (which softens targets) and confidence penalty (which softens predictions). Z-loss is a *logit-magnitude* penalty.

## Mechanism
Add an auxiliary loss that penalizes `log(sum_v exp(logits_v))^2` (a.k.a. `log(Z)^2` where `Z` is the partition function). This prevents the largest logit from growing without bound:
```
z = torch.logsumexp(logits, dim=-1)   # (B, T)
z_loss = (z ** 2).mean()
total_loss = ce_loss + zloss_weight * z_loss
```
At init, `zloss_weight = 0.0` ⇒ the auxiliary loss is 0 ⇒ the total loss is bit-identical to baseline. The optimizer can grow `zloss_weight` to whatever value is best (typically `1e-4` in PaLM). With `zloss_weight=0` init, the lever is a strict no-op at step 0. ~12 LoC; +1 hyperparameter (zloss_weight).

## Design sketch
- **File**: `training/trainer.py` (or wherever the loss is computed) — add `zloss_weight: float = 0.0` config field (default off). When on, after computing `ce_loss`, compute `z = torch.logsumexp(logits, dim=-1).pow(2).mean()` and add `zloss_weight * z` to the loss.
- **Config flag**: `zloss_weight: float = 0.0` (default off on `LLMConfig` or training config). A typical nonzero value is `1e-4` to `1e-2`.
- **Step-0 identity**: `zloss_weight = 0.0` ⇒ auxiliary loss is 0 ⇒ total loss is bit-identical to baseline. Implementer should verify the `z` computation is correct (i.e. `z = log(sum_v exp(logits_v))` not `z = sum_v logits_v`).
- **Intuition**: z-loss prevents the largest logit from growing without bound, which in turn prevents the softmax from becoming a near-delta and the gradients from becoming saturated. At 0.94M / 3M tokens / batch=32, logit explosion is unlikely (the model doesn't have enough capacity or training to blow up the logits), so the lever may be null. But it's cheap, and a null closes the z-loss axis at this tier alongside the other closed loss-shape axes. A win would suggest logit magnitude *is* the binding constraint at 0.94M and z-loss is a cheap regularizer.
- **Why now**: the closed loss-shape axes (066–070) are all on the *target / prediction* side. Z-loss is on the *logit magnitude* side — structurally different. With the rest of the loss-shape family closed, z-loss is the obvious next axis to test.

## Scale evidence
PaLM 540B (Chowdhery et al. 2022) uses z-loss with `1e-4` weight; LLaMA 2, OLMo 2, Gemma all use it. The mechanism is well-validated at ≥100B parameters for training stability. At 0.94M, the typical failure mode z-loss prevents (logit explosion in late training) is unlikely to fire — logit growth is bounded by model capacity. Transfer risk is **med**: well-validated at scale, but the failure mode it targets is largely absent at 0.94M.

## Why it's worth a slot
A win (or even a marginal Δ) would tell us *logit magnitude* is a binding constraint at 0.94M, orthogonal to all five closed loss-shape axes (which target targets/predictions). A null would close the z-loss axis at this tier — logit explosion is not the binding constraint at 0.94M, the model just doesn't have the capacity to drive logits unboundedly. Either result is cheap to obtain and informative.
