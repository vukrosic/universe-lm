---
id: 067-confidence-penalty
status: needs-run
round: 1
updated: 2026-06-11T01:41:35Z
transfer-risk: low
---

# 067 — Confidence penalty

## Source
Pereyra et al., "Regularizing Neural Networks by Penalizing Confident Output Distributions" (arXiv:1701.06548). Jan 2017.

## Mechanism
Add an entropy bonus on the token distribution, equivalent to penalizing low-entropy logits: `loss = CE - λ * H(p)`. It pushes the model away from pathological peaking without changing the architecture or the tokenization path. The gradient w.r.t. logits hits **all V positions** (∂H/∂z_i = p_i · (log p_i + 1)), in contrast to a gold-only softener like PolyLoss whose gradient lands on a single logit.

## Scale evidence
The paper evaluates language modeling, machine translation, and speech recognition and reports stronger generalization from smoother output distributions. `transfer-risk: low` because the mechanism is a direct output-head regularizer with demonstrated NLP use, and the same `−λ·H(softmax(logits))` form has been re-validated on modern LMs in 2024–25 entropy-regularization work.

## Why it's worth a slot
**Sub-question (the only one this slot answers):** does *symmetric-across-vocab* entropy pressure beat *gold-only* soften at tiny1m3m? 010-Polyloss's `(1 − p_t)` factor closed null inside the ~0.005 noise floor (closed.md:43, trt 6.5938 vs ctrls 6.5991/6.6050, Δ −0.0053 < ctrl-gap 0.0059). PolyLoss's gradient lands on the gold logit only; ConfPenalty's gradient lands on **all V logits** with magnitude ~V× larger per step, so the pressure signal survives the 92-step tiny1m3m run even when individual per-logit magnitudes are small.

**Crisp bet:** with β = 0.01 (the pinned default in `configs/output_head_ablations.py:38`), ConfPenalty should land **Δ ≈ −0.01 to −0.02** vs the matching ctrl at tiny1m3m seed 42. Mechanism: per-step ∇ magnitude scales as `V · β · p_i · (1 + log p_i)` summed over V ≈ 4096 vocab positions vs PolyLoss's single-logit `(1 − p_t)` term, so the *aggregate* softness pressure is ~V× stronger even at smaller per-element β. The softmax doesn't change (so architecture is bit-identical), the eval is plain CE (train-side aux only, per `training/trainer.py:474-485, 526-537`), and a null at this magnitude is *informative* — it closes the entire soft-output family (067 + 069-focal-loss; 068-unlikelihood never landed) at this tier, independent of which logit the softener touches.

**Lever is already pre-wired** in `configs/output_head_ablations.py:36` (`Tiny1M3MConfPenaltyConfig(β=0.01)`) and consumed by `training/trainer.py:474-485, 526-537`. This is ≈0 LoC of new code — the cheapness is itself a leverage argument. No prior A/B appears in closed.md or LEADERBOARD for the OH3 ConfPenalty anchor, so this slot also closes the unbenchmarked config.

**Head-to-head framing:** treat the run as adjudicating 010's null. If 067 wins with Δ ≤ −0.01, the family survives via the entropy form (carry it forward; drop 010 as a gold-only special case). If 067 also closes null, soft-output knobs are dead at tiny1m3m regardless of pressure geometry — the next lever to try is *not* another output-softener, and the slot pays for itself by killing a crowded branch.
