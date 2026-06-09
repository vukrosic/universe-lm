---
id: 014-sigmoid-loss
status: rejected
round: 1
updated: 2026-06-09T11:36:34Z
---

# 014 — Sigmoid Loss (with z-loss regularization)

## Source
Xi et al., "Lossy CIKM 2023 best paper" — no, more usefully: "Efficient Training of Language Models with Sigmoid Loss" and "Contrastive Entailment Training" line of work (arXiv:2405.18719-adjacent literature, see also Tian et al. 2023 "Predicting vs. Acting"). Sigmoid loss replaces softmax cross-entropy: `L = -log σ(γ·z_t) - Σ_{j≠t} log σ(−γ·z_j)`, often with a z-loss penalty `log_z = log_z·exp(·)` to prevent logit blow-up. Equivalent up to a constant to the Two-tower / ET loss used in the PaLM-2 technical report.

## Mechanism
- Replace `F.cross_entropy` in `training/trainer.py` loss path with a per-token sigmoid loss: positive class on the gold token, negative class on all others, all-in-one pass.
- z-loss: `L_z = 1e-4 · log(mean(exp(z)))` added to the loss to keep logits bounded.
- Distinct from 010 (PolyLoss) which is a Taylor correction to CE; this is a *family-level* swap (sigmoid vs softmax), not a correction to CE.
- Implementation: ~20-30 LoC. No model-shape change, no extra params.

## Why it's worth a slot
- **Distinct from 010 (PolyLoss).** PolyLoss = Taylor correction to CE. Sigmoid loss = swap to a different family. They probe different parts of the loss surface. Running both is informative.
- **Theoretical nice properties**: bounded gradient, no log(0), smoother optimization landscape. ET / sigmoid-style losses are used in PaLM-2, GLaM, and recent open recipes.
- **Identity-safe**: γ init at 1.0; z-loss is a pure regularizer with one hyperparameter (the coefficient, default 1e-4).
- **Risk**: extra hyperparameter (γ, z-loss coef) — small but non-zero.

## Hypothesis
Δ in [−0.005, −0.02] val loss on tiny1m3m. Mechanism: smoother gradient on hard tokens (no log(0) blow-up) + z-loss keeps logits healthy.

## Wiring
- New file or inline: `training/losses.py` — `sigmoid_loss_with_z(logits, targets, gamma=1.0, z_coef=1e-4)`.
- `LLMConfig.use_sigmoid_loss: bool = False` (replaces CE in the trainer's loss path).
- Token ignore_index=-100 must be respected (mask out, then mean over the rest).
- Pass/fail: PASS ≤ −0.005 vs V+q+SWA+HighRoPE ctrl. NULL = |Δ| < 0.005. DRIFT > +0.01.

## Notes
- 007-sigmoid-loss was DELETED in this repo per the git status (`D 007-sigmoid-loss`). The reason isn't in the repo — likely a duplicate / naming. This filing (014) replaces it with a fuller spec. Reviewer: check 007's old contents in git history before rejecting on duplication grounds.
- Eval must stay plain CE (the output-head rule): report sigmoid-loss-train val perplexity via the *standard* CE eval path so the leaderboard stays comparable.
