---
id: 056-branchnorm
status: rejected
round: 1
updated: 2026-06-10T23:26:24Z
transfer-risk: med
---

# 056 — BranchNorm (training-stage branch rescaling)

## Source
Liu et al., "BranchNorm: Robustly Scaling Extremely Deep Transformers" (arXiv:2305.02790). The paper proposes dynamic rescaling of the non-residual branch over the training period.

## Mechanism
Instead of a fixed residual shrink, apply a training-step schedule `β(t)` to the non-residual branch so early updates are damped and later updates relax toward the ordinary Transformer. This is a simple branch multiplier in front of the attention/FFN output, not a data or LR schedule.

## Scale evidence
The paper reports better stability/performance trade-offs on multiple translation tasks and says BranchNorm is more robust than DeepNorm on key settings like warmup and learning rate. transfer-risk: med — the result is strong but the public evidence is mostly translation, not LM pretraining.

## Why it's worth a slot
BranchNorm separates "need a small early residual" from "need that shrink forever"; if it wins, the schedule is the mechanism, not just the scale constant.
