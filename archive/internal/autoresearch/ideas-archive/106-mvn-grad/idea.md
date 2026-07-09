---
id: 106-mvn-grad
status: needs-plan
round: 1
updated: 2026-06-11T01:18:41Z
transfer-risk: low
---

# 106 — MVN-Grad

## Source
Patitucci and Mokhtari, "Adaptive Optimization via Momentum on Variance-Normalized Gradients" (arXiv:2602.10204).

## Mechanism
Normalize gradients by their estimated variance, then apply momentum after normalization instead of before it. The key idea is to decouple stale momentum from the stochastic normalizer.

## Scale evidence
The paper reports better stability and generalization on CIFAR-100 and GPT-style language modeling, with no extra overhead and bounded response to gradient spikes.

## Why it's worth a slot
This is a strong candidate when we want the Adam family to behave better on noisy updates without paying for a heavy second-order method. It is also a clean contrast to Muon-style orthogonalization.
