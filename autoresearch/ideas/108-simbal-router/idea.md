---
id: 108-simbal-router
status: rejected
round: 1
updated: 2026-06-13T20:19:25Z
transfer-risk: low
plain: It tries to make similar inputs pick similar experts instead of bouncing around and wasting capacity.
---

# 108 — SIMBAL Router Orthogonality

## Source
Load Balancing Mixture of Experts with Similarity Preserving Routers, arXiv:2506.14038 https://arxiv.org/abs/2506.14038

## Mechanism
Add a router regularizer that softly pulls the router weight Gram matrix toward the identity, so the router preserves token similarity while still balancing expert load. Make the regularizer weight start at zero so the model begins as the plain router.

## Scale evidence
The paper trains a 762M-active / 3.14B-total MoE-L model and reports lower perplexity plus faster convergence, which makes this a low-risk routing lever at our scale.

## Why it's worth a slot
This tests whether router geometry itself is the useful ingredient, not just uniform load balancing; a null would narrow the MoE search to non-orthogonal balancing rules.
