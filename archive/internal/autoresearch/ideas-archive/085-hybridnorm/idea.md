---
id: 085-hybridnorm
status: needs-plan
round: 1
updated: 2026-06-11T01:17:26Z
transfer-risk: low
---

# 085 — HybridNorm

## Source
HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization (arXiv:2503.04598). 2025.

## Mechanism
Use QKV normalization inside attention and Post-Norm in the FFN so the block gets both stable attention dynamics and a stronger residual-depth signal. The lever is a placement combo, not a new optimizer, and it should fit the repo as a small block wiring change.

## Scale evidence
The paper reports large-scale dense and sparse experiments, including dense models from 151M to 1.2B parameters and token budgets up to 1T, with HybridNorm showing lower training loss and about a 1.4x pretraining convergence speedup in the reported setup. `transfer-risk: low` because the method already wins in the billion-parameter regime.

## Why it's worth a slot
It is a direct competitor to the existing norm ideas in the repo, but with a different placement hypothesis that could still be the right one for the 135M recipe.
