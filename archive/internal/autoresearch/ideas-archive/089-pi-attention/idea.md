---
id: 089-pi-attention
status: needs-plan
round: 1
updated: 2026-06-11T01:17:52Z
transfer-risk: med
---

# 089 — Pi Attention

## Source
Pi-Attention: Periodic Sparse Transformers for Efficient Long-Context Modeling (arXiv:2511.10696). 2025.

## Mechanism
Combine ring-local attention with periodic stride skips and a learned fusion gate so long-range coverage grows predictably without a dense quadratic path. The key lever is the periodic skip/fusion pattern, which should be implementable as a sparse attention schedule plus a small gate.

## Scale evidence
The paper reports language modeling, retrieval, and vision-language experiments and says the method matches or surpasses dense attention quality with 8.3% lower perplexity than RingAttention while using 50% fewer GPUs for the same context length. `transfer-risk: med` because the results are strong but the main gain is long-context efficiency rather than plain short-context loss.

## Why it's worth a slot
It gives us a concrete sparse-attention family that is more structured than generic sparsity and could be worth carrying into the 135M recipe if the tiny run shows a real signal.
