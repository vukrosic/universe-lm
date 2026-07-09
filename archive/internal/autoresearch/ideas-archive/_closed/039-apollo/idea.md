---
id: 039-apollo
status: rejected
round: 1
updated: 2026-06-10T16:44:51Z
transfer-risk: low
---

# 039 - APOLLO

## Source
APOLLO: SGD-like Memory, AdamW-level Performance (arXiv:2412.05270, 2024).

## Mechanism
Approximate AdamW's learning-rate scaling with a structured low-rank state built from random projection, so the optimizer keeps some adaptivity without full per-parameter second moments. In repo terms this can replace AdamW on the matrix-weight path while leaving 1D parameters alone.

## Scale evidence
The paper reports LLaMA-7B and LLaMA-13B results, plus throughput gains from the lower-memory optimizer state. transfer-risk: low - this is direct large-scale LM pretraining evidence.

## Why it's worth a slot
This is the modern low-rank optimizer family in its simplest form, and a tiny-model null would tell us that the low-rank structure is only useful once the optimizer state itself becomes the bottleneck.
