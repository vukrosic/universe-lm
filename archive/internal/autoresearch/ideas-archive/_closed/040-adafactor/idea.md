---
id: 040-adafactor
status: rejected
round: 1
updated: 2026-06-10T16:46:38Z
transfer-risk: med
---

# 040 - Adafactor

## Source
Adafactor: Adaptive Learning Rates with Sublinear Memory Cost (arXiv:1804.04235, 2018).

## Mechanism
Factor the second-moment estimator into per-row and per-column statistics, add update clipping, and optionally scale updates relative to parameter magnitude. In this repo the clean version is Adafactor on matrix weights, with the scalar and norm path left unchanged.

## Scale evidence
The paper's core evidence is Transformer translation rather than modern LLM pretraining, but the optimizer is widely used in large sequence models and is explicitly designed for memory efficiency. transfer-risk: med - the mechanism is strong, but the source evidence is less direct for our exact setting than the newer LM-specific optimizers above.

## Why it's worth a slot
Adafactor asks a different question from Adam-mini, LAMB, or APOLLO: if the full second-moment tensor is overkill, do we get the same gain just from row/column factorization plus clipping?
