---
id: 105-retrospective-sparse-attn
status: needs-plan
round: 1
updated: 2026-06-11T01:18:41Z
transfer-risk: med
---

# 105 — Retrospective sparse attention

## Source
Retrospective Sparse Attention for Efficient Long-Context Generation (arXiv:2508.09001).

## Mechanism
Let sparse attention revise earlier outputs when new KV entries arrive, rather than treating each sparse pass as final.

## Scale evidence
The paper targets efficient long-context generation and releases reproducible code. The core promise is lower long-context cost without freezing in approximation errors.

## Why it's worth a slot
This is a different sparse-attention philosophy from static masks or top-p selection. It is about correction, not just pruning.
