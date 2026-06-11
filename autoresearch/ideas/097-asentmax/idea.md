---
id: 097-asentmax
status: needs-plan
round: 1
updated: 2026-06-11T01:18:16Z
transfer-risk: med
---

# 097 — ASEntmax long-context sparse attention

## Source
Vasylenko, Treviso, and Martins, "Long-Context Generalization with Sparse Attention" (arXiv:2506.16640).

## Mechanism
Replace softmax with alpha-entmax and make its temperature adaptive to context size and content. The result is sparse support when the model should focus and softer support when it should not.

## Scale evidence
The paper reports strong long-context generalization and very long extrapolation on associative recall, including 95.3% accuracy at 65K after training on 64 tokens. That is a clean "sparse beats dense when length grows" result.

## Why it's worth a slot
This is a better sparse-attention bet than fixed pattern sparsity alone. It learns when to be sparse instead of hard-coding a ring or stride pattern.
