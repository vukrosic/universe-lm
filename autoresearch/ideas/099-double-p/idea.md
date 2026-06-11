---
id: 099-double-p
status: needs-plan
round: 1
updated: 2026-06-11T01:18:17Z
transfer-risk: med
---

# 099 — Double-P hierarchical top-p sparse attention

## Source
Ni et al., "Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs" (arXiv:2602.05191).

## Mechanism
Use a two-stage top-p sparse attention scheme: first coarse cluster-level selection, then token-level refinement only where the coarse pass says it matters.

## Scale evidence
The paper reports near-zero accuracy drop, up to 1.8x less attention computation, and up to 1.3x end-to-end decoding speedup over fixed-budget sparse attention. That is a solid efficiency result, not just an accuracy claim.

## Why it's worth a slot
This is a different sparse-attention philosophy from ring/stride patterns. It preserves attention mass and then spends compute only where the mass is worth it.
