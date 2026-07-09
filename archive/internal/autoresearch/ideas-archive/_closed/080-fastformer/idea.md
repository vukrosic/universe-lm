---
id: 080-fastformer
status: rejected
round: 1
updated: 2026-06-10T16:50:35Z
transfer-risk: high
---

# 080 — Fastformer Additive Attention

## Source
Fastformer: Additive Attention Can Be All You Need (arXiv:2108.09084). 2021.

## Mechanism
Compute global query/key summaries and use additive attention instead of a full pairwise QK matrix, then add the result through a zero-initialized branch. The standard transformer path remains the baseline when the gate is zero.

## Scale evidence
The paper reports competitive NLP accuracy with lower attention cost, but the published LM-scale evidence is thinner than for the relative-position papers. transfer-risk: high - this is a bigger architectural bet with less direct causal-LM proof.

## Why it's worth a slot
This is the strongest "replace pairwise attention with a global summary" bet in the set; a null would say tiny causal LMs still want explicit token-token scoring.
