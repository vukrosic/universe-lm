---
id: 088-rodimus
status: needs-plan
round: 1
updated: 2026-06-11T01:17:52Z
transfer-risk: low
---

# 088 — Rodimus

## Source
Rodimus*: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions (arXiv:2410.06577). 2024.

## Mechanism
Use data-dependent tempered selection in a linear-attention, purely recurrent block so the model can compress semantic content into a fixed hidden state while filtering irrelevant tokens. It is an attention-family rewrite, not a schedule or optimizer tweak.

## Scale evidence
The paper reports downstream evaluations from 130M to 1.3B parameters, and Rodimus+-1.6B trained on 1T tokens beats Qwen2-1.5B and RWKV6-1.6B. `transfer-risk: low` because the gains are shown at and above our scale, even though the model family is more exotic than standard softmax attention.

## Why it's worth a slot
If the selective-compression idea helps at tiny scale, it is a strong candidate for a later long-context recipe because it attacks both memory and relevance filtering at once.
