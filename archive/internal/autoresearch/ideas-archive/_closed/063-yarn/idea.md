---
id: 063-yarn
status: rejected
round: 1
updated: 2026-06-10T16:49:46Z
transfer-risk: low
---

# 063 — YaRN

## Source
Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (arXiv:2309.00071). Aug 2023.

## Mechanism
Blend RoPE interpolation and extrapolation with frequency-aware rescaling, so low-frequency dimensions stretch to longer ranges while high-frequency dimensions preserve local detail. Implemented by modifying the RoPE frequency table plus the attention scale factor, with no architecture swap.

## Scale evidence
The paper reports efficient context extension on LLaMA models with 10x fewer tokens and 2.5x fewer steps than prior methods, including 128k-length reproduction. `transfer-risk: low` because it was validated on real LLaMA-scale models and the lever is a direct RoPE extension.

## Why it's worth a slot
YaRN tests whether a smarter RoPE stretch beats plain interpolation at tiny1m3m without needing any larger model assumptions.

