---
id: 076-synthesizer
status: rejected
round: 1
updated: 2026-06-10T16:50:41Z
transfer-risk: med
---

# 076 — Synthesizer Token Mixing

## Source
Synthesizer: Rethinking Self-Attention in Transformer Models (arXiv:2005.00743). 2020.

## Mechanism
Replace the QK dot-product path with synthetic attention weights produced by learned or factorized token-mixing matrices, then blend that synthetic branch with the normal attention branch through a zero-gated residual. Step 0 stays baseline when the gate is zero.

## Scale evidence
The paper reports competitive language modeling, translation, and GLUE/SuperGLUE results, including random and factorized synthesizers that beat some efficient-attention baselines. transfer-risk: med - the idea is real and broad, but the causal-LM transfer at tiny1m3m is still speculative.

## Why it's worth a slot
This directly asks whether token-token interaction is load-bearing, or whether a learned position-only mixing prior can stand in for dot-product attention.
