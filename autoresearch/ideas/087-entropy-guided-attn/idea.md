---
id: 087-entropy-guided-attn
status: needs-plan
round: 1
updated: 2026-06-11T01:17:26Z
transfer-risk: med
---

# 087 — Entropy-Guided Attention

## Source
Entropy-Guided Attention for Private LLMs (arXiv:2501.03489). 2025.

## Mechanism
Add an entropy-guided attention regularizer and a small attention-side entropy control rule so heads keep diversity when nonlinearities are reduced. This is a lightweight attention-entropy control mechanism rather than a new backbone.

## Scale evidence
The paper trains GPT-2-style models from scratch on CodeParrot and Languini at 12L/18L with context sizes 128/256/512 and 1.2B to 4.8B training tokens. `transfer-risk: med` because the experiments are still modest-scale, but they are clearly beyond toy and the mechanism is directly about head diversity.

## Why it's worth a slot
It could be a cheap way to keep attention diverse when other normalization or gating changes compress the dynamics too hard.
