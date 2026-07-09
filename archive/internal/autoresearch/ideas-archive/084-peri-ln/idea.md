---
id: 084-peri-ln
status: needs-plan
round: 1
updated: 2026-06-11T01:17:26Z
transfer-risk: low
---

# 084 — Peri-LN

## Source
Peri-LN: Revisiting Layer Normalization in the Transformer Architecture (arXiv:2502.02732). 2025.

## Mechanism
Move normalization to the periphery of each sublayer so the block keeps a stronger identity path while still controlling activation growth. This is a placement change, not a new norm formula, so it can be wired with a small config flag and the existing LN modules.

## Scale evidence
The paper reports large-scale experiments on Transformers up to 3.2B parameters and says Peri-LN gives steadier gradient flow, balanced variance growth, and convergence stability. `transfer-risk: low` because the effect is already demonstrated at billion scale and the mechanism is mostly about where normalization sits.

## Why it's worth a slot
If Peri-LN survives the tiny screen, it is a clean structural candidate for the 135M recipe because it improves trainability without changing the model family.
