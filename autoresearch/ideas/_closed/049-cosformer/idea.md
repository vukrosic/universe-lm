---
id: 049-cosformer
status: rejected
round: 1
updated: 2026-06-10T16:50:55Z
transfer-risk: med
---

# 049 — cosFormer

## Source
Qin et al., "cosFormer: Rethinking Softmax in Attention" (arXiv:2202.08791), 2022.

## Mechanism
Replace the softmax kernel with cosine-based reweighting so the attention map stays concentrated while remaining linear-time. A repo-friendly version is a score-reweighting branch with unit scale at init so the baseline remains the starting point.

## Scale evidence
The paper reports results on language modeling, text understanding, and Long Range Arena, showing the kernel change is not just a toy trick. Transfer-risk is `med` because the mechanism is broad and implementable, but the headline evidence is not a single giant pretrain run.

## Why it's worth a slot
This asks whether changing the score geometry beats just tuning temperatures, and a null would say cosine reweighting does not help beyond the existing attention stack.
