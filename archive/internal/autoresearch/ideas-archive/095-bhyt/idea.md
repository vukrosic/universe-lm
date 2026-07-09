---
id: 095-bhyt
status: needs-plan
round: 1
updated: 2026-06-11T01:18:16Z
transfer-risk: med
---

# 095 — BHyT (bounded hyperbolic tanh)

## Source
Byun et al., "Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models" (arXiv:2601.09719).

## Mechanism
Replace Pre-LN with a tanh-based bounded activation that keeps inputs inside a safe range, while computing exact statistics once per block and using a lightweight approximation for the second normalization site.

## Scale evidence
The paper reports 374M and 1B Llama pretraining runs, plus faster training and higher token throughput versus RMSNorm. It also claims a theoretical stability guarantee, which is stronger than the usual "it worked on one model" story.

## Why it's worth a slot
This is a practical norm-free-ish lever with a clear depth story. If tiny models like it, we learn that explicit input bounding can beat repeated statistics even before we get to massive scale.
