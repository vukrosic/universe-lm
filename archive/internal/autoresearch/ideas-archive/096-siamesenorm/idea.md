---
id: 096-siamesenorm
status: needs-plan
round: 1
updated: 2026-06-11T01:18:16Z
transfer-risk: med
---

# 096 — SiameseNorm

## Source
Li et al., "SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm" (arXiv:2602.08064).

## Mechanism
Use two coupled streams with shared parameters: one stream keeps the stable pre-norm identity path, the other keeps the post-norm expressive path. The block receives combined gradients from both views instead of choosing one regime.

## Scale evidence
The paper says the design is robust in 1.3B pretraining and outperforms strong baselines. That is a real signal that the pre/post-norm trade-off can be softened rather than accepted as a binary choice.

## Why it's worth a slot
This is the cleanest "have both, not either/or" architecture in the norm family. It is more invasive than a single-layer tweak, but it tests a very different hypothesis than the rest of the queue.
