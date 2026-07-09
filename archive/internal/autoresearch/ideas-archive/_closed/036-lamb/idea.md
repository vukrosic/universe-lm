---
id: 036-lamb
status: rejected
round: 1
updated: 2026-06-10T16:44:57Z
transfer-risk: low
---

# 036 - LAMB

## Source
Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes (arXiv:1904.00962, 2019).

## Mechanism
Scale each layer's Adam-style update by a trust ratio `||w|| / ||u||`, which balances update magnitude against parameter magnitude on a per-layer basis. In this repo the obvious test is to apply the trust-ratio scaling to the matrix-weight optimizer path and keep the scalar path unchanged.

## Scale evidence
The paper reports BERT training with batch size 32,868 without accuracy loss and the headline 76-minute BERT result. transfer-risk: low - this is a direct transformer-scale optimizer result, not a toy example.

## Why it's worth a slot
LAMB is the canonical "layerwise adaptive LR" optimizer for transformer pretraining, so it is a good baseline for whether our tiny model wants trust ratios more than AdamW's raw adaptive moments.
