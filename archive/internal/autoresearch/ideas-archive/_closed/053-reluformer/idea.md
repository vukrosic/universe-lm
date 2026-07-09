---
id: 053-reluformer
status: rejected
round: 1
updated: 2026-06-10T16:51:58Z
transfer-risk: high
---

# 053 — ReLUFormer (rectified attention instead of softmax)

## Source
Shen et al., "A Study on ReLU and Softmax in Transformer" (arXiv:2302.06461). The paper analyzes ReLU vs softmax in both FFN/memory views and then proposes a full ReLU architecture named ReLUFormer.

## Mechanism
Replace the attention softmax with a rectified normalization: `A = ReLU(scores)` plus the paper's row scaling so the head can go sparse without the probability-simplex constraint. This is a direct attention-activation swap; it is not bit-identical to the baseline at step 0.

## Scale evidence
The paper says ReLU beats softmax when the number of value slots is large and reports ReLUFormer outperforming the baseline Transformer on long-sequence tasks such as document translation. transfer-risk: high — the evidence is sequence-length driven, not LM-pretraining driven, so the transfer signal to tiny1m3m is uncertain.

## Why it's worth a slot
This tests whether our current attention bottleneck is really about the softmax geometry itself; a null would narrow the search to the normalization stack instead of the attention activation.
