---
id: 090-tapernorm
status: needs-plan
round: 1
updated: 2026-06-11T01:17:52Z
transfer-risk: med
---

# 090 — TaperNorm (gated normalization removal)

## Source
Kanavalau, Amo Alonso, and Lall, "Gated Removal of Normalization in Transformers Enables Stable Training and Efficient Inference" (arXiv:2602.10408). The paper's core lever is TaperNorm: start as a standard norm, then taper the sample-dependent branch away.

## Mechanism
Replace each pre-norm RMSNorm/LayerNorm with a TaperNorm module that begins identical to the baseline, then uses a single global gate to smoothly shift from token-statistic normalization to a learned sample-independent affine map. At the end of the taper, fold the remaining affine into adjacent linear layers.

## Scale evidence
The paper reports parity with normalized baselines across transformer setups and up to 1.22x throughput in last-token logits mode after folding. That is a real systems win, not just a math curiosity, and it gives a clean test of whether sample-dependent normalization is still needed once training has stabilized.

## Why it's worth a slot
This is not DyT 2.0. DyT replaces the norm with a pointwise squash immediately; TaperNorm keeps the baseline early, then removes the sample dependence later. That makes it a cleaner probe for "can we train with normalization and ship without it?" If it works here, it directly informs the inference-path cleanup story.
