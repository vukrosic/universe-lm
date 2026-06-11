---
id: 079-cosformer
status: rejected
round: 1
updated: 2026-06-10T16:49:41Z
transfer-risk: med-high
---

# 079 — CosFormer Angular Attention

## Source
cosFormer: Rethinking Softmax in Attention (arXiv:2202.08791). 2022.

## Mechanism
Replace the exponential softmax kernel with cosine-based reweighting / angular similarity, then blend the resulting attention branch in with a zero gate so the model starts at baseline.

## Scale evidence
The paper claims comparable or better accuracy than vanilla Transformer on both causal and cross-attention tasks. transfer-risk: med-high - the evidence is real, but it is mostly from efficient-attention benchmarks rather than this exact causal LM setup.

## Why it's worth a slot
This asks whether angular similarity beats dot-product similarity once the model is stripped down to 0.94M parameters.
