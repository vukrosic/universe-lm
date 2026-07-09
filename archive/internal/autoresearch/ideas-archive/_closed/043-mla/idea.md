---
id: 043-mla
status: rejected
round: 1
updated: 2026-06-10T16:45:40Z
transfer-risk: low
---

# 043 — Multi-Head Latent Attention (MLA)

## Source
DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (arXiv:2405.04434), 2024.

## Mechanism
Compress K/V into a low-rank latent state, then reconstruct the attention memory from that latent channel instead of storing full per-head KV tensors. In this repo the practical version is a latent KV branch with a zero gate at init so the baseline path is preserved before the compressed branch learns.

## Scale evidence
DeepSeek-V2 is a 236B-total / 21B-active model with 128K context, and MLA is one of the core architecture changes behind that result. Transfer-risk is `low` because the mechanism already paid off at very large scale.

## Why it's worth a slot
This asks whether a compressed memory bottleneck helps a tiny model even when KV cache pressure is not the main constraint, and a null would say the latent bottleneck is not buying anything at this scale.
