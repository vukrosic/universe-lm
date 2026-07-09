---
id: 081-scale-invariant-attn
status: needs-plan
round: 1
updated: 2026-06-11T01:17:26Z
transfer-risk: med
---

# 081 — Scale-Invariant Attention

## Source
Scale-invariant attention (arXiv:2505.17083). 2025.

## Mechanism
Apply a small position-dependent transform to attention logits so total attention mass and sparsity stay roughly invariant as context grows. The implementation is a lightweight logit rescaling / shift keyed by token distance, paired with the existing attention path rather than a new backbone.

## Scale evidence
Shown in 162M and 304M GPT-2-style models on FineWeb, including 16x zero-shot long-context generalization and better validation loss when training/validating across longer contexts. `transfer-risk: med` because the win is strongest in long-context regimes, and tiny1m3m may not fully exercise that regime even though the mechanism itself is simple.

## Why it's worth a slot
If it works at tiny scale, it gives us a cheap long-context attention knob that is more structural than plain RoPE/ALiBi variants and could carry into the 135M recipe.
