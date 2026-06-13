---
id: 107-exclusive-self-attn
status: implementing
round: 1
updated: 2026-06-13T06:28:24Z
transfer-risk: low
plain: It tries to make attention stop spending effort on the current word itself so it can use more context.
---

# 107 — Exclusive Self Attention

## Source
Exclusive Self Attention, arXiv:2603.09078 https://arxiv.org/abs/2603.09078

## Mechanism
After standard attention, subtract the component of the attention output that points along the current token's value vector. Implement it as a tiny post-attention projection with a learnable coefficient initialized to zero, so step 0 is the baseline Transformer.

## Scale evidence
The paper reports gains on language modeling runs up to 2.7B parameters trained for 100B tokens, so this is a low-risk transfer bet for a 0.94M model.

## Why it's worth a slot
If the gain comes from removing self-overlap rather than from scale, this should help tiny models too; if not, we learn that attention/FFN separation is already saturated here.
