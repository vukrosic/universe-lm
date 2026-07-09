---
id: 062-pos-interp
status: rejected
round: 1
updated: 2026-06-10T16:52:41Z
transfer-risk: low
---

# 062 — Position interpolation

## Source
Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (arXiv:2306.15595). Jun 2023.

## Mechanism
Linearly down-scale position indices before RoPE, `pos' = pos * alpha`, so long sequences are mapped back into the pretraining range instead of pushing rotary phases into an unseen regime. This is a tiny RoPE helper change and leaves the rest of the model untouched.

## Scale evidence
The paper extends LLaMA 7B to 65B models up to 32k context with minimal fine-tuning and strong long-context results on retrieval, LM, and summarization. `transfer-risk: low` because the method is directly demonstrated on 7B+ LLMs and is almost purely a positional transform.

## Why it's worth a slot
PI is the cleanest "stay in the trained phase band" baseline; if it wins, we know the long-context failure is mostly phase drift rather than missing expressivity.

