---
id: 287-deepnet-poly-alibi-pre-lm-head-rmsnorm
status: done
round: 1
updated: 2026-06-16T11:00:05Z
transfer-risk: low
plain: Keep the champion (alibi + DeepNet-α + poly-alibi) and add a gated RMSNorm right before the (tied) LM head — the Gemma 2 / LLaMA 3 / Qwen 2.5 / OLMo 2 final-norm. The gate starts at 0 so step-0 is byte-identical; the optimizer grows it to normalize the residual just before the logits.
---

# 287 — Pre-LM-head RMSNorm on the deepnet+poly-alibi champion

## Why this, why now
Only **step-0 conditioning** levers bind at 0.94M/92 steps (see 285/286 and
closed.md). This conditions the **output side** of the residual stream — the point
where the deep, DeepNet-α-scaled signal lands before the tied LM head — an axis the
champion stack never touches.

## Source
Final pre-head RMSNorm is standard in Gemma 2 (§2), LLaMA 3 (§3.1), Qwen 2.5 (§2.3),
OLMo 2 (§2.2). In-repo: idea 183-pre-lm-head-rmsnorm.

## Mechanism
`use_pre_lm_head_rmsnorm` builds `nn.RMSNorm(d_model)` + a scalar gate
`pre_head_scale` (init 0) and applies `x = (1−s)·x + s·RMSNorm(x)` between
output_dropout and lm_head. s=0 ⇒ exact `x` at step 0 (byte-identical, dropout
interaction preserved). Cost: 1 scalar + d_model gain weights (~65 params, +0.007%).

## Distinct from the NULL internal-norm levers
265-deepnet-layernorm, 272-qk-norm-depth, 273-qk-layernorm were all NULL — they
normalize INSIDE the block (attention/sublayer). This normalizes the **final
pre-head residual**, after all 12 blocks have written into it: a different site
with a different job (calibrating the logit-producing vector, not the attention
inputs).

## Hypothesis
Right-sign Δ if the deep residual reaching the head is mis-scaled for the tied
embedding's logit geometry; NULL if the head is already well-conditioned (the gate
stays near 0 over 92 steps).

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then a
paired 3-seed confirm before any promotion). Single seed (42).
