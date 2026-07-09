---
id: 032-adams
status: reviewing
round: 1
updated: 2026-06-10T23:48:43Z
transfer-risk: low
---

# 032 - AdamS

## Source
AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training (arXiv:2505.16363, 2025).

## Mechanism
Replace AdamW's per-tensor second-moment buffer `v_t` with `sqrt(α·m_t^2 + β·g_t^2)` (momentum-magnitude + raw gradient, weighted), so the optimizer normalizes the update by a function of the current momentum and gradient instead of a running `v` estimate. In repo terms: a new `AdamS` class in `tools/optimizers.py` (or wherever the matrix-weight optimizers live), drop-in for the matrix-weight path, no architecture changes; scalar/norm params stay on the existing path.

## Differentiation from the in-queue optimizer wave
- vs **031 Adam-mini** (blockwise shared LR): Adam-mini keeps a per-block `v` and only changes its *granularity*; AdamS removes `v` and substitutes a momentum-based denominator. Different lever axis.
- vs **033 Sophia** (Hessian diagonal denominator): Sophia's denominator is a *curvature* estimate refreshed every k steps; AdamS's denominator is a *per-step* function of `m_t, g_t` with no curvature signal. Different denominator source.
- vs **034 Adan** (Nesterov gradient-difference m): Adan modifies *how momentum is computed*; the denominator is still a `v`-like EMA. AdamS leaves the momentum rule at standard AdamW and modifies the *normalization*. Orthogonal axes.
- vs **040 Adafactor** (closed — sublinear `v` factorization): Adafactor is a memory-saving *approximation* to `v`; AdamS replaces `v` *exactly* with a closed-form `m, g` expression. Same memory target, different math, no factorization noise.
- vs **002 Cautious-AdamW** (closed — null at tiny1m3m): cautious-AdamW masks the *update direction* with `sign(update)·1[sign(update)==sign(g)]`; `m, v` dynamics unchanged. AdamS changes the *denominator*, not the direction mask. Different mechanism layer.

## Scale evidence
Paper reports GPT-2 and Llama2 experiments up to 13B parameters in pretraining and post-training. transfer-risk: low — the method is built for transformer-scale training. **Caveat (per r1 taste note): the paper's headline is *parity* + memory ÷ 2; at 0.94M params full `v` state fits trivially, so the slot can only be earned on val loss, not memory.**

## Why it's worth a slot
**Bet (one sentence):** we expect Δval ≈ -0.005 to -0.012 at tiny1m3m (passing the recent ctrl-pair gap of 0.0047 from the 015/016/017 cluster) because the momentum-based denominator is *non-degenerate from step 1* (no cold `v_t` warmup), so it should reduce the early-step update noise that wastes the 92-step / 3M-token budget; a null (Δval inside ±0.005) closes the lever because parity with AdamW at 0.94M means the mechanism only fires at scales where `v` is the bottleneck, not at ours. A win ports to 135M because the same "no-cold-start normalization" argument is scale-free, and a control without `v` state is a cleaner recipe even if the memory win is moot at 0.94M.

## What a null still teaches
A null result at tiny1m3m still discriminates between two stories the paper doesn't separate: (a) "momentum normalization is genuinely better as a denominator" vs (b) "any well-scaled denominator matches AdamW." If (a) is the mechanism, AdamS should show a small but real win; if (b), AdamS is parity and the lever belongs only on the memory-constrained scale-up ladder (where we won't mine it).
