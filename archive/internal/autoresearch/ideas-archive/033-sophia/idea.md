---
id: 033-sophia
status: needs-plan
round: 1
updated: 2026-06-11T01:34:07Z
transfer-risk: low
---

# 033 - Sophia

## Source
Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training (arXiv:2305.14342, 2023).

## Mechanism
Per-coordinate update `clip(m_t / max(h_t, ε), ρ)`, where `m_t = β1·m_{t-1} + (1-β1)·g` is the gradient EMA (β1=0.965) and `h_t = β2·h_{t-1} + (1-β2)·ĥ` is a diagonal Hessian EMA (β2=0.99) refreshed by a Hutchinson estimator `ĥ = u ⊙ ∇²L·u` every k=10 steps. The hard clip ρ ≈ 0.04 caps each coordinate's step magnitude — in directions where the Hessian estimate is noisy or near-zero, the update degenerates to a sign-of-momentum step rather than blowing up.

**Routing (committed — option (a) from r1 taste):** Sophia replaces *only* the existing AdamW path. That covers the 2D non-Muon params `token_embedding.weight`, `emb_proj.weight` (when `emb_rank < vocab`), and `lm_head` (when untied). All 1D scalars and RMSNorm γ stay on plain AdamW (Hessian diagonal on 1D is just `g²`-ish noise; not worth the LoC). Muon on the 2D hidden weights (attn QKVO, FFN up/down) is **untouched** — this is a probe of AdamW-vs-Sophia on the vocab/projection slot, not a Muon ablation. (Options (b) "Sophia replaces Muon" and (c) "Sophia rescales a Muon step" are explicitly rejected: (b) would mostly measure the loss of Muon orthogonalization (load-bearing per 015 WIN), and (c) is a 3× LoC composition that isn't what the source paper validates.)

## Scale evidence
The paper reports GPT-family pretraining from 125M to 1.5B parameters and claims the same perplexity with 50% fewer steps than AdamW. transfer-risk: **low** — direct LM-pretraining evidence at and above the 135M target scale.

**EMA-firing math at tiny1m3m (lifted from r1 taste, so this doesn't get reflex-rejected like 018-ademamix).** 732 optimizer steps (batch 2 · seq 2048 · 3M tok, grad_accum=1). β1=0.965 → momentum half-life ~19 steps → ~38 half-lives across the run (fully fired). β2=0.99 → Hessian-EMA half-life ~69 steps → ~10 half-lives across the run (fully fired). Hessian refresh every k=10 → ~73 refreshes. The "EMA-doesn't-fire" failure mode that killed 018-ademamix (β3=0.9999, half-life ~7k steps vs 732-step run) does **not** apply here.

## Why it's worth a slot
**Bet (directional):** we expect a **WIN of −0.01 to −0.03 vs AdamW** on the vocab/projection slot, because the dominant gradient at tiny1m3m is `token_embedding.weight` (vocab=49152, ~91% of params), and AdamW's `1/√v` underestimates curvature on rare-token rows where `v` is sparse — Sophia's diagonal Hessian estimate is averaged over all positions in the Hutchinson probe so it produces a non-zero damping on those same rows, sharpening the per-coordinate step where AdamW under-conditions. The clip ρ keeps it from over-correcting when ĥ is noisy.

**Why this is not a re-run of the 003-SOAP NULL.** SOAP failed at tiny1m3m because "vocab params on AdamW fallback so SOAP mostly bypassed" (closed.md) — SOAP needs an eigendecomposition on a `(d_out, d_out)` basis per 2D param, which for `token_embedding` (49152 × d_model) is either intractable or falls back to AdamW, exactly killing the discriminator. **Sophia is elementwise.** The diagonal Hessian estimate has the same shape as the gradient — no eigendecomp, no fallback — so it actually runs on the vocab embedding slot where SOAP did not. A NULL here teaches "even an elementwise curvature estimator with the clip safety net can't beat AdamW at 732 steps on the vocab tail", which is genuinely new info; the SOAP NULL only tells us "the eigenbasis path was bypassed." A WIN proves curvature info on the vocab head is the missing ingredient.

**Portfolio position.** Optimizer NULLs in closed.md: 001 cautious-muon, 002 cautious-adamw, 003 SOAP, 005 decoupled-qkv-muon, 006 schedule-free-adamw. Optimizer WINs: 011 cautious-lion, 015 Moonlight-Muon-RMS. As the 8th optimizer probe, the *only* thing that justifies the slot is: elementwise curvature on the vocab head, where SOAP couldn't reach and AdamW under-conditions — and a clip ρ that makes the noisy-Hessian failure mode degrade to SignSGD-with-momentum rather than diverging.
