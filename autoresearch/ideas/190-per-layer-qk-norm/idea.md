---
id: 190-per-layer-qk-norm
status: reviewing
round: 1
updated: 2026-06-15T08:18:25Z
transfer-risk: low
plain: Apply RMSNorm to Q and K inside attention, but with a single shared gain per layer (instead of per-head), starting at 1 so step-0 matches the baseline exactly — a coarser-grained version of the QK-norm win.
---

# 190 — Per-Layer QK Norm (Single Shared RMS Gain Per Layer, Not Per Head)

## Source
- 016-qk-norm (in-repo WIN Δ=−0.0138 at tiny1m3m) — symmetric RMSNorm on Q and K pre-softmax, with **per-head** RMS gain (one gain per head per block = 12 blocks × 4 heads = 48 gains).
- 162-q-only-norm (closed null Δ=−0.0043 inside band) — Q-only RMSNorm with per-head gain.
- 165-k-only-norm (closed null Δ=−0.0293 inside band) — K-only RMSNorm with per-head gain.
- 169-qk-norm-depth (closed null Δ=−0.020 inside band) — per-depth variation of QK norm; closed by 016's WIN.
- Zhang & Sennrich, "RMSNorm" (arXiv:1910.07467, 2019) — base RMSNorm formulation.

## Mechanism
016's QK norm is `Q_norm = RMSNorm(Q) · γ_h`, `K_norm = RMSNorm(K) · γ_h` where γ_h is per-head. 190 keeps the symmetric QK-norm structure but **shares the gain across heads within a block**: one γ_b per block (12 params total instead of 48).

The bet: at d_k=16 and 4 heads, the per-head gain axis is over-parameterized for the binding signal. The QK-norm benefit at 0.94M might be driven by *block-level* scale normalization, not *head-level* specialization. If true, sharing γ across heads at a given block gives 4× fewer params for the same block-level conditioning benefit, with the *side effect* that all heads see the same RMS scale (forcing them to compete on a normalized playing field).

## Design sketch
- **File**: `models/layers.py` — modify the QK norm block to optionally use per-block shared gain instead of per-head gain.
- **Config flag**: `qk_norm_share_across_heads: bool = False` (default), `qk_norm_blockwise_init: float = 1.0` (init at 1 for byte-identity).
- **Bit-identical at step 0**: γ_b = 1.0 ⇒ `Q_norm = RMSNorm(Q) · 1 = RMSNorm(Q)` exactly (matches 016's path with γ=1 per head, which is the 016 init).
- **Params**: 12 γ scalars (one per block) instead of 48 (one per head per block). 4× fewer params. Negligible vs 0.94M.
- **Intuition**: 016's WIN was likely driven by QK-norm's block-level scale control, not head-level specialization. Sharing γ across heads within a block should match 016's gain. If it doesn't (190 null while 016 wins), the WIN was head-specific and the per-head axis is the binding signal.

## Scale evidence
016-qk-norm WIN at tiny1m3m (Δ=−0.0138); QK-norm literature validated at LLaMA / Gemma-2 / Qwen-2.5 (≥7B). Transfer-risk: low (the lever is a strict subset of 016's mechanism).

## Why it's worth a slot
**Attribution insight**: 016 WIN at per-head γ, 162/165 NULL at per-head γ on Q-only or K-only. The bet is that 016's WIN was *block-level* (the symmetric QK RMS scale) and not *head-level* (per-head γ specialization). A 190 WIN (matching 016's WIN magnitude with shared γ) confirms the block-level axis is binding; a 190 NULL (γ=1 per head already at the optimal scale) suggests 016's WIN was indeed head-specific. Either result is informative for the QK-norm attribution puzzle.
