---
id: 195-mid-attn-rmsnorm
status: needs-run
round: 2
updated: 2026-06-15T08:34:04Z
transfer-risk: low
plain: RMS-normalize the attention scores themselves (after Q·K, before softmax) along the key axis, with per-head γ_h init at 1 (the lever's value is in the fixed per-query RMS reshape, NOT the γ_h scalar which is the closed-null axis).
---

# 195 — Mid-Attention RMSNorm (RMSNorm on Pre-Softmax Scores)

## Source
- 016-qk-norm (in-repo WIN Δ=−0.0138) — symmetric RMSNorm on Q and K *pre*-QK^T product. Operates on the inputs to the attention product.
- 190-per-layer-qk-norm (in-repo reviewing) — *granularity* axis of 016 (per-block vs per-head γ), tests whether 016's WIN was block-level scale control or head-level specialization.
- 162-q-only-norm (closed null Δ=−0.0043) — Q-only RMSNorm.
- 165-k-only-norm (closed null Δ=−0.0293) — K-only RMSNorm.
- 184-logit-scale (in-repo needs-run) — global logit scale *post*-LM-head, not attention.
- Shleifer et al., "NormFormer" (arXiv:2110.09423) — extra LN at attention output; 195 is on attention scores, different placement.

## Mechanism
Per-head, per-query, RMS-normalize the scores over the key axis (`dim=-1`) BEFORE softmax:
```
scores_pre = Q · K^T / √d_k                      # [B, H, T, T]
scores_rms = (scores_pre / RMS(scores_pre, dim=-1, keepdim=True)) · γ_h
attn = softmax(scores_rms)
```
with `γ_h = 1.0` init (one per head, per block; H×L = 48 params total, +0.005% of 0.94M).

**Step-0 identity (chosen formulation, ONE):** `γ_h = 1.0` is *not* bit-identical to baseline (RMS of pre-softmax scores is generally not 1, so the normalized-and-rescaled scores differ from `scores_pre`). The A/B is therefore a *non-bit-identity* treatment, evaluated with the standard ctrl-pair protocol (gate: Δ < 0.005 = null, Δ ≤ −0.01 = WIN) — same protocol used for 160-post-AV-gain, 152-logit-bias, 155-per-head-temp. No first-batch stat collection, no inverse-scale-then-rescale, no "auto-init". The control is the same fresh-ctrl mean this queue uses elsewhere.

**Two-line math summary:**
- Operation: `scores_rms[t, :] = scores_pre[t, :] / sqrt(mean(scores_pre[t, :]²) + eps) · γ_h`
- At step 0 with γ_h=1: pre-softmax scores become unit-RMS along the key axis (still differ from baseline; not byte-identical).

## Why it's worth a slot
**Positive WIN hypothesis (magnitude-bounded):** target val loss Δ ≤ −0.01 at tiny1m3m (same bar as 016-qk-norm WIN). The mechanism is that *per-query* score normalization — not the per-head γ_h scalar — is the binding lever: forcing the pre-softmax scores to live on a fixed RMS scale (≈1 per query) prevents the dominant key from monopolizing the softmax via magnitude alone, regardless of head. Three closed per-head scalars (155 temp, 152 bias, 160 post-AV gain) suggest the per-head axis binds weakly at 0.94M; the per-query-variance reshape is a *different* axis (no learnable parameter per query) and is the only one in queue that re-scales the softmax's *input distribution shape* before the per-head γ_h ever sees it.

**Unique contribution vs 016/190 (the placement triangle):**
- 016-qk-norm WIN — *pre*-product QK normalization (normalizes Q and K inputs).
- 190-per-layer-qk-norm (reviewing) — *granularity* axis of 016 (per-block vs per-head γ).
- **195 — *post*-product QK normalization** (normalizes the QK^T output, pre-softmax).

The three together partition the QK-axis attribution: 016 = placement-pre, 190 = granularity, 195 = placement-post. A 195 WIN would suggest 016's WIN is about *score distribution* control, not QK-input control; a 195 NULL would confirm 016's WIN is uniquely a QK-input mechanism and the post-product axis is closed at 0.94M. Either result is informative AND distinct from 190's question (which is about γ granularity within the pre-product placement).

**Score-distribution bet:** the lever's value is the *fixed* per-query RMS reshape, not γ_h. γ_h init at 1.0 is the closest thing to "no parameter" we can offer while still letting the definition gate choose a learnable scale — but the strongest reading of the bet is that the Δ < 0.005 null pattern is *what we'd expect* if γ_h has no binding power, and any Δ ≤ −0.01 would be evidence the fixed per-query normalization (not the γ_h) is doing the work. If the post-RMS variance per query is too small a signal at d_k=16, lever is null; if the signal is sufficient, lever is at least a 016-magnitude match.

## Design sketch (mechanism-level only — code gate's job to implement)
- **File**: `models/layers.py` — add RMSNorm-on-scores block in the manual attention path, between QK^T and softmax.
- **Config flag**: `use_mid_attn_rmsnorm: bool = False`, `mid_attn_rmsnorm_eps: float = 1e-6` (default), `mid_attn_rmsnorm_gain_init: float = 1.0`.
- **Compute**: per (batch, head, query), `r = sqrt(mean(scores_pre[t, :]²) + eps)`; `scores_post[t, :] = scores_pre[t, :] / r * γ_h` with γ_h init 1.0.
- **Params**: H × L = 48 γ scalars (+0.005% of 0.94M); no other params.
- **A/B protocol**: standard ctrl-pair (gate Δ ≤ −0.01 for WIN, |Δ| < 0.005 for null), 92-step horizon, seed 42, 1M3M tier.

## Scale evidence
016 WIN at tiny1m3m (Δ=−0.0138); QK-norm literature validated at LLaMA / Gemma-2 / Qwen-2.5 (≥7B) for the *pre*-product placement. No direct "RMSNorm on attention scores" paper at any scale. Transfer-risk: low (placement is mathematically the same family, but the post-product variant is novel at any scale). The T×T per-query RMS compute is a real cost (~2048 elements reduced per head per query) — tier-mismatch hedge: at 135M, the per-head γ_h axis is also weak per the closed pattern, so a 195 NULL at tiny1m3m is *not* a 135M verdict; the lever should re-enter the queue at 135M Phase-2 where H=12+ and per-query variance carries more head-specific signal.
