---
id: 195-mid-attn-rmsnorm
status: needs-taste
round: 1
updated: 2026-06-15T09:00:00Z
transfer-risk: low
plain: Normalize the attention scores themselves (after Q·K, before softmax) with a single RMS gain per head, starting at 1 so step-0 matches the baseline — a way to re-scale the raw attention logits without changing what they point at.
---

# 195 — Mid-Attention RMSNorm (RMSNorm on Pre-Softmax Scores)

## Source
- 016-qk-norm (in-repo WIN Δ=−0.0138) — symmetric RMSNorm on Q and K *pre*-QK^T product. Operates on the inputs to the attention product.
- 162-q-only-norm (closed null Δ=−0.0043) — Q-only RMSNorm.
- 165-k-only-norm (closed null Δ=−0.0293) — K-only RMSNorm.
- 169-qk-norm-depth (closed null) — depth variation of 016.
- 184-logit-scale (in-repo needs-run) — global logit scale *post*-LM-head, not attention.
- 152-attn-logit-bias (closed null) — per-head additive bias on attention logits. 195 is multiplicative normalization, not additive bias.
- Shleifer et al., "NormFormer" (arXiv:2110.09423) — extra LN at attention output; 195 is on attention scores, different placement.

## Mechanism
Standard attention scores: `scores = Q · K^T / √d_k`, shape `[B, H, T, T]`.

Mid-attention RMSNorm: per-head, per-query, normalize the scores over the key axis (`dim=-1`) BEFORE softmax:
```
scores_pre = Q · K^T / √d_k                      # [B, H, T, T]
scores_rms = RMSNorm(scores_pre, dim=-1) · γ_h    # γ_h per-head scalar, init 1
attn = softmax(scores_rms)
```
At init `γ_h = 1.0`, the post-RMSNorm scores have unit RMS along the key axis (not unit std; RMSNorm doesn't subtract the mean). The softmax is invariant to *additive* shifts but not to *multiplicative* shifts along the key axis, so `γ_h = 1.0` is not byte-identical to baseline (the RMS of baseline scores is not 1).

To preserve step-0 byte-identity: `γ_h = sqrt(var(scores_pre) + eps)` per-query (i.e., a *learnable* multiplier that starts matching the post-RMSNorm RMS to baseline). Alternatively, set `γ_h` such that the post-RMSNorm RMS matches the baseline: `γ_h = 1.0` AND apply the inverse-scale after RMSNorm: `scores_rms = (RMSNorm(scores_pre) - 1.0) · γ_h + baseline_rms` — too complex.

**Simpler formulation**: `scores_post = RMSNorm(scores_pre) · (baseline_rms_estimate)` where `baseline_rms_estimate = sqrt(E[(scores_pre)²])` is computed once on the first batch and frozen. At step 0, this matches baseline. Then `γ_h = 1.0` is a per-head learnable scale around this estimate. Still complex.

**Cleanest formulation (proposed for 195)**: use a per-head learnable scalar `γ_h` init such that `γ_h · E[RMS(scores_pre)] = √d_k` (matches the standard scale). At init, `γ_h_init = √d_k / E[RMS(scores_pre)]`. The lever is then per-head deviation from this baseline scale — bit-identical at step 0 if `γ_h_init` is computed correctly from the first batch's statistics.

## Design sketch
- **File**: `models/layers.py` — add a `mid_attn_rmsnorm` block to the manual attention path.
- **Config flag**: `use_mid_attn_rmsnorm: bool = False`, `mid_attn_rmsnorm_gain_init: float = "auto"` (compute γ_h_init from baseline statistics on first batch).
- **Compute**: per head h, per query t, compute `RMS(scores_pre[t, :]) = sqrt(mean(scores_pre[t, :]²))`. Normalize: `scores_norm[t, :] = scores_pre[t, :] / (RMS(scores_pre[t, :]) + eps)`. Multiply by `γ_h · baseline_rms_h` where `baseline_rms_h` is the expected RMS from the baseline (≈ √d_k = 4).
- **Bit-identical at step 0**: with `γ_h_init` correctly computed, the normalized-then-rescaled scores match `scores_pre` exactly.
- **Params**: H × L = 4 × 12 = 48 γ scalars (+0.005% of 0.94M).
- **Intuition**: 016's WIN was at the QK-norm input level. 195 tests a different placement: at the *output* of QK^T (pre-softmax). The bet is that RMS-normalizing the attention logits over the key axis lets the softmax operate on a more uniform scale, preventing any single key from dominating due to magnitude alone.

## Scale evidence
016 WIN at tiny1m3m; NormFormer extra LN at attention output at BERT-base / ViT-base. No direct "RMSNorm on attention scores" paper I can cite — this is a placement variant of 016. Transfer-risk: low (lever is a strict variant of QK-norm with the same mechanism family).

## Why it's worth a slot
**Attribution insight**: 016-QK-norm WIN was attributed to *block-level* scale control (190). 195 tests a different placement: post-QK^T, pre-softmax. The bet: normalizing the scores themselves (not the QK inputs) lets the optimizer control the *relative* magnitude of attention scores per query, distinct from 016's *absolute* scale control. A 195 WIN would suggest the binding axis is on the attention-output side; a 195 NULL would confirm 016's WIN is uniquely a QK-input mechanism.
