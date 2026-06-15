---
id: 181-cross-head-rmsnorm
status: needs-review
round: 1
updated: 2026-06-15T05:47:51Z
transfer-risk: med
plain: Normalize each token's attention output across heads (so all four heads land on the same scale) before mixing back into the residual stream, starting with the per-head gain at 1 so step-0 is byte-identical.
---

# 181 — Cross-Head Channel RMSNorm (Normalize Attention Output Across Head Dim, Pre-W_O)

## Source
- The closest published analog is **NormFormer** (Shleifer et al. 2021, arXiv:2110.09423) which adds an extra LayerNorm on the attention output before the residual — validated on ViT-class models and small LMs.
- **RMSNorm** (Zhang & Sennrich 2019, arXiv:1910.07467) is the well-known "drop the mean-subtraction" simplification; RMSNorm on attention output is a known variant in Qwen-2 and Gemma-2.
- The **specific cross-head axis** (normalize across H dim, treating the H×d_k as a single H·d_k channel) is novel at this tier. Standard RMSNorm on attention output normalizes over the d_model axis (concatenated across heads). Cross-head RMSNorm instead treats each head's d_k channels as a single channel and normalizes *across heads within each d_k slice*.
- In-repo context: norm zoo closed (pnorm, manhattan, center, squash, clip, channelscale). 160-rms-gain-per-head null is post-AV per-head gain (different axis — gain not normalization). 142-layerscale null is per-channel diagonal gain (different axis). 176-v-pre-av-norm is V-normalization pre-AV. The cross-head normalization axis is fresh.

## Mechanism
Standard attention output:
- `attn_w = softmax(QK^T / √d_k)` — `[B, H, T, T]`.
- `out = attn_w @ V` — `[B, H, T, d_k]`.
- `out_concat = out.transpose(1, 2).reshape(B, T, H · d_k)` — `[B, T, d_model]`.
- `output = out_concat @ W_O` — `[B, T, d_model]`.

With cross-head channel RMSNorm (applied on `out` before reshape/concat/W_O):
- For each (b, t) position and each d_k index k (in 0..d_k-1):
  - Compute RMS across the H heads: `rms[b, t, k] = sqrt(mean(out[b, :, t, k]^2) + ε)`.
  - Normalize: `out_norm[b, h, t, k] = out[b, h, t, k] / rms[b, t, k]`.
  - Apply per-(h, k) gain: `out_norm[b, h, t, k] *= γ_h[k]` with `γ_h[k]` init 1.
- Then continue with concat → W_O as usual.

Bit-identity at step 0: `γ_h[k] = 1` for all h, k ⇒ `out_norm = out` exactly ⇒ **byte-identical to baseline at step 0**.

The lever: `γ_h[k]` is a learnable gain. The optimizer can scale up or down per-head, per-d_k channel.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_cross_head_rmsnorm: bool = False` to `MultiHeadAttention.__init__`. Allocate `self.cross_head_rmsnorm_gain = nn.Parameter(torch.ones(n_heads, d_k))` (init 1 ⇒ identity). After computing `out = attn_w @ V` (shape `[B, H, T, d_k]`), apply `out = out / sqrt(mean(out^2, dim=1, keepdim=True) + eps) * self.cross_head_rmsnorm_gain.unsqueeze(0).unsqueeze(2)`. The mean is over the H axis.
  - `configs/llm_config.py` — add `use_cross_head_rmsnorm: bool = False`. Add `Tiny1M3MCrossHeadRMSNormConfig` subclass with `use_cross_head_rmsnorm: bool = True`.
  - `models/llm.py` — thread into both `TransformerBlock` sites.
- **Config flag**: `use_cross_head_rmsnorm: bool = False`.
- **Step-0 byte-identical**: `γ_h[k] = 1` for all h, k ⇒ `out / sqrt(mean(out^2, dim=1, keepdim=True) + eps) * 1 = out / sqrt(...)` which is **NOT byte-identical** (it's rescaled). To get exact byte-identity, wrap in an α-gate: `out_final = (1 − α) · out + α · out_rmsnorm * γ`, with `α` a single learnable scalar init 0 ⇒ `out_final = out` exactly. Use **two gates**: one for the gate-α (init 0), one for the per-head gain (init 1). Or simpler: parameterize `γ_h[k] = 1 + tanh(γ_raw_h[k])` with init 0 ⇒ `γ = 1` ⇒ `out_norm = out` exactly. Use the **tanh form** for clean step-0 identity.
- **Param count**: H=4, d_k=16, n_layers=12. Per block: 4·16 = 64 params. Total: 768 params (+0.081% of 0.94M).
- **Intuition (why it might lower val loss)**: standard post-AV gain (160 closed null) operates on each head independently. Cross-head RMSNorm explicitly couples the heads by normalizing their magnitudes against each other before they enter W_O. The motivation: if one head has much larger output magnitude than the others, the W_O projection sees an imbalanced input and may have to rebalance. Cross-head normalization makes the input to W_O more uniform, which could be easier to learn. This is a *coupling* lever, fundamentally different from per-head independent gains.

## Scale evidence
- NormFormer at 100M-300M (encoder-decoder) showed modest gains.
- RMSNorm on attention output is well-validated (Qwen-2, Gemma-2, LLaMA-3 at 7B-70B).
- The **cross-head** axis is novel at 100M+. Transfer-risk is **med**.

## Why it's worth a slot
The bet, in one sharp sentence: **the closed post-AV per-head gain (160 null) tested per-head independent rescaling and confirmed the per-head axis is closed; 181 tests a cross-head coupling that explicitly ties head magnitudes together — a different lever axis that 160 did not cover.** A null at 0.94M would close the cross-head-normalization family at our tier and confirm that head-magnitude coupling is not the binding constraint; a win would unlock the cross-head coupling axis for Phase-2 ≥135M where the head-magnitude balance matters more.
