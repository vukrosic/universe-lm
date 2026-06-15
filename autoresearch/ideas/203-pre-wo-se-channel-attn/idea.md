---
id: 203-pre-wo-se-channel-attn
status: revising
round: 2
updated: 2026-06-15T16:46:44Z
transfer-risk: med
plain: Insert a tiny Squeeze-Excitation channel-attention block right before the W_O projection (init so it returns all-ones, byte-identical at step 0), letting each token softly up- or down-weight its own channels.
---

# 203 — Pre-W_O Squeeze-Excitation Channel Attention (Per-Token Channel Reweighting)

## Source
- Hu et al., "Squeeze-and-Excitation Networks" (SE, TPAMI 2019, arXiv:1709.01507) — channel attention in CNNs; validated at ImageNet across all scales (ResNet, EfficientNet).
- 181-cross-head-rmsnorm (closed null Δ=+0.1722 wrong-sign above band) — cross-head RMSNorm on attention output pre-W_O. Different axis (cross-head normalization vs channel attention).
- 142-layerscale (closed null Δ=+0.0172 wrong-sign) — per-channel diagonal gain on residual stream; SE is per-token channel attention (different placement and different shape).
- 160-rms-gain-per-head (closed null Δ=−0.0023 inside band) — per-head gain on attention output. Different axis (head-wise vs channel-wise).
- 191-token-attn-gain (in-repo idea, status needs-taste) — per-token *scalar* on attention output. 203 is per-token *channel vector* on attention output (broader).
- Woo et al., "CBAM" (ECCV 2018, arXiv:1807.06521) — channel + spatial attention; SE is the channel-only branch.

## Mechanism
Standard pre-W_O: `attn_out_post = attn_out` (shape `[B, T, d_model]`), then `W_O(attn_out_post)`.

SE channel attention: insert a per-token channel attention block that reweights each token's channels:
```
attn_out_se = attn_out · se_weight(attn_out)        # se_weight: [B, T, d_model] → [B, T, d_model]
attn_out_post = (1 − γ) · attn_out + γ · attn_out_se # γ learnable, init 0
out = W_O(attn_out_post)
```
Where `se_weight(x) = sigmoid(W_2 · gelu(W_1 · x_pooled))` with x_pooled = mean over T axis, W_1, W_2 ∈ R^{d_model × d_model/r} with reduction ratio r=4 (so W_1 is `d_model × 16`, W_2 is `16 × d_model`).

At init γ=0, `attn_out_post = attn_out` exactly (bit-identical baseline). The SE branch is silent. As γ grows, the SE branch is *added* to the residual via γ-weighted blend.

## Design sketch
- **File**: `models/layers.py` — add an `se_channel_attn` module to attention block.
- **Config flag**: `use_se_pre_wo: bool = False`, `se_reduction_ratio: int = 4`, `se_alpha_init: float = -10.0` (sigmoid ≈ 0).
- **Compute**: SE block computes per-token channel weights. `attn_out_post = (1 − sigmoid(γ_raw)) · attn_out + sigmoid(γ_raw) · (attn_out · se_weight)`.
- **Bit-identical at step 0**: γ_raw = -10 ⇒ sigmoid ≈ 0 ⇒ `attn_out_post = attn_out` exactly.
- **Params**: SE block: 2 × (d_model × d_model/r) = 2 × (64 × 16) = 2048 params per block × 12 blocks = 24,576 params (+2.6% of 0.94M); plus 12 γ scalars.
- **Intuition**: SE channel attention reweights each token's channels based on the *content* of that token. The bet: at 0.94M, attention already mixes cross-token information, but within-token channel mixing is implicit through W_O. An explicit per-token channel attention gives the optimizer a clean axis to up-weight informative channels and down-weight noise channels — distinct from LayerScale (per-channel diagonal gain) and cross-head RMSNorm (cross-head normalization).

## Scale evidence
SE-Net validated at ImageNet (all scales); CBAM validated at ImageNet. No published "pre-W_O SE channel attention for LMs" paper I'm aware of. Transfer-risk: med (lever is a well-known CNN primitive applied to a fresh placement in attention output).

## Why it's worth a slot
**Pattern**: per-channel gain (142 LayerScale) and cross-head RMSNorm (181) both closed null at 0.94M. 203 is *per-token channel attention* — a learned *content-dependent* channel reweighting, distinct from per-channel static gain (LayerScale) and cross-head normalization. The bet: at 0.94M, the *content* of the token matters for which channels are informative, and a per-token SE block exploits this. A 203 WIN would mean content-dependent channel reweighting binds; a 203 NULL would mean channel reweighting is generally redundant with W_O at 0.94M.
