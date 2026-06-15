---
id: 191-token-attn-gain
status: needs-plan
round: 1
updated: 2026-06-15T08:23:17Z
transfer-risk: low
plain: Multiply each token's attention output by a learnable per-token scalar (init 1, byte-identical at step 0), so the model can softly up- or down-weight its own attention — a token-level gate on top of the residual stream.
---

# 191 — Per-Token Attention Output Gain (Token-Wise Soft Attention Magnitude)

## Source
- 142-layerscale (closed null Δ=+0.0172, depth-conditional at 12L too shallow) — per-channel diagonal gain on residual stream (H×d_k=d_model channels). Different axis.
- 160-rms-gain-per-head (closed null Δ=−0.0023 inside band) — per-head RMS gain on attention output *post-AV*. Different axis (head-wise, not token-wise).
- 176-v-pre-av-norm (closed null Δ=+0.0303 inside band) — V-side normalization pre-attention-product. Different placement.
- Shleifer et al., "NormFormer" (arXiv:2110.09423, 2021) — extra LN at attention/FFN output, validated at ViT/BERT scale.
- Touvron et al., "ResNet-vd" / "CaiT" (2021) — class-attention with learned CLS-token gain; per-token attention output gain is the residual-stream analog.

## Mechanism
Standard attention output: `out = softmax(QK^T/√d) @ V` — shape `[B, H, T, d_k]`, then reshape to `[B, T, d_model]` and project via W_O.

Per-token attention gain: add a learnable per-token scalar `γ_t ∈ ℝ` that multiplies the attention output before W_O:
```
out = softmax(QK^T/√d) @ V                      # [B, T, d_model]
out = out * (1 + γ_t)                            # γ_t: [B, T] or [T] learnable, init 0
out = W_O(out)
```
At init γ_t = 0, the gain is `(1 + 0) = 1` exactly (bit-identical to baseline). γ_t is a scalar per token (either broadcast across the batch, or per-batch learnable). Total params: T scalars × 12 blocks = 24,576 params (+2.6% of 0.94M) if per-block-per-position; or T × 12 = 24,576 if shared across batch.

The bet: per-token scale on attention output lets the model *gate* which tokens contribute strongly to the residual stream (a soft attention sink mechanism, but at the output level rather than the input level).

## Design sketch
- **File**: `models/layers.py` — add a `token_attn_gain` parameter to `attention_block`: `nn.Parameter(torch.zeros(T))`, init 0.
- **Config flag**: `use_token_attn_gain: bool = False`, `token_attn_gain_positionwise: bool = True` (True = per-token scalar; False = single scalar per block).
- **Compute**: `out_post = out_post * (1 + self.token_attn_gain.unsqueeze(-1))`. Init γ=0 ⇒ `1+0 = 1` (bit-identity).
- **Bit-identical at step 0**: γ_t = 0 everywhere ⇒ multiplication by 1 exactly.
- **Params**: 24,576 params (+2.6%). Marginal but non-trivial at 0.94M.
- **Intuition**: per-token scale on attention output = "which tokens are informative this block?". Different from attention sink (closed) which adds learned keys to the sequence; 191 scales *existing* token attention outputs.

## Scale evidence
NormFormer at BERT-base (~110M) — extra LN at attention output. ResNet-vd depthwise-gating at ImageNet scale. Transfer-risk: low (per-token scale is a well-known conditioning primitive; the lever is well-validated at ≥100M).

## Why it's worth a slot
The attention-output axis is fresh (142 closed per-channel LayerScale; 160 closed per-head gain; 191 is per-token, distinct). The bet: at 0.94M, the network has no spare capacity to learn "which tokens are important this block" implicitly; an explicit per-token gain gives the optimizer a clean axis. A null means the axis is redundant with the W_O projection that follows; a win means per-token scale is a binding lever at tiny1m3m.
