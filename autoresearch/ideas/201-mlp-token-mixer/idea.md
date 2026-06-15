---
id: 201-mlp-token-mixer
status: needs-repitch
round: 1
updated: 2026-06-15T08:30:04Z
transfer-risk: med
plain: Add a tiny token-mixing MLP over the attention output before the W_O projection (init at 0 contribution so step-0 is byte-identical), letting the model softly share information across nearby tokens outside of attention.
---

# 201 — MLP Token Mixer on Attention Output (Pre-W_O Cross-Token MLP)

## Source
- Tolstikhin et al., "MLP-Mixer" (NeurIPS 2021, arXiv:2105.01601) — uses MLP-only token mixing and channel mixing as an alternative to attention. Validated at ImageNet scale.
- 163-v-mix-conv (in-repo needs-run) — depthwise Conv1d on attention output pre-W_O. Conv is a *local* token mixer; 201 is an *MLP* (global) token mixer.
- 143-shortconv (closed borderline, Δ=−0.0134 not passing WIN bar) — pre-attention depthwise Conv1d on residual. Different placement (pre-attention vs pre-W_O).
- 157-conv-ffn (closed null) — depthwise conv post-FFN-activation. Different placement.
- Liu et al., "Pay Attention to MLPs" (gMLP, NeurIPS 2021, arXiv:2105.08050) — gMLP uses spatial gating unit (essentially an MLP token mixer) as alternative to attention.
- 116-hyper-connections (closed null Δ=+0.0666 wrong-sign) — residual-stream expansion via mHC. Different axis.

## Mechanism
Standard attention output: `out = softmax(QK^T/√d) @ V` shape `[B, T, d_model]`. Then `W_O(out)` projects back to the residual stream.

MLP token mixer: apply a 2-layer MLP that mixes information across the token axis (T), placed between the attention output and W_O:
```
attn_out = softmax(QK^T/√d) @ V              # [B, T, d_model]
attn_out = attn_out + α · token_mlp(attn_out) # α learnable, init 0
out = W_O(attn_out)
```
Where `token_mlp` is a 2-layer MLP: `token_mlp(x) = W_2 · gelu(W_1 · x)` with W_1, W_2 operating over the token axis (shape `[d_model × d_model]` for a per-token linear, or `[d_model × T × d_model]` for a full token-mixing MLP — too many params).

**Cleaner formulation**: use a 1D depthwise Conv1d with kernel size = T (full token mixing) or a smaller kernel size k=4 for local mixing. But this is just 163 with a different kernel.

**Different formulation**: an MLP across the **token axis only**, applied *per channel*:
```
token_mlp(x)[t, c] = sum_s W_1[c, s] · x[t, s]    # W_1: [d_model × T], W_2: [T × d_model]
                   = W_2[c, s] · gelu(...)
```
This is a per-channel linear mixing across tokens. At init α=0, no contribution (bit-identical).

## Design sketch
- **File**: `models/layers.py` — add a `token_mlp_mixer` module to attention block.
- **Config flag**: `use_mlp_token_mixer: bool = False`, `token_mixer_alpha_init: float = -10.0` (sigmoid ≈ 0).
- **Compute**: 2-layer MLP with W_1, W_2 ∈ R^{d_model × d_model}, applied per-channel across tokens. `α = sigmoid(α_raw)`. `attn_out_post = attn_out + α · token_mlp(attn_out)`.
- **Bit-identical at step 0**: α ≈ 0 ⇒ `attn_out_post = attn_out` exactly.
- **Params**: 2 × d_model × d_model = 8192 + 1 α × 12 blocks = 12 α's, total ~98k extra params (+10.4% of 0.94M).
- **Intuition**: attention is a *content-based* token mixer; an MLP token mixer is a *position-based* (or content-free) token mixer. Together they cover both axes. The bet: at 0.94M, attention alone may not be enough for cross-token information flow (e.g., for tokens that don't share content but are spatially related), and an MLP mixer adds a complementary channel. Different from 163 (depthwise conv = local MLP) and from 143-shortconv (pre-attention conv).

## Scale evidence
MLP-Mixer validated at ImageNet (up to ~432M params); gMLP validated at ImageNet. No published "MLP token mixer on attention output" win for LMs that I'm aware of. Transfer-risk: med (lever is a known architectural primitive applied to a fresh placement).

## Why it's worth a slot
**Pattern**: pre-attention conv (143-shortconv) closed borderline; post-attention conv (163) needs-run. 201 is the *MLP* analog of 163 — broader token mixing (kernel = full token axis or large window) vs conv's narrow kernel. The bet: conv mixes locally (kernel=3); MLP mixes globally. If the binding axis is *global* token mixing on attention output (not local), 201 wins where 163 doesn't. A 201 WIN would mean global token mixing complements attention; a 201 NULL would mean the cross-token information is fully handled by attention.
