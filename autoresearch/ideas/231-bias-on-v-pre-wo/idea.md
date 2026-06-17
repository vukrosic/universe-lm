---
id: 231-bias-on-v-pre-wo
status: needs-taste
round: 1
updated: 2026-06-16T01:05:00Z
transfer-risk: low
plain: Add a learnable per-head additive bias to the value vector V, before the W_O projection. Different from the closed V-pre-AV norm (176, which was on V before softmax) — this is an additive bias on V before W_O, not a normalization.
---

# 231 — Additive Per-Head Bias on V (pre-W_O)

## Source
Closed 176-v-pre-av-norm null at 0.94M applied RMSNorm to V before the softmax (different operation: normalization not additive bias). 187-lm-head-bias filed but never tested an additive bias on the LM head's output, not the value path.

**231 adds a learnable per-head additive bias `b_v ∈ R^{d_k}` to V, before the softmax and AV concat**. Mathematically:
```
V_proj = X @ W_V                                    # [B, T, H*d_k]
V_proj = V_proj + self.v_head_bias.view(1, 1, H, d_k)   # [B, T, H, d_k]
V_proj = V_proj.transpose(1, 2)                     # [B, H, T, d_k]
attn   = softmax(QK^T / sqrt(d_k)) @ V_proj
```

Init: `v_head_bias = 0`. Step-0 bit-identical to baseline (additive zero).

## Mechanism & Design sketch
- **Files**: `models/layers.py` — locate the value projection. Add `nn.Parameter(H, d_k)` initialized to 0. Add it to V after the V projection and reshape.
- **Config flag**: `use_v_head_bias: bool = False`, `v_head_bias_init: float = 0.0`.
- **Cost**: 4 heads × 16 d_k × 12 blocks = +768 params, +0.082% of 0.94M. Cheap.
- **Why it should help at tiny1m3m**: per closed 152-attn-logit-bias null and 155-per-head-temp null, per-head attention-shape axes don't bind at 0.94M. But those were on QK logits (pre-softmax). 231 is on V (pre-AV), which feeds into the value path *after* softmax. The motivation: at H=4 the V outputs are aggregated through softmax weights, so the *post-softmax magnitude* matters more than the pre-softmax QK. Adding a learnable per-head V bias lets the model re-balance the per-head contribution to the residual stream. Different axis from the closed per-head logit-bias (152) and per-head temperature (155) levers.
- **Why it might be null**: at H=4 with 92 update steps, the per-head V bias may not have enough signal to learn a useful pattern. The closed 176-v-pre-av-norm null (V pre-AV *normalization*) lost, suggesting the V path's distribution is already well-conditioned at 0.94M.

## Scale evidence
Additive V bias is novel at the per-head granularity; the closest analog is per-head gain (160 closed null, post-AV), which was null at 0.94M. Transfer-risk **low** (architecturally simple, scale-agnostic) but the closed-null pattern of "per-head V/AV shape axes" suggests low prior probability of win.

## Why it's worth a slot
A win would say the per-head V bias is the missing axis the model needs — *additive* on V (different from 176's *normalization* on V). A null would close the per-head V-shape axis at 0.94M alongside 176 (norm) and 160 (gain). The lever is cheap (+768 params, ~10 LoC), bit-identical at init=0, and structurally different from the closed per-head V/AV levers.
