---
id: 232-q-only-logit-bias
status: needs-taste
round: 1
updated: 2026-06-16T01:05:00Z
transfer-risk: med
plain: Add a learnable per-head additive bias to the query vector Q, *before* the QK dot product. Different from the closed QK^T logit bias (152) — 152 was on the post-dot-product scores; 232 is on Q directly, before the dot product.
---

# 232 — Per-Head Q-side Additive Bias (pre-dot-product)

## Source
Closed 152-attn-logit-bias null at 0.94M was per-head QK^T *logit bias* — additive on the post-dot-product scores. Closed 162-q-only-norm null at 0.94M was RMSNorm on Q (pre-softmax), which is a normalization not an additive bias.

**232 adds a learnable per-head additive bias `b_q ∈ R^{d_k}` to Q, before the QK dot product**. Mathematically:
```
Q_proj = X @ W_Q                                    # [B, T, H*d_k]
Q_proj = Q_proj + self.q_head_bias.view(1, 1, H, d_k)  # add per-head bias
Q_proj = Q_proj.transpose(1, 2)                     # [B, H, T, d_k]
K_proj = X @ W_K.transpose(...)                     # [B, H, T, d_k]
scores = Q_proj @ K_proj.transpose(-1, -2) / sqrt(d_k)
```

Init: `q_head_bias = 0`. Step-0 bit-identical to baseline (additive zero). The lever is asymmetric — adds bias to Q only, leaves K raw. The intuition: Q controls "what each token looks for" so a per-head Q bias lets each head specialize its query.

## Mechanism & Design sketch
- **Files**: `models/layers.py` — locate the Q projection. Add `nn.Parameter(H, d_k)` initialized to 0. Add to Q after projection.
- **Config flag**: `use_q_head_bias: bool = False`, `q_head_bias_init: float = 0.0`.
- **Cost**: 4 × 16 × 12 = +768 params, +0.082% of 0.94M.
- **Why it should help at tiny1m3m**: per the closed 162-q-only-norm null, Q-side normalization didn't bind (with K raw). 232 is a *complementary* Q-side intervention — additive bias instead of normalization. The closed 165-k-only-norm null also nulled on the K-side. So 232 is testing whether the *additive* Q-side intervention binds where the *normalization* Q-side intervention (162) didn't. If the model wants to push Q toward specific values, an additive bias is more direct than normalization.
- **Why it might be null**: closed 152-attn-logit-bias null (per-head QK^T bias) didn't bind; 232 is on Q directly so it has a similar effect on QK^T (additive on the row-direction of QK^T). Likely null by the same reasoning.

## Scale evidence
Asymmetric Q/K interventions are novel at this scale; the closest is tied-QK (closed line 23). Transfer-risk **med** (architecturally simple, novel combination).

## Why it's worth a slot
A win would say the model wants a Q-side-only bias axis (different from the closed normalization on Q or logit-bias on QK^T). A null confirms the per-head Q-bias family is closed at 0.94M. The lever is cheap (+768 params, ~10 LoC), bit-identical step 0, and structurally novel from the closed 152, 162, 165 family.
