---
id: 029-v-norm
status: running
round: 1
updated: 2026-06-10T12:28:10Z
transfer-risk: med
---

# 029 — V-Norm (per-head LayerNorm on Value projection)

## Source
Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities" (2023, arXiv:2309.14322). QK-Norm is the canonical form (016 is in the recipe stack); Value normalization is the symmetric partner: normalizing V vectors before the AV aggregation step prevents outlier values from dominating the attention output irrespective of attention weights.

## Mechanism
After the V projection and before the AV product in MultiHeadAttention, apply `nn.LayerNorm(d_head)` to V along the head-dim axis:

```
V = v_proj(x).view(B, H, T, d_head)
V = v_norm(V)             # per-head, per-token, LayerNorm along d_head
out = (A @ V).reshape(B, T, d_model)
```

`v_norm` is a separate `nn.LayerNorm(d_head)` (no weight sharing with the existing q_norm/k_norm from 016). Zero-init bias, unit-init weight at construction — identity at step 0 before training. ~15 LoC change in `models/layers.py` MultiHeadAttention, mirroring the existing QK-Norm code path (idea 016).

## Scale evidence
Wortsman et al. 2023: V-normalization reduces activation outliers in transformer training; used as a diagnostic proxy at ViT/BERT scale (hundreds of millions of parameters). Less direct LM-pretraining evidence at ≥100M than QK-Norm's adoption record (Qwen3, SmolLM3). transfer-risk: med — mechanistic argument is strong (symmetric to QK-Norm which won by Δ −0.014); direct LM evidence at 100M+ is thin.

## Why it's worth a slot
QK-Norm (016) is in the recipe stack at Δ −0.0138. V-Norm is the symmetric per-head LayerNorm on the remaining projection and costs identical code. The bet: the same "bound-the-per-head-magnitude" mechanism that helps Q and K also helps V, because large value vectors after the AV aggregation produce large residual updates that destabilize training at small model scale. A null would isolate the logit-bounding (Q,K side) as the causal lever and rule out output-value bounding — informative regardless.
