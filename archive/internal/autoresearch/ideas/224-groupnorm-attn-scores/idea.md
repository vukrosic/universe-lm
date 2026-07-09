---
id: 224-groupnorm-attn-scores
status: needs-taste
round: 1
updated: 2026-06-16T01:00:00Z
transfer-risk: med
plain: Apply Group Normalization (with learnable per-group affine) to the raw QK attention scores, *before* the softmax. Different from softmax itself (normalizes a distribution) and different from RMSNorm on Q/K (normalizes per-token), this normalizes per-row across the heads/tokens of each group.
---

# 224 — GroupNorm on Attention Scores (pre-softmax)

## Source
GroupNorm (Wu & He 2020, arXiv:1803.08314) divides channels into groups and normalizes within each group. Typically used in CNNs for small-batch regimes. The novel application here is on attention scores: `GroupNorm1d(attn_scores)` before the softmax, with groups along the head axis (so each head's scores are normalized together across token positions within a sample).

Related work:
- 195-qk-clamp-min-max null at 0.94M clamps QK scores pre-softmax (different operation: clamp vs normalize).
- 016-qk-norm WIN applies RMSNorm to Q and K separately (normalizes per-token, not per-score-matrix).
- 173-entmax-15 reject (sparsemax family) tested a different softmax-replacement mechanism and exhausted recodes.

224 is structurally different from all three: it normalizes the score matrix `[B, H, T, T]` along the head axis using a learnable group-norm with `num_groups = H` (one group per head), then computes softmax as usual. The intuition: out of H=4 heads, some have larger QK scales than others — GroupNorm re-centers them so all heads contribute equally to softmax mass.

## Mechanism
Standard attention: `softmax(QK^T / sqrt(d_k)) V`.
With 224: `softmax(GroupNorm(QK^T / sqrt(d_k))) V` where `GroupNorm` is applied along the last dim (token axis) *and* along the head axis via grouping.

Specifically:
```
scores = Q @ K.transpose(-1, -2) / sqrt(d_k)            # [B, H, T, T]
scores = scores.permute(0, 2, 3, 1)                     # [B, T, T, H]
scores = GroupNorm_h_per_group(H, scores, num_groups=H) # normalize across heads per (b, t, t) position
scores = scores.permute(0, 3, 1, 2)                     # [B, H, T, T]
attn = softmax(scores, dim=-1) @ V
```

The GroupNorm has per-group affine (`gain`, `bias`) initialized at `gain=1, bias=0` so the operation is the identity at step 0 ⇒ step-0 bit-identical to baseline softmax. After step 1, the gain/bias can move to normalize the cross-head variance.

## Design sketch
- **Files**: `models/layers.py` — locate the manual attention branch (similar to where 016-qk-norm sits). Apply `nn.GroupNorm(num_groups=H, num_channels=H)` along the head axis after the QK dot product and before softmax. The `num_channels=H=4` dimension is small, so the GroupNorm has 2*H=8 params (gain+bias per group) per block × 12 blocks = +96 params, +0.010% of 0.94M.
- **Config flag**: `use_groupnorm_attn_scores: bool = False`, `groupnorm_attn_num_groups: int = 4` (default = n_heads; can also be `H//2=2` for finer grouping). Init `weight=1, bias=0`.
- **Why it should help at tiny1m3m**: at 0.94M/12L/4H, the 4 attention heads have very different QK-magnitude profiles (per closed 152-attn-logit-bias null where per-head bias was tried — different axis but same observation that heads specialize). GroupNorm gives a *soft* constraint that all heads contribute roughly equal softmax mass, similar to how 175-alibi-slopes WIN gives each head a different distance bias. Different mechanism, similar motivation.
- **Why it might be null**: at H=4 with only 92 update steps, GroupNorm's 8 params per block may not have enough data to learn a useful cross-head scale. The closed 016-qk-norm WIN already normalizes Q and K individually (per-token, per-feature); the additional per-head cross-normalization may be redundant.

## Scale evidence
GroupNorm is well-validated for CNNs at small-batch regimes (Wu & He 2020). Its application to attention scores is novel; no direct prior. Transfer-risk **med** because the mechanism itself is well-understood (GroupNorm) but the application is new.

## Why it's worth a slot
A win would say the model benefits from cross-head score normalization, providing a complementary axis to 016-qk-norm (which normalizes pre-dot-product). A null closes the cross-head softmax-normalization axis at 0.94M. The lever is cheap (+96 params, ~15 LoC), bit-identical step 0, and structurally different from all prior closed attention-shape levers (152 bias, 155 temp, 166 RPE, 195 clamp, 016 norm — all act on Q or K individually, not on the score matrix).
