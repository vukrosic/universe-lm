---
id: 026-fire-x-qknorm
status: needs-run
round: 2
updated: 2026-06-10T12:33:38Z
transfer-risk: low
---

# 026 — FIRE × QK-Norm (positional bias + per-head Q,K normalization)

## Source
Internal composition of two in-stack levers:
- `autoresearch/ideas/009-fire-pe/evidence.md` — WIN, Δ −0.064/−0.082 at tiny1m3m (seed 42)
- `autoresearch/ideas/016-qk-norm/evidence.md` — WIN, Δ −0.0138/−0.0185 at tiny1m3m (seed 42)

## Mechanism
Enable both `use_fire_pe` and `use_qk_norm` simultaneously. In `models/layers.py` MultiHeadAttention forward: (1) QK-Norm applies `nn.LayerNorm(d_head)` to Q and K along the head-dim axis before the dot product, bounding per-head logit magnitude to `sqrt(d_head)`; (2) FIRE adds its learned input-dependent positional bias (`γ(i−j)·f(φ(x_i),φ(x_j))`) to the post-dot-product logits. These are sequential operations at well-separated points in the forward pass with no shared state — enabling both is a two-flag change.

## Scale evidence
- FIRE (009): demonstrated WIN at tiny1m3m (0.94M, 3M tokens); NeurIPS 2023 paper (Li et al., arXiv:2306.02613) on LM benchmarks at multiple scales. transfer-risk: low.
- QK-Norm (016): demonstrated WIN at tiny1m3m; adopted by Qwen3 (all sizes), SmolLM3, Google 22B ViT (Dehghani et al., arXiv:2302.05442). transfer-risk: low.

## Why it's worth a slot
009 is the largest single-lever win in the pipeline (−0.064); 016 is tied for second (−0.014). Stacking them tests whether QKNorm's per-head logit bounding makes FIRE's learned positional bias more consistent (superadditive, e.g. −0.09+) or whether the two mechanisms are already orthogonal and simply add (additive, expected −0.078). A null or interference result flags that both levers are hunting the same attention-logit headroom and the recipe stack is already saturated in the attention domain at this scale.
