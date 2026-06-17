---
id: 274-deepnet-qk-norm-post-rope
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T05:02:43Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_qk_norm_post_rope (QK norm applied AFTER RoPE rotation, distinct from 273 which is symmetric QK LayerNorm).
---

# 274 — deepnet + use_qk_norm_post_rope on alibi

Stack on the new champion. use_qk_norm_post_rope applies QK norm AFTER the RoPE rotation. Distinct from 273 (use_qk_layernorm = symmetric QK LayerNorm pre-softmax).

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_274-deepnet-qk-norm-post-rope.py` for inline config.
