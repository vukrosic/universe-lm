---
id: 273-deepnet-qk-layernorm
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T04:58:38Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_qk_layernorm (the 016-qk_norm WIN lever on the base, retest on deepnet-augmented base). Symmetric pre-softmax QK LayerNorm.
---

# 273 — deepnet + use_qk_layernorm on alibi

Stack on the new champion. 016-qk_norm was the strongest 1-way WIN on the base (no alibi). Now testing on the deepnet-augmented alibi base.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_273-deepnet-qk-layernorm.py` for inline config.
