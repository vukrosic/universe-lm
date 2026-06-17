---
id: 279-deepnet-poly-alibi-qk-layernorm
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T06:12:03Z
transfer-risk: low
plain: 3-way stack: alibi + deepnet + poly-alibi + qk_layernorm. Combines 267 (best 1-seed) with 273 (the 016 WIN lever on base).
---

# 279 — deepnet + poly-alibi + qk_layernorm (3-way) on alibi

3-way stack. Combines 267 (deepnet+poly-alibi, 6.2147) with 273 (deepnet+qk_layernorm, 6.2225 — the 016-qk_norm WIN on base, retest on deepnet).

A/B vs 6.2147, band 0.04, WIN < 6.1747. Single seed (42).

See `_arq_279-deepnet-poly-alibi-qk-layernorm.py` for inline config.
