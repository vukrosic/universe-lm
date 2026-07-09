---
id: 280-deepnet-poly-alibi-rope-base-qk-layernorm
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T07:47:01Z
transfer-risk: low
plain: 5-flag stack: alibi + deepnet + poly-alibi + rope-base + qk-layernorm. Combines 278's 4-flag (val 6.2075) + 273's qk-layernorm.
---

# 280 — 5-flag stack on alibi champion

Combines the 4-flag 278 (alibi+deepnet+poly-alibi+rope-base, val 6.2075) with 273's qk-layernorm lever. Goal: break 6.20 record.

A/B vs champion 6.2209, band 0.04, WIN < 6.1809. Single seed (42).

See `_arq_280-deepnet-poly-alibi-rope-base-qk-layernorm.py` for inline config.
