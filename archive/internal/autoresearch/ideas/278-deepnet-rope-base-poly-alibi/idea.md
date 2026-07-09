---
id: 278-deepnet-rope-base-poly-alibi
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T06:05:30Z
transfer-risk: low
plain: 3-way stack: alibi + deepnet + rope-base + poly-alibi. Combines the two best 1-seeds (264 and 267). Test if they compound.
---

# 278 — deepnet + rope-base + poly-alibi (3-way) on alibi

3-way stack on the two best 1-seeds. 264 (deepnet+rope-base, val 6.2172) and 267 (deepnet+poly-alibi, val 6.2147) both bind through deepnet. Test if they compound.

A/B vs the 1-seed best 6.2147, band 0.04, WIN < 6.1747. Single seed (42).

See `_arq_278-deepnet-rope-base-poly-alibi.py` for inline config.
