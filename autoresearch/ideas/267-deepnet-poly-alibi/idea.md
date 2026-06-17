---
id: 267-deepnet-poly-alibi
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T04:27:12Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_poly_alibi (convex quad distance). 230 was -0.0111 best single seed but 3-seed confirm NULL on alibi alone. Retest on deepnet-augmented base.
---

# 267 — deepnet + use_poly_alibi on alibi

Stack on the new champion. 230-poly-alibi had best single seed -0.0111 on alibi but 3-seed confirm was NULL (mean +0.0017). Now testing on the deepnet-augmented base. +48 params (per-block slopes).

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_267-deepnet-poly-alibi.py` for inline config.
