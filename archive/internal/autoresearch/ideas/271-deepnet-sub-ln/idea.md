---
id: 271-deepnet-sub-ln
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T04:47:33Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_sub_ln (sandwich LN residual: x + LN(f(x))). 017-sub-ln was null on alibi. Retest on deepnet-augmented base.
---

# 271 — deepnet + use_sub_ln on alibi

Stack on the new champion. 017-sub-ln was null on alibi (Δ inside band). Now testing on the deepnet-augmented base.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_271-deepnet-sub-ln.py` for inline config.
