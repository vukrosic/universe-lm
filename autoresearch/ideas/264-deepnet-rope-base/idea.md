---
id: 264-deepnet-rope-base
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T04:09:40Z
transfer-risk: low
plain: Stack on new champion (deepnet-alpha): alibi + deepnet + per-head RoPE base. Tests whether per-head rope specialization compounds with deepnet's residual conditioning.
---

# 264 — deepnet + per-head RoPE base on alibi

Stack on the new champion (deepnet-alpha). 172-per-head-rope-base was null on base (Δ+0.0109). 258 was staged in needs-plan for the alibi base; now testing on the deepnet-augmented base.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_264-deepnet-rope-base.py` for inline config.
