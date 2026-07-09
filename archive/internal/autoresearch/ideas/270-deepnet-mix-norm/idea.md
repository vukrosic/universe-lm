---
id: 270-deepnet-mix-norm
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T04:41:27Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_mix_norm (per-block RMS+LN convex mixture, sigmoid(α)·RMS + (1-sigmoid(α))·LN). 217 was -0.0030 null on alibi. Retest on deepnet-augmented base.
---

# 270 — deepnet + use_mix_norm on alibi

Stack on the new champion. 217-mix-norm was null on alibi (Δ-0.0030 inside band). Now testing on the deepnet-augmented base. +1984 params (+0.21%).

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_270-deepnet-mix-norm.py` for inline config.
