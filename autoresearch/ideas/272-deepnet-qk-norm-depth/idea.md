---
id: 272-deepnet-qk-norm-depth
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T04:51:37Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_qk_norm_depth (depth-conditional QK norm). 169 was null on alibi (Δ-0.020). Retest on deepnet-augmented base.
---

# 272 — deepnet + use_qk_norm_depth on alibi

Stack on the new champion. 169-qk-norm-depth was null on alibi (Δ-0.020 inside band). Now testing on the deepnet-augmented base. 169 had 12 shared scalars (one per layer) on the symmetric QK axis.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_272-deepnet-qk-norm-depth.py` for inline config.
