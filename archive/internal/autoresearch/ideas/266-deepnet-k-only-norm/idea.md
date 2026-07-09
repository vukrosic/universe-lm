---
id: 266-deepnet-k-only-norm
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T04:19:56Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_k_only_norm. K-side pre-softmax RMSNorm. 165 was null on base (Δ-0.0293 borderline). Tests if asymmetric K-norm binds through deepnet.
---

# 266 — deepnet + K-only RMSNorm on alibi

Stack on the new champion. 165-k-only-norm was null on base (Δ-0.0293 borderline). 260 was staged in needs-plan for the alibi base; now testing on the deepnet-augmented base.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_266-deepnet-k-only-norm.py` for inline config.
