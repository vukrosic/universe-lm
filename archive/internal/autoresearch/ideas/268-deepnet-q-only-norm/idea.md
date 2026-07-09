---
id: 268-deepnet-q-only-norm
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T04:31:17Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_q_only_norm. Q-side pre-softmax RMSNorm. 162 was null on base (Δ-0.0043). Tests if asymmetric Q-norm binds through deepnet.
---

# 268 — deepnet + Q-only RMSNorm on alibi

Stack on the new champion. 162-q-only-norm was null on base (Δ-0.0043). Asymmetric to 266 (K-only); both test whether pre-softmax QK asymmetry binds through deepnet.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_268-deepnet-q-only-norm.py` for inline config.
