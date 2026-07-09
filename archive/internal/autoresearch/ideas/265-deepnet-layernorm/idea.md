---
id: 265-deepnet-layernorm
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T04:13:47Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_layernorm (RMSNorm→LayerNorm global swap). +1984 params. Tests if the global norm-swap binds through deepnet.
---

# 265 — deepnet + use_layernorm on alibi

Stack on the new champion. 017-sub-ln was null on base. 259 was staged in needs-plan for the alibi base; now testing on the deepnet-augmented base. Heaviest param overhead (+1984, +0.21%) of the 5 stack experiments.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_265-deepnet-layernorm.py` for inline config.
