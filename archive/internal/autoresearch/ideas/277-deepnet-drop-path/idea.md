---
id: 277-deepnet-drop-path
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T05:16:58Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_drop_path. Per-block stochastic depth. Step-0-active.
---

# 277 — deepnet + use_drop_path on alibi

Stack on the new champion. DropPath: each sublayer's residual contribution is randomly gated to zero during training. Step-0-active.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_277-deepnet-drop-path.py` for inline config.
