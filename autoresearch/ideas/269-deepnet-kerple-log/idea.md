---
id: 269-deepnet-kerple-log
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T04:37:23Z
transfer-risk: low
plain: Stack on new champion: alibi + deepnet + use_kerple_log (logarithmic distance decay, concave relative to alibi linear). 231 was +0.0449 worse on alibi alone. Retest on deepnet-augmented base.
---

# 269 — deepnet + use_kerple_log on alibi

Stack on the new champion. 231-kerple-log-alibi was worse on alibi (Δ+0.0449 wrong-sign). Now testing on the deepnet-augmented base.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_269-deepnet-kerple-log.py` for inline config.
