---
id: 275-deepnet-value-residual
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T05:08:49Z
transfer-risk: low
plain: 3-way stack: alibi + deepnet + use_value_residual. 021 was WIN on base; 208 was NULL on alibi alone. Test if deepnet's residual stream dynamics change value_residual's binding.
---

# 275 — deepnet + use_value_residual (3-way) on alibi

3-way stack on the new champion. 021-value-residual WIN on base (Δ-0.034) but 208-v-residual-alibi was NULL (Δ+0.0191 wrong-sign). Now testing the 3-way on alibi+deepnet-augmented base.

A/B vs new champion val 6.2367, band 0.04, WIN < 6.1967. Single seed (42).

See `_arq_275-deepnet-value-residual.py` for inline config.
