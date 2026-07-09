---
id: 304-combo-lr-0p7x
status: done
round: 4
updated: 2026-06-16T19:01:46Z
transfer-risk: low
plain: global peak LR ×0.7 (muon 0.0168/adamw 0.0042) — lower bracket to locate the LR optimum. Stacks on the 296 combo champion (val 6.1998).
---

# 304-combo-lr-0p7x

NEW axis: optimizer/LR conditioning (the locality axis is exhausted; optimizer-SWAP
is closed, but the peak-LR VALUE of the base Muon+AdamW was never swept). At a
92-step horizon peak LR is the highest-leverage knob and "learn faster in few steps"
is the meta-pattern that wins here. See `_arq_304-combo-lr-0p7x.py` docstring for mechanism + hypothesis.
LR fields verified wired: trainer.py reads config.muon_lr/config.adamw_lr directly.
A/B vs the 296 combo champion (val 6.1998), SCREEN band 0.02. Single seed (42).
