---
id: 305-combo-no-warmup
status: done
round: 4
updated: 2026-06-16T19:07:51Z
transfer-risk: low
plain: drop LR warmup (warmup_ratio 0.02→0) — peak LR from step 0, more useful steps. Stacks on the 296 combo champion (val 6.1998).
---

# 305-combo-no-warmup

NEW axis: optimizer/LR conditioning (the locality axis is exhausted; optimizer-SWAP
is closed, but the peak-LR VALUE of the base Muon+AdamW was never swept). At a
92-step horizon peak LR is the highest-leverage knob and "learn faster in few steps"
is the meta-pattern that wins here. See `_arq_305-combo-no-warmup.py` docstring for mechanism + hypothesis.
LR fields verified wired: trainer.py reads config.muon_lr/config.adamw_lr directly.
A/B vs the 296 combo champion (val 6.1998), SCREEN band 0.02. Single seed (42).
