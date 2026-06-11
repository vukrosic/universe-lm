---
id: 098-lp-qknorm
status: needs-plan
round: 1
updated: 2026-06-11T01:18:17Z
transfer-risk: low
---

# 098 — Lp QK-Norm

## Source
Lopez-Rubio, Montes-Perez, and Palomo, "Enhanced QKNorm normalization for neural transformers with the Lp norm" (arXiv:2602.05006).

## Mechanism
Generalize QK-Norm from L2 to Lp, so the attention geometry can tune "spikiness" through the norm choice itself instead of only through the temperature.

## Scale evidence
This is preliminary, but it is still a direct extension of a lever that already has a win in the queue. That makes it a low-friction follow-up rather than a brand-new theory branch.

## Why it's worth a slot
If 016-qk-norm helps because geometry matters, then changing the geometry family should matter too. This is the neatest way to probe that without changing the rest of the attention stack.
