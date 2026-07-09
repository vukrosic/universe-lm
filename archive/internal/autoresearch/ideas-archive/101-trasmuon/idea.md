---
id: 101-trasmuon
status: needs-plan
round: 1
updated: 2026-06-11T01:18:41Z
transfer-risk: low
---

# 101 — TrasMuon

## Source
Cheng et al., "TrasMuon: Trust-Region Adaptive Scaling for Orthogonalized Momentum Optimizers" (arXiv:2602.13498).

## Mechanism
Keep Muon's orthogonalized direction, but add global RMS calibration and energy-based trust-region clipping so the update magnitude stays in a stable zone.

## Scale evidence
The paper reports faster convergence than baselines and strong stability even without warmup. That is exactly the kind of control we want if orthogonalization is good in direction but brittle in scale.

## Why it's worth a slot
This is the magnitude-control companion to MUD. If 100 says "orthogonalize cheaper," 101 says "orthogonalize, then make the scale safe."
