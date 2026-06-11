---
id: 102-sf-normuon
status: needs-plan
round: 1
updated: 2026-06-11T01:18:41Z
transfer-risk: low
---

# 102 — SF-NorMuon

## Source
Apte et al., "Anytime Training with Schedule-Free Spectral Optimization" (arXiv:2605.23061).

## Mechanism
Use a schedule-free spectral optimizer instead of a fixed-horizon LR schedule, with weight decay applied at the fast iterate for long-horizon stability.

## Scale evidence
The paper says a single configuration matches or beats tuned AdamW on 125M and 772M language models across 1-8x Chinchilla horizons. That is a good sign the method is not just a short-run trick.

## Why it's worth a slot
This is the cleanest "no schedule, still good" optimizer candidate in the recent batch. It is useful whenever we want a horizon-agnostic training loop with less manual tuning.
