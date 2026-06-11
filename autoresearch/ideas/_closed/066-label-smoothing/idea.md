---
id: 066-label-smoothing
status: rejected
round: 1
updated: 2026-06-10T16:51:58Z
transfer-risk: low
---

# 066 — Label smoothing

## Source
Müller, Kornblith, and Hinton, "When Does Label Smoothing Help?" (arXiv:1906.02629). Jun 2019.

## Mechanism
Replace hard one-hot targets with a soft target distribution `(1 - ε)` on the gold token plus `ε` spread uniformly across the vocabulary. This is a tiny loss-wrapper change that directly trains softer output probabilities.

## Scale evidence
The paper shows improved generalization and calibration across language translation, speech recognition, and language modeling, with better beam-search behavior. `transfer-risk: low` because the method is simple, widely validated, and directly changes token targets rather than model shape.

## Why it's worth a slot
If tiny1m3m is overconfident, label smoothing is the simplest soft-target baseline to reveal whether the output head is too sharp for its own good.

