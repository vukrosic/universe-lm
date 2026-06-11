---
id: 050-performer
status: rejected
round: 1
updated: 2026-06-10T16:52:13Z
transfer-risk: med
---

# 050 — Performer

## Source
Choromanski et al., "Rethinking Attention with Performers" (arXiv:2009.14794), 2020.

## Mechanism
Approximate softmax attention with positive orthogonal random features (FAVOR+), turning the quadratic attention kernel into a linear-time feature map. In this repo, the clean ablation is a feature-map attention path mixed in from zero so the model begins as baseline and learns how much approximation it wants.

## Scale evidence
The paper evaluates pixel prediction, text models, and protein sequence modeling, and frames FAVOR+ as a scalable replacement for regular attention on large-scale tasks. Transfer-risk is `med` because the evidence is broad and principled, but the biggest public claims are cross-domain rather than a single canonical 100M+ LM checkpoint.

## Why it's worth a slot
This tests whether kernelized attention is already a good enough inductive bias for a tiny decoder-only LM, and a null would say exact dot-product attention still matters here.
