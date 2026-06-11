---
id: 059-admin
status: rejected
round: 1
updated: 2026-06-10T16:48:42Z
transfer-risk: med
---

# 059 — Admin (adaptive initialization for Transformers)

## Source
Liu et al., "Understanding the Difficulty of Training Transformers" (arXiv:2004.08249). The paper analyzes residual-branch instability and proposes ADMIN as an adaptive initialization method.

## Mechanism
Initialize the residual branches so the model starts with a controlled dependency on each branch, then allows that dependence to grow once training stabilizes. In code terms this is a depth-aware init and residual scaling recipe, not a schedule or optimizer swap.

## Scale evidence
The paper trains 60 encoder / 12 decoder layer Transformers and reports up to +2.5 BLEU over a 6-layer baseline, with 46.4 BLEU on WMT14 EN-FR with back-translation. transfer-risk: med — the evidence is strong but lives in NMT, so LM transfer is plausible rather than proven.

## Why it's worth a slot
Admin tests whether the model needs a smarter launch condition rather than a different steady-state normalization; a null would point away from init geometry as the main issue.
