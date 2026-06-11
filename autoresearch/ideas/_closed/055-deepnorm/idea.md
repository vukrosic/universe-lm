---
id: 055-deepnorm
status: rejected
round: 1
updated: 2026-06-10T16:51:08Z
transfer-risk: low
---

# 055 — DeepNorm (depth-derived residual scaling)

## Source
Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers" (arXiv:2203.00555). The paper introduces DeepNorm plus a theoretically derived initialization to stabilize extremely deep Transformers.

## Mechanism
Scale the attention and FFN residual branches with the paper's depth-dependent constants and pair that with the matching initialization so the update magnitude stays bounded as depth grows. In this repo the 12-layer model just gets a fixed set of DeepNorm branch multipliers and init tweaks; it is a deliberate non-identity init, not a hyperparameter sweep.

## Scale evidence
The paper scales Transformers to 1,000 layers, and its 200-layer 3.2B model beats a 48-layer 12B baseline by 5 BLEU on multilingual translation. transfer-risk: low — the scale evidence is strong and directly about deep transformer training.

## Why it's worth a slot
If DeepNorm helps at tiny1m3m, then the model is already paying for unstable update geometry; if it does not, the benefit is probably only realized in much deeper stacks.
