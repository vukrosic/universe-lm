---
id: 093-derf
status: needs-plan
round: 1
updated: 2026-06-11T01:17:53Z
transfer-risk: med
---

# 093 — Derf (erf pointwise norm replacement)

## Source
Chen et al., "Stronger Normalization-Free Transformers" (arXiv:2512.10938). The paper's best design is Derf: a pointwise erf-based replacement for normalization.

## Mechanism
Swap each LayerNorm/RMSNorm site for a learned pointwise function of the form `erf(αx + s)`. This keeps the module cheap and local while giving the network a bounded, smooth squashing curve instead of token-statistic normalization.

## Scale evidence
The source reports Derf outperforming LayerNorm, RMSNorm, and DyT across vision, speech, and DNA sequence modeling after a large-scale search. That is a strong sign that the exact shape of the saturating nonlinearity matters, not just whether the layer has statistics or not.

## Why it's worth a slot
We already have a closed tanh-squash result in the queue history, but Derf is not just "tanh again." The erf curve is smoother and empirically stronger in the source paper, so it is a distinct mechanism worth testing if we want one representative from the norm-free family.
