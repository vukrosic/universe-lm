---
id: 103-momentum-streams
status: needs-plan
round: 1
updated: 2026-06-11T01:18:41Z
transfer-risk: med
---

# 103 — Momentum Streams

## Source
Gai, Huang, and Wu, "Momentum Streams for Optimizer-Inspired Transformers" (arXiv:2605.24425).

## Mechanism
Recast the residual update as optimizer-like dynamics and build transformer variants that mirror triple momentum, AdamW, Muon, or SOAP inside the architecture itself.

## Scale evidence
The paper says its triple-momentum TMMFormer beats vanilla and prior variants under matched compute, and the ablation points to momentum as the main source of gain. That makes the architectural translation more than a naming trick.

## Why it's worth a slot
This is the most "optimizer becomes architecture" idea in the batch. If it works, it tells us momentum structure belongs in the model, not only in the optimizer.
