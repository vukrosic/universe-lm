---
status: done
---
# 334-tied-output-mlp

A small MLP transform on the hidden state before the (tied) output embedding head: adds a learned non-linear readout between the final residual stream and the tied unembedding, an output-head mechanism distinct from the bare tied projection. NOVEL ARCHITECTURE (structural mechanism, EXPERIMENT-DESIGN RULE 0 — NOT hyperparameter search). Step-0 active on champion (probed, distinct diff), not in closed.md. Stacks on 323 champion. Seed 42. A/B vs champion 6.1720.
