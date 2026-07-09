---
id: 094-keel-highway-postln
status: needs-plan
round: 1
updated: 2026-06-11T01:18:16Z
transfer-risk: med
---

# 094 — Keel highway Post-LN

## Source
Chen and Wei, "Post-LayerNorm Is Back: Stable, Expressive, and Deep" (arXiv:2601.19895). Their Keel variant keeps Post-LN but swaps the residual pathway for a highway-style connection.

## Mechanism
Keep the post-norm layout, but replace the plain residual add with a gated highway connection that preserves gradient flow through very deep stacks. This is a depth-scaling lever, not a normalization tweak.

## Scale evidence
The paper reports stable training at depths beyond 1000 layers and better perplexity/depth-scaling behavior than Pre-LN. That is a strong hint that the residual path itself can be the bottleneck once width is no longer the main constraint.

## Why it's worth a slot
This is the most structural idea in the batch. If it works, it says the repo can get expressivity from depth without relying on the standard Pre-LN compromise. If it fails, we learn that the highway fix is too big a rewrite for the tiny1m3m regime.
