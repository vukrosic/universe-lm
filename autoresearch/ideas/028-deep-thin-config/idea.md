---
id: 028-deep-thin-config
status: needs-review
round: 1
updated: 2026-06-10T07:52:03Z
transfer-risk: low
---

# 028 — Deep-and-Thin Config (depth/width ratio swap at fixed param budget)

## Source
Ma et al., "MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases" (ICML 2024, arXiv:2402.14905). Reported +2.7% average gain at 125M and +4.3% at 350M from depth/width swap vs prior sub-billion SOTA, holding total parameters and training tokens fixed.

## Mechanism
At a fixed ~0.94M param budget, use a deeper-thinner architecture (more transformer layers, smaller d_model and d_ff, same total params). The hypothesis: each layer contributes a nonlinear transformation; more layers → more transformation steps → better representational depth per parameter. A/B is simply: baseline config (current n_layers, d_model) vs a new config class at ~same param count but with n_layers increased and d_model/d_ff scaled down proportionally to preserve the param budget. One new config class, ~30 LoC. Per-head d_head is held constant; n_heads scales with d_model.

## Scale evidence
MobileLLM paper (Ma et al., ICML 2024): explicit ablations at 125M and 350M in Table 2 showing depth-prioritized architectures beat width-prioritized architectures at same parameter count by 2.7–4.3% on zero-shot benchmarks. transfer-risk: low — gains demonstrated at exactly the target model class (sub-400M, from-scratch training).

## Why it's worth a slot
The litrev (`plans/litrev-sub200m.md` §3) flags this as the highest-impact structural lever in winning sub-400M recipes, ahead of any optimizer or attention variant. Winning here compounds with every other stack lever (deeper model + FIRE + QKNorm). A null at tiny1m3m would indicate the depth benefit only materialises at longer training horizons (>3M tokens), which is informative for the ladder decision at 10M+ tier.
