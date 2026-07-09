---
id: 086-muddformer
status: needs-plan
round: 1
updated: 2026-06-11T01:17:26Z
transfer-risk: low
---

# 086 — MUDDFormer

## Source
MUDDFormer: Breaking Residual Bottlenecks in Transformers via MUDD Connections (arXiv:2502.12170). 2025.

## Mechanism
Add lightweight MUDD connections to break residual bottlenecks and let the model mix information more freely across layers without changing the overall transformer shell. This is a structural residual-path modification, so it is a good fit for a small config-gated ablation.

## Scale evidence
The paper reports experiments from a 405M model on 7B tokens up to 2.8B models on 300B tokens, and says the method reaches roughly 1.8x–2.4x compute-equivalent performance gains. `transfer-risk: low` because the improvement is seen across the full scale range, not only in toy settings.

## Why it's worth a slot
If the tiny model benefits, it would tell us that residual-path bottlenecks are a real lever in this stack, not just a large-model convenience.
