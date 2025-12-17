
# Research / Engineering Contributions

## Adding QK Norm

2025-12-15 <br>
[@RohanKhanBD](https://github.com/RohanKhanBD) <br>
Model Config: GPU24GBMoE

| Type | Val Loss | Val Acc | Perplexity |
|----------|---------|------------|------------|
| Without QK Norm (Baseline) | 3.7395 | 34.97% | 42.08 |
| With QK Norm (New) | 3.6539 ✓ | 35.83% ✓ | 38.63 ✓ |

## Zero-compute expert

[Experiment](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/16) <br>
[@RohanKhanBD](https://github.com/RohanKhanBD) <br>
Small scale experiments didn't show zero-compute expert benefits the LLM. We might try this again with large scale Mixture of Experts.
