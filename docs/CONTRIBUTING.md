For the code to be accepted it must be measured and break the record. 

Previously we were adding more features / code / architectures but records weren't getting broken and there was no progress.

Consider more code = bad (complexity, bloat, maintenance, bugs when upgrading), unless there is a new record in the training speed / loss, which justifies adding code.



How to contribute:

1. Pick a topic / task from [issues](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/issues) (issues are general name for tasks), carefully read it and understand it
2. Fork the repo
3. Clone it and implement the experiment, follow README
4. **Benchmark your changes** against a the baseline that you also measured beforehand. If hardware is limited, use free GPUs (Lightning AI, Colab) and reduce model size (`n_layer`, `n_embd`) for testing.
5. Submit a PR with your findings and comparison data.


# Research / Engineering Contributions

Contributors list (growing): [@RohanKhanBD](https://github.com/RohanKhanBD) | [Shehab Ashraf](https://github.com/shehab-ashraf)

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

## Token Smearing

[Experiment](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/19) <br>
[Shehab Ashraf](https://github.com/shehab-ashraf) <br>
Token smearing directly mixes a small, learnable fraction of the previous token’s embedding into the current token so local context is handled cheaply, reducing the burden on attention for short-range dependencies.
- Currently not enough compute to experiment more extensively.