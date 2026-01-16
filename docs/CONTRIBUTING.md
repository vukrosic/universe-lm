For the code to be accepted it must be measured and break the record. 

Previously we were adding more features / code / architectures but records weren't getting broken and there was no progress.

Consider more code = bad (complexity, bloat, maintenance, bugs when upgrading), unless there is a new record in the training speed / loss, which justifies adding code.



How to contribute:

1. Pick a topic / task from [issues](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/issues) (issues are general name for tasks), carefully read it and understand it
2. Fork the repo
3. Clone it and implement the experiment, follow README
4. **Benchmark your changes** against a the baseline that you also measured beforehand. If hardware is limited, use free GPUs (Lightning AI, Colab) and reduce model size (`n_layer`, `n_embd`) for testing.
5. Submit a PR with your findings and comparison data.


Please check [LEADERBOARD](LEADERBOARD.md) for architecture, records and contribution history.