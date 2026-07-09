# Coding Model Leaderboard

Pass@1 on standard coding benchmarks. Only models evaluated through `evals/run_baseline.py` count.

| model | size | humaneval | mbpp | date | run_id | notes |
|---|---|---|---|---|---|---|
| (empty) |  |  |  |  |  | Add a row after the first successful eval |

## How to add a row

1. Run the eval: `python -m evals.run_baseline --model <hf-id-or-path> --device mps|cuda`
2. Open `evals/results/report__<model>__<timestamp>.json`
3. Copy `pass@1` from each eval into a new row
4. Commit with the report JSON

## Reference targets (from public papers, for comparison only)

These are *not* run through our harness. Use them as sanity checks for what pass@1 numbers a real model of this size achieves.

| model | size | humaneval | mbpp |
|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct | 0.5B | ~0.40 | ~0.55 |
| Qwen2.5-Coder-1.5B-Instruct | 1.5B | ~0.65 | ~0.70 |
| DeepSeek-Coder-1.3B-Instruct | 1.3B | ~0.65 | ~0.65 |
| Qwen2.5-Coder-3B-Instruct | 3B | ~0.78 | ~0.80 |
| DeepSeek-Coder-6.7B-Instruct | 6.7B | ~0.78 | ~0.74 |

Real numbers from your harness will vary. The point is not to match these — the point is to beat the same-size reference on your own run.
