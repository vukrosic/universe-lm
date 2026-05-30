# Speedrun Leaderboard

Race to the **lowest val loss at a fixed token budget.** Rules: [docs/PLAN.md](docs/PLAN.md). How to enter: [CONTRIBUTING.md](CONTRIBUTING.md).

**Pinned for every entry:** `seed=42` · bf16 · A new entry takes the record only if it beats the standing best by **≥0.01** val loss. `time` is metadata, not ranked.

> **Older GPUs without native bf16 (e.g. Colab T4):** not banned — submit anyway, flagged as such. bf16 may run emulated with slightly different numerics; we'll decide how to rank these the first time one is submitted.

## Toy — 32,768 tokens (workflow only, NOT science)

The `toy` preset (3.24M params, 94% embedding) exists to debug the
train→eval→log→compare→promote loop in seconds, not to measure mechanisms.
Screens can flip sign at this size — never promote a toy result to 135M.

**Measured noise floor: ±~0.007 val loss** (two seed-42 runs: 9.3837, 9.3769,
bf16/CUDA nondeterminism). An experiment must beat baseline by **more than the
noise floor** to mean anything — here that's effectively the full 0.01 margin.

| # | Val loss | Who | Batch | Time | GPU | Evidence |
|---|---|---|---|---|---|---|
| 0 | 9.3769 | baseline | 2 | 14s | RTX 5060 | `--config toy --seed 42`, [metrics](runs/toy/toy_8step_seed42.json) |

## 135M — 2.7B tokens

| # | Val loss | Who | Batch | Time | GPU | Evidence |
|---|---|---|---|---|---|---|
| — | _baseline TBD_ | — | — | — | — | — |



---

`Evidence` links to the entrant's frozen branch (config + seed + loss log + `final_metrics.json` + checkpoint hash).*

This will be used to release useful 135M LLMs, which will be trained on a lot more tokens than this benchmark.