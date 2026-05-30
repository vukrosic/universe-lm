# Speedrun Leaderboard

Race to the **lowest val loss at a fixed token budget.** Rules: [docs/PLAN.md](docs/PLAN.md). How to enter: [CONTRIBUTING.md](CONTRIBUTING.md).

**Pinned for every entry:** `seed=42` · bf16 · A new entry takes the record only if it beats the standing best by **≥0.01** val loss. `time` is metadata, not ranked.

> **Older GPUs without native bf16 (e.g. Colab T4):** not banned — submit anyway, flagged as such. bf16 may run emulated with slightly different numerics; we'll decide how to rank these the first time one is submitted.

## 135M — 2.7B tokens

| # | Val loss | Who | Batch | Time | GPU | Evidence |
|---|---|---|---|---|---|---|
| 0 | **OPEN — needs a compute donor** | — | — | — | — | `--config 135m --seed 42` |

> **Why OPEN:** 135M × 2.7B tokens ≈ 3.6×10¹⁷ param-tokens — roughly **100 h on a single
> consumer GPU** (e.g. RTX 5070). Needs an **A100/H100-class or multi-GPU** donor to land
> in reasonable time. This is exactly the kind of run the contributor model exists for —
> if you have the compute, claim this row.



---

`Evidence` links to the entrant's frozen branch (config + seed + loss log + `final_metrics.json` + checkpoint hash).*

This will be used to release useful 135M LLMs, which will be trained on a lot more tokens than this benchmark.