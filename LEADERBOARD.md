# Speedrun Leaderboard

Race to the **lowest val loss at a fixed token budget.** Rules: [docs/PLAN.md](docs/PLAN.md). How to enter: [CONTRIBUTING.md](CONTRIBUTING.md).

**Pinned for every entry:** `seed=42` · bf16 · A new entry takes the record only if it beats the standing best by **≥0.01** val loss. `time` is metadata, not ranked.

## 25M — 0.5B tokens

| # | Val loss | Who | Batch | Time | GPU | Evidence |
|---|---|---|---|---|---|---|
| — | _baseline TBD_ | — | — | — | — | — |

## 135M — 3B tokens

| # | Val loss | Who | Batch | Time | GPU | Evidence |
|---|---|---|---|---|---|---|
| — | _baseline TBD_ | — | — | — | — | — |

---

`Evidence` links to the entrant's frozen branch (config + seed + loss log + `final_metrics.json` + checkpoint hash).*

This will be used to release useful 135M LLMs, which will be trained on a lot more tokens than this benchmark.