# Speedrun Leaderboard

Race to the **lowest val loss on the `10m` model** (Full10M200M — ~10M params, 200M tokens).
Pinned: `seed=42` · bf16. Beat the standing record by **≥0.01** to take it.

**Acceptance rule:** `10m` is the target — only a win here is a real record. Smaller configs
(`screen3m`, `screen10m`) are just for quick experimentation: use them to find promising
mechanisms, but nothing counts until it beats the `10m` record. Hyperparameters go in the
**Run**/**Summary** text — no per-mechanism columns ([Parameter Golf](https://github.com/openai/parameter-golf) style).

> **Proposed (not yet decided):** the ranked run stays `seed=42`, but before a result
> *takes the record* we may require confirming it on **2–3 extra seeds** (off-leaderboard)
> to prove the win isn't specific to seed 42's draw. Screening stays single-seed. Open
> question — not enforced yet.

## `10m` — Full10M200M · 10M · 200M tokens  ·  **the target**

| # | Val loss | Run | Author | Summary | Date | Evidence |
|---|---|---|---|---|---|---|
| 0 | 5.015 | baseline | vukrosic | QK-gain init=0 (learnable per-head scalar starting at 0) — *not* the plain model; reproduce from tag `baseline/10m`, not main's HEAD. seed=42, bf16, batch=2, 48,829 steps, 33m on RTX 5070. | 2026-05-30 | [metrics](baselines/10m_baseline.json) · tag `baseline/10m` |

## Screens — quick experimentation (not records)

For finding promising mechanisms and reproducing baselines fast. Not ranked — only a `10m`
win counts. The **→ 10m** column ties each screen run to its real `10m` entry (or shows it
hasn't been validated there yet). Full QK-gain sweeps: [qk_leaderboard.md](qk_leaderboard.md).

**`screen3m`** — 3.2M · 32k tokens · `--config screen3m --seed 42`

| Val loss | Run | Time | → 10m |
|---|---|---|---|
| 9.2894 | QK-gain init=2.2 | 9s | not run at 10m yet |
| 9.3769 | baseline | 13s | ties to `10m` #0 (gain=0, 5.015) |

**`screen10m`** — 10M · 20M tokens · `--config screen10m --seed 42`

| Val loss | Run | Time | → 10m |
|---|---|---|---|
| 4.9816 | QK-gain init=4.0 | 3m51s | not run at 10m yet |
| 5.2041 | baseline | 3m44s | ties to `10m` #0 (gain=0, 5.015) |

---

`Evidence` = frozen branch/tag (config + seed + commit + `final_metrics.json`).

**The mission (not a race yet):** beat [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
with a fully-open 135M model. We can't train it yet — it needs ~100 h on a single consumer GPU
(or a bigger/multi-GPU box we don't have). The `135m` config is ready (`--config 135m`) for when
the compute is there; until then, the `10m` race finds the recipe that will get us there.
