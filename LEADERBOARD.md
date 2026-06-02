# Speedrun Leaderboard

Race to the **lowest val loss on the `10m` model** (Full10M200M — ~10M params, 200M tokens).
Pinned: `seed=42` · bf16. Beat the standing record by **≥0.01** to take it.

**Acceptance rule:** `10m` is the target — only a win here is a real record. Smaller configs
(`screen3m`, `screen10m`) are just for quick experimentation: use them to find promising
mechanisms, but nothing counts until it beats the `10m` record. Hyperparameters go in the
**Run**/**Summary** text — no per-mechanism columns ([Parameter Golf](https://github.com/openai/parameter-golf) style).

## `10m` — Full10M200M · 10M · 200M tokens  ·  **the target**

| # | Val loss | Run | Author | Summary | Date | Evidence |
|---|---|---|---|---|---|---|
| 0 | **4.3011** | emb-factor-depth | vukrosic | Low-rank embedding (emb_rank=48) + depth 3→24 layers — the 10m architecture. Factorizes the 91%-of-params embedding `49152×144 → (49152×48)@(48×144)`, tied head, spends the freed ~4.7M params on depth. seed=42, bf16, batch=2, 200M tokens, 48,829 steps, 162m on RTX 3050. Reproduce: `--config 10m --seed 42`. | 2026-06-02 | commit `cbe5677` · tag `result/10m-emb-factor-depth` · [metrics](baselines/10m_baseline.json) |
| 1 | 4.5486 | warmup_decay_w002 | vukrosic | warmup_decay_to_zero, warmup_ratio=0.02, seed=42, bf16, 200M tokens, 48,829 steps. Frozen result tag: `result/issue30-warmup-decay-w002-10m`. | 2026-05-31 | commit `e1de876` · tag `result/issue30-warmup-decay-w002-10m` |
| 2 | 5.015 | baseline | vukrosic | QK-gain init=0 (learnable per-head scalar starting at 0) — *not* the plain model; reproduce from tag `baseline/10m`, not main's HEAD. seed=42, bf16, batch=2, 48,829 steps, 33m on RTX 5070. | 2026-05-30 | tag `baseline/10m` |

## `screen16m` — 10M · 16.384M tokens · step 4,000 · **speedrun screen**

Fast public screen for iteration. This ranks ideas before spending the full 200M-token run,
but it is not the final `10m` record. `16M` here means the exact comparison point
`4,000 steps × 4,096 tokens/step = 16,384,000 tokens`.

Reproduce a direct screen run with `--config screen10m --train_tokens 16384000 --seed 42`.

| # | Val loss | Run | Author | Summary | Date | Evidence |
|---|---|---|---|---|---|---|
| 0 | **4.7728** | value-embeddings (full) | vukrosic | Value embeddings (`#29`) run to the screen's natural endpoint (step 4,882, 20M tokens): **4.7728**. Beats the prior screen record (4.9381) by **0.1653** and the control (5.0088) by **0.2360**. Run-to-run variance with the same seed is ~0.16 (this run vs the original 4.9381 stop-at-4k run, both seed 42), so treat absolute numbers as ±0.10. seed=42, bf16, batch=2. **Screen only — the 4,882-step result, not a full-length champion.** Reproduce: `--config 10m --config_class configs.llm_config.Screen10M200MValueEmbedConfig --seed 42`. | 2026-06-02 | branch `exp/resid-levers` · `logs/s_valembed_full.log` |
| 1 | 4.8159 | query-embeddings (full) | vukrosic | Query embeddings (`#30`) run to the screen's natural endpoint: **4.8159**. **Does not beat V-embed at the endpoint** — Q learns faster early (better at steps 500, 1000) but V overtakes by step 1500 and ends lower (4.7728 vs 4.8159). For an end-of-training screen, V is the better pick. A second Q run gave 4.8753 — same seed, different result, so the comparison is in the run-to-run noise band. seed=42, bf16, batch=2. **Screen only.** Reproduce: `--config 10m --config_class configs.llm_config.Screen10M20MQueryEmbedConfig --seed 42`. | 2026-06-02 | branch `exp/resid-levers` · `logs/s_qembed.log`, `logs/s_qembed_4k.log` · tag `result/screen16m-query-embed-30` (superseded) |
| 2 | 4.8228 | key-embeddings (full) | vukrosic | Key embeddings (`#31`) run to the screen's natural endpoint: **4.8228**. K-embed has the **fastest warmup of all three** (best at 500, 1000, 4000) but loses to V at the natural end. Same pattern as Q-embed: K/Q win warmup, V wins endpoint. K is essentially tied with Q at the end (4.8228 vs 4.8159, inside noise). seed=42, bf16, batch=2. **Screen only.** Reproduce: `--config 10m --config_class configs.llm_config.Screen10M20MKeyEmbedConfig --seed 42`. | 2026-06-02 | branch `exp/resid-levers` · `logs/s_kembed_full.log` |
| 3 | 4.9381 | value-embeddings (stop@4k) | vukrosic | Value embeddings (`#29`), original `--stop_at_step 4000` run. The V-embed architecture — inject the factorized token embedding into attention V at every layer via a zero-init Muon projection `V += W·ve`, reusing the existing `emb_rank=48` table (~55k extra params, +0.7%). Beats the control (5.0088) by **0.0707**. Run was gated at step 4000 by the milestone eval there, so this number is the eval at 4000. Re-run to natural end is 4.7728 (row 0). seed=42, bf16, batch=2. [tutorial](docs/tutorials/value_embeddings/README.md) | 2026-06-02 | branch `exp/resid-levers` · `logs/s_valembed.log` |
| 4 | 5.0088 | emb-factor-depth | vukrosic | First screen record, taken from the current `10m` champion's intermediate eval at step 4,000 / 16,384,000 tokens. Low-rank embedding (emb_rank=48) + depth 3→24 layers, seed=42, bf16, batch=2. | 2026-06-02 | commit `cbe5677` · tag `result/10m-emb-factor-depth` · [metrics](baselines/10m_baseline.json) |

## Screens — quick experimentation (not records)

For finding promising mechanisms and reproducing baselines fast. Not ranked — only a `10m`
win counts. The **→ 10m** column ties each screen run to its real `10m` entry (or shows it
hasn't been validated there yet). Full QK-gain sweeps: [qk_leaderboard.md](qk_leaderboard.md).

**`screen10m`** — 10M · 20M tokens · `--config screen10m --seed 42`

| Val loss | Run | Time | → 10m |
|---|---|---|---|
| 4.9816 | QK-gain init=4.0 | 3m51s | not run at 10m yet |
| 5.2041 | baseline | 3m44s | ties to `10m` #0 (gain=0, 5.015) |

---

`Evidence` = frozen branch/tag plus metrics artifact (`final_metrics` for full runs,
`history` for screen capture points).

**The mission (not a race yet):** beat [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
with a fully-open 135M model. We can't train it yet — it needs ~100 h on a single consumer GPU
(or a bigger/multi-GPU box we don't have). The `135m` config is ready (`--config 135m`) for when
the compute is there; until then, the `10m` race finds the recipe that will get us there.
