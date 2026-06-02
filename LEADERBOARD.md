# Speedrun Leaderboard

Race to the **lowest val loss on the `10m` model** (Full10M200M — ~10M params, 200M tokens).
Pinned: `seed=42` · bf16. Beat the standing record by **≥0.01** to take it.

**Acceptance rule:** `10m` is the target — only a win here is a real record. Smaller configs
(`screen16m`, `screen10m`) are just for quick experimentation: use them to find promising
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
`4,000 steps × 4,096 tokens/step = 16,384,000 tokens`. Every row below is the eval at **step
4,000** (16,384,000 tokens) so all comparisons are apples-to-apples.

Reproduce a direct screen run with `--config screen10m --train_tokens 16384000 --seed 42`.

| # | Val loss | Δ vs ctrl | Run | Summary | Date | Evidence |
|---|---|---|---|---|---|---|
| 0 | **4.7875** | -0.2203 | vq-embeddings | V+Q combo (`#32`) at step 4,000. The V-embed + Q-embed projections are added together (24 layers × (q_size 144 + kv_size 48) × emb_rank 48 = 221,184 extra params, +2.9%). Beats V-embed alone at 4k (4.9381) by **0.1506**. Value at 4k taken from `s_vqembed_full`'s milestone history (the run continued to natural end, row 0 of the tier below). | 2026-06-02 | `logs/s_vqembed_full.log` · milestone @ step 4000 |
| 1 | 4.8722 | -0.1356 | key-embeddings | K-embed (`#31`) at step 4,000 (taken from `s_kembed_full`'s milestone history). Fastest warmup of the Q/K/V family at this checkpoint. | 2026-06-02 | `logs/s_kembed_full.log` · milestone @ step 4000 |
| 2 | 4.8834 | -0.1244 | output-embeddings | O-embed (`#33`) at step 4,000 (from `s_oembed_full` history). Inject `e_j` into attention **output** (post-O), bypasses the attention computation entirely. At 4k it's competitive with the V/Q/K family — token signal works at 4k regardless of where it lands. | 2026-06-02 | `logs/s_oembed_full.log` · milestone @ step 4000 |
| 3 | 4.9009 | -0.1069 | key-embeddings (gated) | K-embed (`#31`) gated run, evaluated at step 4,000. ~0.03 worse than the milestone-history read of the same arch (4.8722) — same seed, same step, run-to-run noise. | 2026-06-02 | `runs/s_kembed/metrics.json` |
| 4 | 4.9381 | -0.0697 | value-embeddings (gated) | V-embed (`#29`) gated run, evaluated at step 4,000. The architecture: inject factorized token embedding into attention V at every layer via a zero-init Muon projection `V += W·ve` (~55k extra params, +0.7%). [tutorial](docs/tutorials/value_embeddings/README.md) | 2026-06-02 | `runs/s_valembed/gate.pt` |
| 5 | 5.0078 | 0 | control | Plain `Screen10M20MConfig` (no embed flags), gated at step 4,000. seed=42, bf16, batch=2. 13.8 min on RTX 3050. | 2026-06-02 | `runs/s_ctrl/metrics.json` |

**Missing from this tier (would need a `--stop_at_step 4000` rerun):** Q-embed and a clean
V-only gated run. Q-embed and V-embed-natural-end have step 4,000 evals inside their
natural-end run histories (4.8607 and 4.9381 respectively) but a clean apples-to-apples
read needs a gated rerun. Will fill when the next probe needs them.

## `screen20m` — 10M · 20M tokens · step 4,882 · **natural-end tier**

Same screen configs as above but trained to the **natural end** (step 4,882 ≈ 20M tokens).
This is where the "real" comparison lives — the 4k tier measures warmup speed, this tier
measures end-of-training quality. **Pending:** a control at the natural end (currently
gated at 4k) so the Δ-vs-ctrl column is honest.

| # | Val loss | Run | Summary | Date | Evidence |
|---|---|---|---|---|---|
| 0 | **4.7428** | vq-embeddings | V+Q combo (`#32`) at natural end, 4,883 steps. The screen's first natural-end winner under the V/Q/K/V+Q family. Beats V-embed alone (row 1) by **0.0300** and K-embed (row 3) by 0.0800. The 0.0300 V+Q → V delta is inside the 0.06-0.16 V-embed noise band, so single-seed certainty is weak — but the curve never crosses (V+Q is better at every step from 500 onward) and the direction matches the additive hypothesis (Q's warmup edge + V's end-game edge). | 2026-06-02 | `runs/s_vqembed_full/metrics.json` · `logs/s_vqembed_full.log` |
| 1 | 4.7728 | value-embeddings | V-embed (`#29`) at natural end, 4,882 steps. Beats K-embed (row 3) by 0.0500 and Q-embed (row 2) by 0.0431, but loses to V+Q (row 0) by 0.0300. | 2026-06-02 | `runs/s_valembed_full/metrics.json` · `logs/s_valembed_full.log` |
| 2 | 4.8159 | query-embeddings | Q-embed (`#30`) at natural end. Re-run (s_qembed) gave 4.8753 — same seed, different result, so single-seed Q values are inside the 0.06 noise band. Best Q read is 4.8159. | 2026-06-02 | `runs/s_qembed_4k/metrics.json` · `logs/s_qembed_4k.log` |
| 3 | 4.8228 | key-embeddings | K-embed (`#31`) at natural end, 4,882 steps. Same pattern as Q-embed: K is essentially tied with Q (4.8228 vs 4.8159, inside noise). | 2026-06-02 | `runs/s_kembed_full/metrics.json` · `logs/s_kembed_full.log` |
| 4 | 4.8350 | output-embeddings | O-embed (`#33`) at natural end, 4,883 steps. The "fundamentally different lever" probe: inject `e_j` into attention **output** (post-O), bypasses the score computation entirely. **Underperforms all of V/Q/K** (worst natural-end of the embed family), but is still way better than control. Lesson: the token-signal win is **inside attention** (V/Q/K), not in the residual. The signal needs to be inside the attention computation to earn its keep. | 2026-06-02 | `runs/s_oembed_full/metrics.json` · `logs/s_oembed_full.log` |
| 5 | **pending** | control | A `Screen10M20MConfig` (no embed flags) run to step 4,882 — running on the box as `s_ctrl_full`, ~17 min from launch. Will fill this row. | — | — |

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
