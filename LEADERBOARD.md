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
| 0 | **4.7637** | -0.0850 | vo-embeddings | V+O combo (`#35`) at step 4,000 (from `s_voembed_full` history). V (inside attention) + O (post-O residual). 4k eval — best of the embed family at this checkpoint. | 2026-06-02 | `logs/s_voembed_full.log` · milestone @ step 4000 |
| 1 | 4.7875 | -0.0612 | vq-embeddings | V+Q combo (`#32`) at step 4,000. The V-embed + Q-embed projections are added together (24 layers × (q_size 144 + kv_size 48) × emb_rank 48 = 221,184 extra params, +2.9%). Beats V-embed alone at 4k (4.9381) by **0.1506**. Value at 4k taken from `s_vqembed_full`'s milestone history (the run continued to natural end, row 0 of the tier below). | 2026-06-02 | `logs/s_vqembed_full.log` · milestone @ step 4000 |
| 2 | 4.8722 | +0.0235 | key-embeddings | K-embed (`#31`) at step 4,000 (taken from `s_kembed_full`'s milestone history). Fastest warmup of the Q/K/V family at this checkpoint. | 2026-06-02 | `logs/s_kembed_full.log` · milestone @ step 4000 |
| 3 | 4.8834 | +0.0347 | output-embeddings | O-embed (`#33`) at step 4,000 (from `s_oembed_full` history). Inject `e_j` into attention **output** (post-O), bypasses the attention computation entirely. At 4k it's competitive with the V/Q/K family — token signal works at 4k regardless of where it lands. | 2026-06-02 | `logs/s_oembed_full.log` · milestone @ step 4000 |
| 4 | 4.9009 | +0.0522 | key-embeddings (gated) | K-embed (`#31`) gated run, evaluated at step 4,000. ~0.03 worse than the milestone-history read of the same arch (4.8722) — same seed, same step, run-to-run noise. | 2026-06-02 | `runs/s_kembed/metrics.json` |
| 5 | 4.9381 | +0.0894 | value-embeddings (gated) | V-embed (`#29`) gated run, evaluated at step 4,000. The architecture: inject factorized token embedding into attention V at every layer via a zero-init Muon projection `V += W·ve` (~55k extra params, +0.7%). [tutorial](docs/tutorials/value_embeddings/README.md) | 2026-06-02 | `runs/s_valembed/gate.pt` |
| 6 | 4.8487 | 0 | control | Plain `Screen10M20MConfig` (no embed flags) at step 4,000 (from `s_ctrl_full` history). 19m on RTX 3050, seed=42, bf16, batch=2. Apples-to-apples baseline. | 2026-06-02 | `runs/s_ctrl_full/metrics.json` · milestone @ step 4000 |

**Missing from this tier (would need a `--stop_at_step 4000` rerun):** Q-embed and a clean
V-only gated run. Q-embed and V-embed-natural-end have step 4,000 evals inside their
natural-end run histories (4.8607 and 4.9381 respectively) but a clean apples-to-apples
read needs a gated rerun. Will fill when the next probe needs them.

## `screen20m` — 10M · 20M tokens · step 4,882 · **natural-end tier**

Same screen configs as above but trained to the **natural end** (step 4,882 ≈ 20M tokens).
This is where the "real" comparison lives — the 4k tier measures warmup speed, this tier
measures end-of-training quality. **All rows below are at step 4,883 (apples-to-apples
on token count and training step).**

**Noise band warning:** run-to-run variance with the same seed is **0.06–0.16** for the
embed-family runs (e.g. Q-embed produced 4.8159 and 4.8753 on two runs with seed 42).
The 4k gated control (`s_ctrl`, 5.0078) and the natural-end control (`s_ctrl_full`,
4.8487 at the same step) differ by 0.16. **Single-seed deltas inside ±0.10 should be
treated as noise, not signal.** The 0.30–0.09 V-embed / V+Q deltas vs the *old* gated
control were illusory; vs the *fresh* control rerun, the picture is sharper.

| # | Val loss | Δ vs ctrl | Run | Summary | Date | Evidence |
|---|---|---|---|---|---|---|
| 0 | **4.6797** | -0.1187 | vq-gain | V-embed + per-head Q-gain (`#39`) at natural end, 4,883 steps. **New screen20m #0** (essentially tied with V+O+q_gain 4.6789). V (inside attention) + per-head Q-gain. **O-embed is redundant when combined with q_gain** — V+q_gain ≈ V+O+q_gain within noise. Cost: 24×(kv_size 48 × emb_rank 48) + 24×n_heads = 55,296 + 144 = 55,440 extra params. Beats V alone (4.7728) by 0.0931, beats control (4.7984) by 0.1187. Single-seed seed 42; multi-seed confirmation running. | 2026-06-02 | `runs/s_vqgain_full/metrics.json` · `logs/s_vqgain_full.log` |
| 1 | 4.6789 | -0.1195 | voq-embeddings | V+O + per-head Q-gain (`#38`) at natural end, 4,883 steps. **Multi-seed mean = 4.6789, std = 0.0048 across seeds 42/43/44** (4.6745, 4.6766, 4.6855) — exceptionally reproducible. Beats V+O (4.7188) by 0.0443, beats control (4.7984) by 0.1239. **But O-embed is now known to be redundant** — V+q_gain (row 0) gives the same result with simpler architecture. | 2026-06-02 | `runs/s_voqgain_full/metrics.json` · `runs/s_voqgain_s43/` · `runs/s_voqgain_s44/` |
| 2 | 4.7200 | -0.0784 | q-gain | **q_gain alone** (`#41`, no embeds) at natural end, 4,883 steps. **q_gain is a real standalone lever** — captures most of V+O's benefit. Beats control (4.7984) by 0.0784, beats V alone (4.7728) by 0.0528, essentially tied with V+O (4.7188). Per-head learnable scalar on Q (init zero) is a meaningful architectural change. | 2026-06-02 | `runs/s_qgain_full/metrics.json` · `logs/s_qgain_full.log` |
| 3 | 4.7188 | -0.0796 | vo-embeddings | V+O combo (`#35`) at natural end, 4,883 steps. V (inside attention) + O (post-O residual) — additive. Beats V+Q (4.7428) by 0.0240, beats V (4.7728) by 0.0540, beats control by 0.0796. | 2026-06-02 | `runs/s_voembed_full/metrics.json` · `logs/s_voembed_full.log` |
| 2 | 4.7294 | -0.0690 | vok-embeddings | V+O+K combo (`#36`) at natural end. K is **essentially neutral** in V+O context (4.7294 vs V+O's 4.7188, +0.0106, inside noise). K's anti-additive behavior is specific to V+Q context. | 2026-06-02 | `runs/s_vokembed_full/metrics.json` · `logs/s_vokembed_full.log` |
| 3 | 4.7428 | -0.0556 | vq-embeddings | V+Q combo (`#32`) at natural end, 4,883 steps. Beats V-embed alone (row 1 in tier 1) by 0.0300, control by 0.0556. Single-seed deltas are inside the noise band, but the direction is consistent and V+Q is better than V at every step from 500 onward. | 2026-06-02 | `runs/s_vqembed_full/metrics.json` · `logs/s_vqembed_full.log` |
| 4 | 4.7728 | -0.0256 | value-embeddings | V-embed (`#29`) at natural end, 4,883 steps. Beats control by 0.0256, but **this delta is inside the noise band** — single-seed V-embed is at most ~1σ above control. Curve never crosses control after step 1000. | 2026-06-02 | `runs/s_valembed_full/metrics.json` · `logs/s_valembed_full.log` |
| 5 | 4.7984 | 0 | control | Plain `Screen10M20MConfig` (no embed flags) at natural end, 4,883 steps. Fresh rerun, seed 42, bf16, batch 2, 19m on RTX 3050. The apples-to-apples baseline. Beats **Q/K/O-embed** at the natural end (they're slightly *worse* than control — 0.02-0.04 deltas, inside noise but the direction is the same as V's win direction). | 2026-06-02 | `runs/s_ctrl_full/metrics.json` · `logs/s_ctrl_full.log` |
| 6 | 4.8159 | +0.0175 | query-embeddings | Q-embed (`#30`) at natural end. Q-embed at 4k is competitive (4.8607, 4th place), but loses to control at the end. **The "Q wins warmup" story is a warmup story, not an end-game story** — by step 4883 the Q projection learns to do something slightly worse than nothing. | 2026-06-02 | `runs/s_qembed_4k/metrics.json` · `logs/s_qembed_4k.log` |
| 7 | 4.8228 | +0.0244 | key-embeddings | K-embed (`#31`) at natural end, 4,882 steps. Same pattern as Q-embed: K is essentially tied with Q (4.8228 vs 4.8159, inside noise), and slightly worse than control. K-embed is "Q's mirror" — both inject into score terms, both end up at the same operating point. | 2026-06-02 | `runs/s_kembed_full/metrics.json` · `logs/s_kembed_full.log` |
| 8 | 4.8250 | +0.0266 | vqk-embeddings | V+Q+K combo (`#34`) at natural end, 4,883 steps. **K is anti-additive** in the V+Q context. Adding K to V+Q drops val loss 0.0240 → 4.8250 (worse than V+Q alone by 0.0822, worse than V alone by 0.0522, worse than control by 0.0266). Hypothesis: the K projection creates a gradient conflict with V and Q, or with GQA the K signal is "too much" of the same identity. | 2026-06-02 | `runs/s_vqkembed_full/metrics.json` · `logs/s_vqkembed_full.log` |
| 9 | 4.8350 | +0.0366 | output-embeddings | O-embed (`#33`) at natural end, 4,883 steps. O-embed alone is the worst of the embed family. **But O-embed is highly additive with V** — see row 1 (V+O = 4.7188). O is a bad position in isolation, a great position when paired with V. | 2026-06-02 | `runs/s_oembed_full/metrics.json` · `logs/s_oembed_full.log` |

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
