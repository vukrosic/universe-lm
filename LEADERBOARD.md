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

**Multi-seed note:** V+q_gain (`#39`, row 0) and V+O+q_gain (`#38`, row 2) have **3-seed
confirmation** (seeds 42/43/44), with std 0.0057 and 0.0048 respectively. V+q+k_gain
(`#43`, row 1) is **3-seed mean 4.6949, std 0.0196** — anti-additive with V+q_gain.

| # | Val loss | Δ vs ctrl | Run | Summary | Date | Evidence |
|---|---|---|---|---|---|---|
| 0 | **4.6797** | -0.1187 | vq-gain | V-embed + per-head Q-gain (`#39`) at natural end, 4,883 steps. **Multi-seed mean = 4.6815, std = 0.0057** across seeds 42/43/44 (4.6797, 4.6889, 4.6758) — exceptionally reproducible. V (inside attention) + per-head Q-gain. **O-embed is redundant when combined with q_gain** — V+q_gain ≈ V+O+q_gain within noise. Beats V alone (4.7728) by 0.0931, beats control (4.7984) by 0.1187. | 2026-06-02 | `runs/s_vqgain_full/metrics.json` · `runs/s_vqgain_s43/` · `runs/s_vqgain_s44/` |
| 1 | 4.6949 | -0.1035 | vqk-gain | V-embed + per-head Q-gain + per-head K-gain (`#43`) at natural end. **3-seed mean = 4.6949, std = 0.0196** (4.6750, 4.7141, 4.6956). **K-gain is anti-additive in V+q context** — V+q+k_gain is 0.0134 *worse* than V+q_gain, well outside noise. K-gain alone is a real lever (4.7553), but adding it on top of V+q_gain hurts. | 2026-06-02 | `runs/s_vqkgain_full/metrics.json` · `runs/s_vqkgain_s43/` · `runs/s_vqkgain_s44/` |
| 2 | 4.6789 | -0.1195 | voq-embeddings | V+O + per-head Q-gain (`#38`) at natural end, 4,883 steps. **Multi-seed mean = 4.6789, std = 0.0048** across seeds 42/43/44 (4.6745, 4.6766, 4.6855). Beats V+O (4.7188) by 0.0399, beats control (4.7984) by 0.1195. **O-embed is now known to be redundant** — V+q_gain (row 0) gives the same result with simpler architecture. | 2026-06-02 | `runs/s_voqgain_full/metrics.json` · `runs/s_voqgain_s43/` · `runs/s_voqgain_s44/` |
| 3 | 4.7200 | -0.0784 | q-gain | **q_gain alone** (`#41`, no embeds) at natural end, 4,883 steps. **q_gain is a real standalone lever** — captures most of V+O's benefit. Beats control (4.7984) by 0.0784, beats V alone (4.7728) by 0.0528, essentially tied with V+O (4.7188). Per-head learnable scalar on Q (init zero) is a meaningful architectural change. | 2026-06-02 | `runs/s_qgain_full/metrics.json` · `logs/s_qgain_full.log` |
| 4 | 4.7188 | -0.0796 | vo-embeddings | V+O combo (`#35`) at natural end, 4,883 steps. V (inside attention) + O (post-O residual) — additive. Beats V+Q (4.7428) by 0.0240, beats V (4.7728) by 0.0540, beats control by 0.0796. | 2026-06-02 | `runs/s_voembed_full/metrics.json` · `logs/s_voembed_full.log` |
| 5 | 4.7169 | -0.0815 | vq+vgain | V+Q embed + per-head Q-gain (`#40`) at natural end, 4,883 steps. **Q-embed is redundant with V+q_gain** — V+Q+q_gain (4.7169) is *worse* than V+q_gain (4.6797) by 0.0372, **significantly outside noise**. Q-embed projects into the score side, q_gain is also a score-side lever; they conflict. | 2026-06-02 | `runs/s_vqqgain_full/metrics.json` · `logs/s_vqqgain_full.log` |
| 6 | 4.7259 | -0.0725 | qk-gain | q_gain + k_gain (`#44`, no embeds) at natural end. Both gains without embed: 4.7259 vs q_gain alone 4.7200 (+0.0059, essentially tied). K-gain is a real but weak lever. Adding K-gain on top of Q-gain gives essentially nothing when no embed is present. | 2026-06-02 | `runs/s_qkgain_full/metrics.json` · `logs/s_qkgain_full.log` |
| 7 | 4.7294 | -0.0690 | vok-embeddings | V+O+K combo (`#36`) at natural end. K is **essentially neutral** in V+O context (4.7294 vs V+O's 4.7188, +0.0106, inside noise). K's anti-additive behavior is specific to V+Q context. | 2026-06-02 | `runs/s_vokembed_full/metrics.json` · `logs/s_vokembed_full.log` |
| 8 | 4.7428 | -0.0556 | vq-embeddings | V+Q combo (`#32`) at natural end, 4,883 steps. Beats V-embed alone (row 1 in tier 1) by 0.0300, control by 0.0556. Single-seed deltas are inside the noise band, but the direction is consistent and V+Q is better than V at every step from 500 onward. | 2026-06-02 | `runs/s_vqembed_full/metrics.json` · `logs/s_vqembed_full.log` |
| 9 | 4.7478 | -0.0506 | deepv+vgain | Deep V-embed + per-head Q-gain (`#46`) at natural end. **Deep V+q_gain is anti-additive** — 4.7478 is 0.0681 worse than V+q_gain (4.6797), well outside noise. The non-linearity in V-embed conflicts with the q_gain lever. | 2026-06-02 | `runs/s_deepvqgain_full/metrics.json` · `logs/s_deepvqgain_full.log` |
| 10 | 4.7553 | -0.0431 | k-gain | k_gain alone (`#42`, no embeds) at natural end, 4,883 steps. K-gain is a real standalone lever, weaker than Q-gain (4.7553 vs 4.7200, gap of 0.0353). K-gain beats control by 0.0431 (just outside noise band of 0.06-0.16). Cost: 24 × n_kv_heads × 1 = 48 extra params (cheaper than q_gain due to GQA). | 2026-06-02 | `runs/s_kgain_full/metrics.json` · `logs/s_kgain_full.log` |
| 11 | 4.7594 | -0.0390 | deep-v-embed | 2-layer non-linear V-embed (`#45`) at natural end. V += GELU(ve @ W1) @ W2. **Beats linear V-embed (4.7728) by 0.0134** — the non-linearity has more capacity. But is anti-additive with q_gain (row 9). Cost: 221,184 extra params (+2.9%). | 2026-06-02 | `runs/s_deepv_full/metrics.json` · `logs/s_deepv_full.log` |
| 12 | 4.7728 | -0.0256 | value-embeddings | V-embed (`#29`) at natural end, 4,883 steps. Beats control by 0.0256, but **this delta is inside the noise band** — single-seed V-embed is at most ~1σ above control. Curve never crosses control after step 1000. | 2026-06-02 | `runs/s_valembed_full/metrics.json` · `logs/s_valembed_full.log` |
| 13 | 4.7984 | 0 | control | Plain `Screen10M20MConfig` (no embed flags) at natural end, 4,883 steps. Fresh rerun, seed 42, bf16, batch 2, 19m on RTX 3050. The apples-to-apples baseline. Beats **Q/K/O-embed** at the natural end (they're slightly *worse* than control — 0.02-0.04 deltas, inside noise but the direction is the same as V's win direction). | 2026-06-02 | `runs/s_ctrl_full/metrics.json` · `logs/s_ctrl_full.log` |
| 14 | 4.8159 | +0.0175 | query-embeddings | Q-embed (`#30`) at natural end. Q-embed at 4k is competitive (4.8607, 4th place), but loses to control at the end. **The "Q wins warmup" story is a warmup story, not an end-game story** — by step 4883 the Q projection learns to do something slightly worse than nothing. | 2026-06-02 | `runs/s_qembed_4k/metrics.json` · `logs/s_qembed_4k.log` |
| 15 | 4.8228 | +0.0244 | key-embeddings | K-embed (`#31`) at natural end, 4,882 steps. Same pattern as Q-embed: K is essentially tied with Q (4.8228 vs 4.8159, inside noise), and slightly worse than control. K-embed is "Q's mirror" — both inject into score terms, both end up at the same operating point. | 2026-06-02 | `runs/s_kembed_full/metrics.json` · `logs/s_kembed_full.log` |
| 16 | 4.8250 | +0.0266 | vqk-embeddings | V+Q+K combo (`#34`) at natural end, 4,883 steps. **K is anti-additive** in the V+Q context. Adding K to V+Q drops val loss 0.0240 → 4.8250 (worse than V+Q alone by 0.0822, worse than V alone by 0.0522, worse than control by 0.0266). Hypothesis: the K projection creates a gradient conflict with V and Q, or with GQA the K signal is "too much" of the same identity. | 2026-06-02 | `runs/s_vqkembed_full/metrics.json` · `logs/s_vqkembed_full.log` |
| 17 | 4.8350 | +0.0366 | output-embeddings | O-embed (`#33`) at natural end, 4,883 steps. O-embed alone is the worst of the embed family. **But O-embed is highly additive with V** — see row 1 (V+O = 4.7188). O is a bad position in isolation, a great position when paired with V. | 2026-06-02 | `runs/s_oembed_full/metrics.json` · `logs/s_oembed_full.log` |

## Fresh axes (2026-06-03, this session)

The embed/gain/FFN-activation family is closed. New axes tested in this session:
**attention pattern (SWA), positional encoding (NoPE), layer tying (ALBERT-style),
GQA ratio, MLP activation (GELU)**. All single-seed except where noted — multi-seed
confirmation pending. Closed axes go to a separate section below.

| # | Val loss | Δ vs ctrl | Run | Summary | Date | Evidence |
|---|---|---|---|---|---|---|
| 18 | 4.6700 | -0.1284 | vq-gain+swa (s42) | V+q_gain + sliding-window attention (window=512) (`#51`) at natural end, 4,883 steps, seed 42. First run to demonstrate SWA additivity. Flag-only, no extra params — SDPA's `is_causal=True` replaced by an explicit causal-local boolean mask (density 0.2188 at window=512, seq=2048). | 2026-06-03 | `runs/s_vqgain_swa_full/metrics.json` · `logs/s_vqgain_swa.log` |
| 18b | 4.6652 | -0.1332 | vq-gain+swa (s43) | V+q_gain + SWA, seed 43. **Multi-seed mean = 4.6676, std = 0.0034** (s42/s43) — exceptionally reproducible, tighter than V+q_gain. | 2026-06-03 | `runs/s_vqgain_swa_s43/metrics.json` |
| 18c | 4.6608 | -0.1376 | vq-gain+swa+gelu | V+q+SWA + GELU FFN (`#62`) at natural end, seed 42. Single-seed, -0.009 below V+q+SWA s42 (4.6700), -0.015 below V+q_gain 3-seed mean 4.6815. GELU is **additive** with V+q+SWA on top of squared_relu. The FFN activation IS a real lever, but only after SWA unlocks the right operating point. Cost: same param count as squared_relu. | 2026-06-03 | `runs/s_vqgain_swa_gelu_full/metrics.json` |
| 18d | **4.6364** | **-0.1620** | vq-gain+swa+highrope | V+q+SWA + RoPE base=500000 (Llama-style) (`#64`) at natural end, seed 42. **CURRENT BEST single-seed result** — -0.024 below V+q+SWA+GELU (4.6608), -0.031 below V+q+SWA 2-seed mean (4.6676), -0.045 below V+q_gain 3-seed mean (4.6815). The default RoPE base=10000 was leaving positional headroom on the table at our seq_len=2048. **The positional decay axis was a real lever** — base=500000 keeps positional information sharper over longer distances and is additive with V+q+SWA. Single-seed. | 2026-06-03 | `runs/s_vqgain_swa_highrope_full/metrics.json` |
| 18e | 4.6527 | -0.1457 | vq-gain+swa+highrope+gelu | V+q+SWA+HighRoPE + GELU FFN (`#65`) at natural end, seed 42. 4.6527 vs 4.6364 = **+0.016** (worse). **GELU is CLOSED on HighRoPE** — flips from additive on base=10000 (#62: -0.009 below V+q+SWA s42) to anti-additive on base=500000. RoPE base changes which lever is on top. | 2026-06-03 | `runs/s_vqgain_swa_highrope_gelu_full/metrics.json` |
| 18f | 4.7133 | -0.0851 | vq-gain+swa+highrope+tied2 | V+q+SWA+HighRoPE + layer tying (group=2) (`#66`) at natural end, seed 42. 4.7133 vs 4.6364 = **+0.077** (worse). Layer tying is **CLOSED on the new best baseline** — anti-additive again, just like on V+q. | 2026-06-03 | `runs/s_vqgain_swa_highrope_tied2_full/metrics.json` |
| 18g | 4.6384 | -0.1600 | vq-gain+swa+highrope+mha | V+q+SWA+HighRoPE + full MHA (n_kv_heads=6) (`#67`) at natural end, seed 42. 4.6384 vs 4.6364 = **+0.002** (essentially tied). **MHA is a wash on the new best baseline** — confirms GQA ratio is not a lever at this scale. | 2026-06-03 | `runs/s_vqgain_swa_highrope_mha_full/metrics.json` |
| 18h | 4.6500 | -0.1484 | vq-gain+swa+highrope+tiedqk | V+q+SWA+HighRoPE + Tied QK (PaLM-style) (`#72`) at natural end, seed 42. 4.6500 vs 4.6364 = **+0.014** (worse). Tied QK is **CLOSED on the new best baseline** — PaLM's QK tying doesn't help when SWA+HighRoPE are already on. | 2026-06-03 | `runs/s_vqgain_swa_highrope_tiedqk_full/metrics.json` |
| 18i | 4.6672 | -0.1312 | vq-gain+highrope+swa256 | V+q+HighRoPE + SWA(window=256) (`#68`) at natural end, seed 42. 4.6672 vs 4.6364 = **+0.031** (worse). Smaller window hurts — window=512 is the sweet spot. | 2026-06-03 | `runs/s_vqgain_highrope_swa256_full/metrics.json` |
| 18j | 4.6517 | -0.1467 | vq-gain+highrope+swa1024 | V+q+HighRoPE + SWA(window=1024) (`#69`) at natural end, seed 42. 4.6517 vs 4.6364 = **+0.015** (worse). Larger window also hurts — window=512 is the sweet spot. | 2026-06-03 | `runs/s_vqgain_highrope_swa1024_full/metrics.json` |
| 18k | 4.6841 | -0.1143 | vq-gain+highrope+noswa | V+q+HighRoPE + NO SWA (`#70`) at natural end, seed 42. 4.6841 vs 4.6364 = **+0.048** (worse). **SWA is still load-bearing on the new best baseline** — removing it loses 0.048 even with HighRoPE on. | 2026-06-03 | `runs/s_vqgain_highrope_noswa_full/metrics.json` |
| 18l | 4.6777 | -0.1207 | vq-gain+swa+highrope+softcap | V+q+SWA+HighRoPE + logit softcap=15 (`#71`) at natural end, seed 42. 4.6777 vs 4.6364 = **+0.041** (worse). **Logit softcap is CLOSED** — Gemma's cap doesn't help at this scale. | 2026-06-03 | `runs/s_vqgain_swa_highrope_softcap_full/metrics.json` |
| 18m | 4.7269 | -0.0715 | vq-gain+swa+highrope+mla | V+q+SWA+HighRoPE + MLA (DeepSeek-V2-style) (`#73`) at natural end, seed 42. 4.7269 vs 4.6364 = **+0.091** (worse). **MLA is CLOSED on the new best baseline** — the latent bottleneck loses 0.091. | 2026-06-03 | `runs/s_vqgain_swa_highrope_mla_full/metrics.json` |
| 18n | 5.2494 | +0.4510 | vq-gain+swa+highrope+dilated | V+q+SWA+HighRoPE + dilated attention (d=2) (`#74`) at natural end, seed 42. 5.2494 vs 4.6364 = **+0.613** (much worse). **Dilated attention catastrophically breaks training** — every-other-position pattern doesn't have enough density for attention to converge. | 2026-06-03 | `runs/s_vqgain_swa_highrope_dilated_full/metrics.json` |

### Closed-this-session summary

12 new arch axes tested on the new best baseline (V+q+SWA+HighRoPE 4.6364). All closed:

| # | axis | val | Δ vs best | verdict |
|---|---|---|---|---|
| 65 | GELU | 4.6527 | +0.016 | GELU flips from additive to anti-additive on HighRoPE |
| 66 | layer tying | 4.7133 | +0.077 | tying anti-additive again |
| 67 | MHA | 4.6384 | +0.002 | GQA ratio wash |
| 68 | SWA window=256 | 4.6672 | +0.031 | smaller window worse |
| 69 | SWA window=1024 | 4.6517 | +0.015 | larger window worse — 512 is sweet spot |
| 70 | no SWA | 4.6841 | +0.048 | SWA still load-bearing on best baseline |
| 71 | logit softcap=15 | 4.6777 | +0.041 | Gemma cap doesn't help |
| 72 | Tied QK (PaLM) | 4.6500 | +0.014 | QK tying doesn't help |
| 73 | MLA (DeepSeek-V2) | 4.7269 | +0.091 | latent bottleneck worse |
| 74 | dilated attention (d=2) | 5.2494 | +0.613 | strided pattern breaks training |

Not run (config not available on remote when q45 fired):
- #75 post-norm — left for next session
- #76 GQA=1 on best baseline — left for next session
- #77 no embedding scale — left for next session
- #78 SWA window=2048 (full) — left for next session

**V+q+SWA+HighRoPE 4.6364 holds as the current best single-seed screen20m record.** 12 axes closed on top of it. SWA + HighRoPE is the load-bearing combination.
| 19 | 4.7419 | -0.0565 | vq-gain+tied2 | V+q_gain + layer tying (ALBERT-style, group_size=2) (`#56`) at natural end, 4,883 steps, seed 42. 12 unique blocks, 24 layer passes. -0.057 vs control, but +0.062 vs V+q_gain. **Layer tying is CLOSED on V+q** — still beats control (so depth uniqueness is not the *only* thing) but adding tying on top of V+q costs ~0.06. | 2026-06-03 | `runs/s_vqgain_tied2_full/metrics.json` · `logs/s_vqgain_tied2.log` |
| 20 | 4.7552 ± 0.027 | -0.043 | swa-only | Sliding-window attention only (`#52`), no embeds, no gains, window=512, seeds 42+43. **Real standalone lever** — ~1.6σ below control, beats V alone (4.7728) by 0.018. 2-seed mean 4.7552, std 0.0273. SWA gives back ~half the V+q_gain win on its own. | 2026-06-03 | `runs/s_swa_only_full/metrics.json` · `runs/s_swa_only_s43/metrics.json` |
| 21 | 4.7981 | -0.0003 | mha | Full multi-head attention (`#58`, n_kv_heads=6) at natural end, 4,883 steps, seed 42. **Effectively tied with control** (4.7984, -0.0003). GQA=2 is a wash at this scale — removing KV sharing (full MHA) gives nothing. **GQA is not a lever at this scale.** | 2026-06-03 | `runs/s_mha_full/metrics.json` · `logs/s_mha.log` |

### Closed this session (do not retry)

| Run | val | Verdict | Why closed |
|---|---|---|---|
| V+q+NoPE (`#54`) | 5.2406 | **CLOSED** | RoPE is load-bearing. -0.561 vs V+q_gain, +0.442 vs control. NoPE hurts V+q catastrophically — RoPE is a structural requirement, not a free lever. |
| V+q+LayerTied2 (`#56`) | 4.7419 | **CLOSED on V+q** | +0.062 vs V+q_gain. Layer tying acts as a regularizer but conflicts with V+q — V+q already provides depth-anchored signal that tying disrupts. Still beats control, so depth uniqueness isn't the only thing, but the lever is anti-additive with V+q. |

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
