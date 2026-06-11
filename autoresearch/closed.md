# Closed levers — do not re-propose

The dedup list for the autoresearch loop. The **miner/scout read this before
filing** an idea; the **reviewer reads it** to catch closed re-proposals and
**appends to it on every `reject`**. One line per closed lever, terse and
greppable. (Full results live in `LEADERBOARD.md` — not edited by this system.)

## Who may append here

| Closer | When | Entry format |
|---|---|---|
| reviewer | verdict = `reject` (killed on paper) | `<NNN-slug or lever> — reject: <reason> — <date>` |
| evidence/run step | post-run null or failed pass-bar (killed by data) | `<NNN-slug> — null: <val, Δ vs ctrl> — <date>` |

The code-implementer never closes — if blocked it bounces the idea back to
`needs-review`. Keeping a single agent (reviewer) on the write path avoids races.

## Closed axes (seed — migrated from queue.md, 2026-06-08)

- V/Q/K/O embeds + combos, q_gain / k_gain (screen20m rows 0-17)
- SWA window sweep (256/384/512/768/1024/2048) — 512 winner
- RoPE base sweep — 500k winner
- NoPE, post-norm, layer tying, MHA vs GQA, MLA, Tied QK (on best baseline), dilated attention, logit softcap
- Norm zoo (pnorm, manhattan, center, squash, clip, channelscale)
- NSA / diff-attn / hybrid heads
- Multiscale heads / parallel block / attn sink (2026-06-04 batch)

## Closed by the loop (append below, newest first)

<!-- reviewer/evidence step appends one line per close here -->
- 068-unlikelihood — taste-reject: Welleck et al. (arXiv:1908.04319) explicitly claim "maintaining perplexity" — the lever's wins are on generation-quality metrics (rep-rate, distinct-n, self-BLEU) that our val-loss screen does not measure; null is guaranteed by the paper's own framing → info-free A/B; pitch is generation-quality with no "we expect val loss to drop by X because Y" sentence; mechanism is HP-grid-shaped (which tokens are negatives? coef? count?) not lever-shaped; 5th adjacent loss-shaper (066 label-smoothing / 067 confidence-penalty / 068 unlikelihood / 069 focal-loss / 070 mtp-head) — portfolio-crowded; transfer doesn't rescue (135M lever-effect is still generation, tiny null doesn't bound it) — 2026-06-11
- 070-mtp-head — taste-reject: paper's own Fig 3 (Gloeckle et al. arXiv:2404.19737) explicitly shows MTP HURTS at <3B and gains require 6.7B-13B; tiny1m3m (0.94M) is ~13000× smaller than the paper's validation scale, ~300× smaller than where MTP is worst vs NTP; transfer-risk: low is mis-tagged (large-scale-validated-with-small-scale-regression is the canonical *high* risk); param-budget injection (extra `d_model × vocab` head ≈ sizeable fraction of 0.94M) makes A/B unfair-by-param; info-value zero — null reconfirms paper, win would be unbelievable on one seed; closed twin 018-ademamix (same scale-axis failure mode) — 2026-06-11
- 056-branchnorm — taste-reject: tier-mismatch — BranchNorm is an "extremely deep" (≥100L) residual-stability lever (Liu et al. 2305.02790, paper title); tiny1m3m is 6L, depth-drift the β(t) schedule fixes does not accumulate; 017-sub-ln-sandwich null at 6L on 2026-06-09 already closed the DeepNet-family at our tier; guaranteed null teaches nothing new — 2026-06-11
- 040-adafactor — taste-reject: sublinear-memory v factorization is a scale-up tool, not a quality lever (at 0.94M params full v fits trivially, factorization is strictly-worse approx with no memory benefit); paper headline is *equivalence* not improvement → info-free either way; crowded by 9 better AdamW-variant bets (031 Adam-mini, 036 LAMB, 039 APOLLO already cover structured-2nd-moment thesis) — 2026-06-11
- 036-lamb — taste-reject: large-batch lever (paper headline BS 32,868 BERT); trust-ratio mechanism doesn't fire at tiny1m3m small-batch/~92-step regime; crowded optimizer wave (031-040); vibe-pitch — 2026-06-11
- 043-mla — taste-reject: closed lever (closed.md axes line) + two prior nulls — tiny1m arch row #3 (6.3253 behind tied QK/MHA/LayerNorm) and screen20m #73 (+0.091 vs best baseline, explicit "MLA is CLOSED") — 2026-06-11
- 013-cope — drift: trt=6.4659 vs ctrls 6.3969/6.3891 (Δ +0.069/+0.077 ≫ gap 0.0078); +0.143 vs 009 FIRE-alone (6.3234); stacked FIRE+CoPE is destructive at tiny1m3m — 2026-06-09
- 015-moonlight-muon-rms — WIN: trt=6.3906 vs ctrls 6.4044/6.4091 (Δ -0.0138/-0.0185 ≫ gap 0.0047; passes plan bar -0.01) at tiny1m3m — 2026-06-09
- 016-qk-norm — WIN: trt=6.3906 vs ctrls 6.4044/6.4091 (Δ -0.0138/-0.0185 ≫ gap 0.0047; passes plan bar -0.005) at tiny1m3m — 2026-06-09
- 017-sub-ln-sandwich — null: trt=6.4084 vs ctrls 6.4044/6.4091 (Δ +0.0040/-0.0007 inside gap 0.0047; expected null at 6L, lever fires at 100+ layers per DeepNet) at tiny1m3m — 2026-06-09
- 019-dyt — reject: mathematical duplicate of closed `squash` lever (models/layers.py:52-61 SquashNorm is `g·tanh(α·x)`, no reduction, operationally identical); closed.md:24 + findings.md:60/165 already mark `squash/DyT` as falsified (val 7.6278, diverged); α-shape tweak (per-dim→scalar, 1.0→0.5) is an init HP, not a mechanism change — 2026-06-09
- 018-ademamix — taste-reject: bet cannot fire at tiny1m3m (slow EMA half-life ~7k steps vs ~92 step run; 99% init-weighted; lever only fires at ≥100k steps per paper) — 2026-06-09
- 010-polyloss — null: trt=6.5938 vs ctrls 6.5991/6.6050 (Δ-0.0053 < ctrl-gap 0.0059, inside variance) at tiny1m3m — 2026-06-09
- 006-schedule-free-adamw — null: trt=6.8056 vs ctrls 6.5953/6.6091 (+0.21 worse, wrong sign) at tiny1m3m; ⚠️ session ctrl drifted +0.19 vs prior days (6.39) — suspected baseline pollution from wholesale trainer.py/llm_config.py sync — 2026-06-09
- 012-gated-deltanet — taste-reject: re-pitch of 008; same Yang et al. gated delta-rule linear attention, miner concedes mechanism "doesn't fire at this scale" — slot would confirm known tier-mismatch — 2026-06-09
- 014-sigmoid-loss — taste-reject: re-pitch of 007; same mechanism (sigmoid + z-loss) and same citation flaw (cites arXiv:2405.18719 = CoPE paper, not sigmoid loss) — 2026-06-09
- 001-cautious-muon — null: trt=6.4125 vs ctrls 6.3875/6.4050 (loses to both; previous orphan-sweep pass inside variance) at tiny1m3m — 2026-06-09
- 004-retnet-retention — null: trt=6.4162 vs ctrls 6.3875/6.4050 (loses to both; v1 ships kernel+probe, v2 wiring is the real A/B) at tiny1m3m — 2026-06-09
- 005-decoupled-qkv-muon — null: trt=6.3909 vs ctrls 6.3875/6.4050 (sits between, inside variance) at tiny1m3m — 2026-06-09
- 009-fire-pe — WIN: trt=6.3234 vs ctrls 6.3875/6.4050 (Δ -0.064/-0.082 ≫ gap 0.0175; far exceeds plan bar) at tiny1m3m — 2026-06-09
- 007-sigmoid-loss — reject: arXiv ID unverifiable after 3 rounds (r1 cited wrong paper, r2 guess-by-authority unconfirmable) — 2026-06-09
- 002-cautious-adamw — null: A_emb=+0.0003 B_gain=-0.0066 at tiny1m3m (both inside ~0.04 run-to-run variance) — 2026-06-09
- 008-gated-deltanet — taste-reject: off-niche on tier (proposed screen20m+; pipeline runs tiny1m3m only) — 2026-06-09
- 003-soap — null: trt=6.4191 vs ctrls 6.4078/6.4072 (worse than both; vocab params on AdamW fallback so SOAP mostly bypassed at tiny1m3m) at tiny1m3m — 2026-06-09
- 021-value-residual — WIN w/ caveat: trt=6.3075 vs shared fire-ctrl 6.3419 (Δ -0.034); shared fire-ctrl was buggy (subclass override dropped use_fire_pe=False), within-session delta is joint V-res+FIRE vs no-FIRE not V-res+FIRE vs FIRE — re-test for isolation at tiny1m3m — 2026-06-10
- 023-canon-conv — WIN w/ caveat: trt=6.2581 vs shared fire-ctrl 6.3419 (Δ -0.084); shared fire-ctrl was buggy (use_fire_pe=False); after stripping FIRE (~-0.07 from 009), Canon effect still ~-0.06 ≫ WIN bar — best of 020-025 cluster for Phase-2 at tiny1m3m — 2026-06-10
- 024-gated-attention — WIN w/ caveat: trt=6.3316 vs plan-024 ctrl 6.4269 (Δ -0.095); plan-024 ctrl was buggy (use_fire_pe=False dropped), within-session delta is joint gated+FIRE vs plain baseline; isolated gated effect requires proper fire-equipped ctrl at tiny1m3m — 2026-06-10
- 025-scalable-softmax — WIN w/ caveat: trt=6.3359 vs plan-025 ctrl 6.4269 (Δ -0.091); BOTH bugs: plan-025 ctrl missing use_fire_pe (subclass override dropped) AND trt config Tiny1M3MSSMaxConfig missing use_fire_pe=True (pre-baked anomaly); within-session delta is SSMax-alone vs plain baseline; spec'd SSMax+FIRE vs FIRE unmeasured — re-run with use_fire_pe baked in at tiny1m3m — 2026-06-10
