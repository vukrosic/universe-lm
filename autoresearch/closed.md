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
