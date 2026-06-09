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
- 001-cautious-muon — null: trt=6.4125 vs ctrls 6.3875/6.4050 (loses to both; previous orphan-sweep pass inside variance) at tiny1m3m — 2026-06-09
- 004-retnet-retention — null: trt=6.4162 vs ctrls 6.3875/6.4050 (loses to both; v1 ships kernel+probe, v2 wiring is the real A/B) at tiny1m3m — 2026-06-09
- 005-decoupled-qkv-muon — null: trt=6.3909 vs ctrls 6.3875/6.4050 (sits between, inside variance) at tiny1m3m — 2026-06-09
- 009-fire-pe — WIN: trt=6.3234 vs ctrls 6.3875/6.4050 (Δ -0.064/-0.082 ≫ gap 0.0175; far exceeds plan bar) at tiny1m3m — 2026-06-09
- 007-sigmoid-loss — reject: arXiv ID unverifiable after 3 rounds (r1 cited wrong paper, r2 guess-by-authority unconfirmable) — 2026-06-09
- 002-cautious-adamw — null: A_emb=+0.0003 B_gain=-0.0066 at tiny1m3m (both inside ~0.04 run-to-run variance) — 2026-06-09
- 008-gated-deltanet — taste-reject: off-niche on tier (proposed screen20m+; pipeline runs tiny1m3m only) — 2026-06-09
- 003-soap — null: trt=6.4191 vs ctrls 6.4078/6.4072 (worse than both; vocab params on AdamW fallback so SOAP mostly bypassed at tiny1m3m) at tiny1m3m — 2026-06-09
