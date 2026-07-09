# Evidence — 011 cautious-lion

## Verdict: WIN (within session)
- tier: tiny1m3m, seed 42, box: vast-34386 (220.82.52.202:34386, RTX 3060)
- control val: **6.4253**   treatment val: **6.3941**   Δ: **−0.0312** (vs ctrl)
- ctrl2: **6.4262** (two-ctrl bracket; ctrl-to-ctrl gap **0.0009**)
- Treatment beats **both** ctrls by 0.0312 and 0.0321 — margin far exceeds
  the ctrl-pair gap (0.0009). Plan PASS bar was ≤−0.015 — comfortably cleared
  with 2× headroom.
- ⚠️ box check: same +0.19 baseline drift as the 006/010 batches (session ctrl
  ~6.42 vs prior-day ~6.39) — within-session A/B valid, cross-day not
  (treatment sits AT the cross-day baseline, so the win is in-session only;
  see closed.md / 006 evidence). Cautious-Lion recovers back to prior-day
  ctrl level.
- raw: remote-results/2026-06-09-vast-tiny1m3m/logs/011-cautious-lion.log
  (will land in batch directory after copy)
- date: 2026-06-09
