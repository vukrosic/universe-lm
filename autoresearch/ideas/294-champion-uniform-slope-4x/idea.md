---
id: 294-champion-uniform-slope-4x
status: needs-confirm
round: 1
updated: 2026-06-16T14:54:59Z
transfer-risk: low
plain: Scale sweep on 290's right-sign lever. 290 (uniform-3× alibi slope warm-start) landed Δ-0.019 right-sign — val 6.2019, just 0.001 over the 0.02 screen bar. This pushes the same warm-start to ×4 to test whether a stronger linear locality prior clears the bar. Env-driven, 0 new params.
---

# 294 — champion + uniform-4× alibi slope init (scale sweep up)

## Why this, why now
[290](../290-champion-uniform-slope/idea.md) transferred the deep-dive's slope
warm-start to the champion and got Δ-0.019 right-sign — the strongest positive
signal in the champion era (268–293) — but missed the 0.02 screen bar by 0.001
(val 6.2019 vs 6.2009). On plain alibi the deep-dive found ×3 optimal; on the
champion poly-alibi already supplies some locality, so the slope's optimal warm-
start magnitude may sit higher. ×4 tests that directly.

## The one change vs 290
`ALIBI_SLOPE_SCALE=4.0` instead of 3.0 (uniform per-head, learnable). Everything
else identical to 290.

## Hypothesis
Clears the bar if the champion wants a stronger linear prior than plain alibi did
(poly-alibi having absorbed some of the ×3 effect); NULL/worse if ×3 was already
near the optimum and ×4 over-penalizes mid-range tokens. Brackets 290's ×3 from
above — a 2-point read (×3 known 6.2019, ×4 here) on the slope magnitude.

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then a
paired 3-seed confirm). Single seed (42).

## TRIAGE 2026-06-16 — subsumed by 296 (do NOT spend a confirm here)
This screen-win sits **at the confirm band** (Δ≈−0.020 vs band 0.018, ~1.3σ single
seed) and is a strict **subset** of the 296 combo (slope + curvature warm-start).
296 (Δ−0.026) dominates it. Decision: do **not** run an individual paired confirm —
- if 296 CONFIRMS, the combo supersedes this solo lever (promoting a weaker subset is pointless);
- if 296 FAILS confirm, this whole axis is seed-noise and this solo would fail too.
Stays `needs-confirm` as a record of the screen-win, but is deferred indefinitely.
