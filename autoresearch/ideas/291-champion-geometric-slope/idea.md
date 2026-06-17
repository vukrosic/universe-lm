---
id: 291-champion-geometric-slope
status: needs-confirm
round: 1
updated: 2026-06-16T13:27:09Z
transfer-risk: low
plain: Fold the alibi-deep-dive thread's winning slope-INIT lever back onto the champion (deepnet-α + poly-alibi). GEOMETRIC-3× (deep-dive geo3lrn 6.2301) per-head alibi slope init, learnable, vs the champion's learn-from-0 slopes. Env-driven (committed in models/layers.py), 0 new params.
---

# 291-champion-geometric-slope — champion + geometric-3× alibi slope init (cross-thread transfer)

## Why this, why now
The main box is otherwise starved (288/289 DeepNet-β both NULL; tier saturated).
The **alibi-deep-dive** thread found a real, untransferred win: learning alibi
slopes from 0 underfits the locality prior at tiny1m3m/92 steps, and warm-starting
at the classic geometric magnitudes scaled ~3× (geometric per-head distribution) beats
learn-from-0 by **Δ-0.040** on plain alibi (GEOMETRIC-3× (deep-dive geo3lrn 6.2301)). The champion still uses
learn-from-0 slopes — this is the brief's explicit stretch goal: fold the sharper
positional kernel back into the champion.

## Mechanism
Committed env knob in models/layers.py sets the alibi-slope init (default off =
byte-identical to champion). This stub: `ALIBI_SLOPE_INIT=geometric`,
`ALIBI_SLOPE_DIST=geometric`, `ALIBI_SLOPE_SCALE=3.0`, learnable, on
Tiny1M3MAlibiConfig + use_deepnet_alpha + use_poly_alibi. 0 new params (949,200).

## Hypothesis
Right-sign Δ if the uniform locality prior adds signal the champion's learn-from-0
slopes never reach in 92 steps; NULL if poly-alibi's per-head curvature already
captures that locality (redundant). The deep-dive showed uniform > geometric on
plain alibi, so 290 (uniform) is the stronger bet, 291 (geometric) the control.

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then paired
3-seed confirm before promotion). Single seed (42).

## TRIAGE 2026-06-16 — subsumed by 296 (do NOT spend a confirm here)
This screen-win sits **at the confirm band** (Δ≈−0.020 vs band 0.018, ~1.3σ single
seed) and is a strict **subset** of the 296 combo (slope + curvature warm-start).
296 (Δ−0.026) dominates it. Decision: do **not** run an individual paired confirm —
- if 296 CONFIRMS, the combo supersedes this solo lever (promoting a weaker subset is pointless);
- if 296 FAILS confirm, this whole axis is seed-noise and this solo would fail too.
Stays `needs-confirm` as a record of the screen-win, but is deferred indefinitely.
