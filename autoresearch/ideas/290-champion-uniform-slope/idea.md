---
id: 290-champion-uniform-slope
status: done
round: 1
updated: 2026-06-16T13:21:03Z
transfer-risk: low
plain: Fold the alibi-deep-dive thread's winning slope-INIT lever back onto the champion (deepnet-α + poly-alibi). UNIFORM-3× (best deep-dive arm geo3uni 6.2181) per-head alibi slope init, learnable, vs the champion's learn-from-0 slopes. Env-driven (committed in models/layers.py), 0 new params.
---

# 290-champion-uniform-slope — champion + uniform-3× alibi slope init (cross-thread transfer)

## Why this, why now
The main box is otherwise starved (288/289 DeepNet-β both NULL; tier saturated).
The **alibi-deep-dive** thread found a real, untransferred win: learning alibi
slopes from 0 underfits the locality prior at tiny1m3m/92 steps, and warm-starting
at the classic geometric magnitudes scaled ~3× (uniform per-head distribution) beats
learn-from-0 by **Δ-0.040** on plain alibi (UNIFORM-3× (best deep-dive arm geo3uni 6.2181)). The champion still uses
learn-from-0 slopes — this is the brief's explicit stretch goal: fold the sharper
positional kernel back into the champion.

## Mechanism
Committed env knob in models/layers.py sets the alibi-slope init (default off =
byte-identical to champion). This stub: `ALIBI_SLOPE_INIT=geometric`,
`ALIBI_SLOPE_DIST=uniform`, `ALIBI_SLOPE_SCALE=3.0`, learnable, on
Tiny1M3MAlibiConfig + use_deepnet_alpha + use_poly_alibi. 0 new params (949,200).

## Hypothesis
Right-sign Δ if the uniform locality prior adds signal the champion's learn-from-0
slopes never reach in 92 steps; NULL if poly-alibi's per-head curvature already
captures that locality (redundant). The deep-dive showed uniform > geometric on
plain alibi, so 290 (uniform) is the stronger bet, 291 (geometric) the control.

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then paired
3-seed confirm before promotion). Single seed (42).
