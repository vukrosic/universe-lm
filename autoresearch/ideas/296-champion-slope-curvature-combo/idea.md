---
id: 296-champion-slope-curvature-combo
status: done
round: 1
updated: 2026-06-16T14:55:03Z
transfer-risk: low
plain: Stack BOTH locality-prior warm-starts — uniform-3× linear slope init (290, Δ-0.019 right-sign) + geometric-3× quadratic curvature init (292) — on the champion. Pre-load the full linear+quadratic distance kernel at step 0 so the optimizer starts from the locality prior instead of learning it from 0 in 92 steps. Env-driven, 0 new params.
---

# 296 — champion + slope warm-start + curvature warm-start (combo)

## Why this, why now
[290](../290-champion-uniform-slope/idea.md) warm-started the LINEAR slope and got
Δ-0.019 right-sign, just shy of the 0.02 bar — the linear locality prior helps but
poly-alibi's curvature partly covers it. [292](../292-champion-poly-curvature-init/idea.md)
warm-starts the QUADRATIC curvature. This stacks both: the champion's poly-alibi
subtracts `m_h·d + c_h·d²/L`, and BOTH coefficients start at 0 today. 296 pre-loads
both halves of the distance kernel at step 0, the complete locality prior, rather
than learning either from scratch in 92 steps.

## The change vs champion
`ALIBI_SLOPE_INIT=geometric, DIST=uniform, SCALE=3.0` (the 290 lever) AND
`POLY_ALIBI_C_INIT=geometric, SCALE=3.0` (the 292 lever), together. Both knobs
committed in models/layers.py; 0 new params; flag-off path byte-identical.

## Hypothesis
Clears the bar if the linear and quadratic priors are COMPLEMENTARY (slope sets
near/mid decay, curvature sets far-token decay — together the full kernel); NULL or
no-better-than-290 if they are REDUNDANT (both encode "attend local", competing for
the same gradient signal so stacking double-counts); worse if combined they over-
penalize. Reads directly against 290 (slope-only 6.2019) and 292 (curvature-only).

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then a
paired 3-seed confirm). Single seed (42).
