---
id: 292-champion-poly-curvature-init
status: needs-confirm
round: 1
updated: 2026-06-16T14:54:52Z
transfer-risk: low
plain: Warm-start poly-alibi's QUADRATIC curvature term c_h (the d²/L convex far-token penalty) instead of learning it from 0. The slope-init win (290) showed the LINEAR locality prior underfits in 92 steps; this is the curvature analog — init c_h>0 uniform per head (geometric magnitudes ×3 ≈ 0.249) so the convex distance decay is present at step 0. Env-driven (committed models/layers.py), 0 new params.
---

# 292 — champion + poly-alibi curvature warm-start (×3)

## Why this, why now
Only positional-kernel and step-0 conditioning levers bind at 0.94M/92 steps
(closed.md 268–289 all NULL). The champion's poly-alibi (230) is on the binding
axis and subtracts `m_h·d + c_h·d²/L`; both coefficients start at 0. The
alibi-deep-dive thread proved the LINEAR half (`m_h·d`) underfits when learned
from 0 — warm-starting it won Δ-0.040 (uniform-3×, transferred in 290). The
QUADRATIC half (`c_h·d²/L`) has the same learn-from-0 init and the same 92-step
budget, so it plausibly underfits the curvature prior identically. This is the
untested curvature analog of the one lever that has moved the needle this round.

## The one change vs champion
`POLY_ALIBI_C_INIT=geometric, POLY_ALIBI_C_SCALE=3.0` → init `poly_alibi_c` at the
classic geometric slope magnitudes, made UNIFORM per head (the deep-dive's winning
distribution), scaled 3× ⇒ c_h ≈ 0.249 for all 4 heads. c_h>0 ⇒ a convex
far-token penalty present at step 0 (vs the champion's c_h=0). poly_alibi_c stays
learnable. Mechanism, param count (0 new), and A/B otherwise identical to the
champion. Flag-off path bit-identical (default zeros).

## Distinct from 290
290 warm-starts the LINEAR `alibi_slope` (slope of the distance bias). 292 warm-
starts the QUADRATIC `poly_alibi_c` (curvature). Different coefficient, different
distance kernel shape (linear vs convex), different code site. 293 brackets the
magnitude at ×1 (gentle, c_h ≈ 0.083) — together a two-point read on how hard the
curvature should be pre-loaded, the same bracket strategy 288/289 used for β.

## Hypothesis
Right-sign Δ if the convex curvature prior, like the linear one, needs warm-starting
to bind in 92 steps; NULL if poly-alibi's c_h already converges fast enough from 0
(its d²/L gradient is high-leverage for far tokens) or if the linear term already
captures the available locality signal so curvature is redundant; wrong-sign if
×3 over-penalizes far tokens (c_h·2048 ≈ 510 logit penalty kills distant attention)
— which is exactly what the gentler 293 (×1) is there to catch.

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then a
paired 3-seed confirm before any promotion). Single seed (42).

## TRIAGE 2026-06-16 — subsumed by 296 (do NOT spend a confirm here)
This screen-win sits **at the confirm band** (Δ≈−0.020 vs band 0.018, ~1.3σ single
seed) and is a strict **subset** of the 296 combo (slope + curvature warm-start).
296 (Δ−0.026) dominates it. Decision: do **not** run an individual paired confirm —
- if 296 CONFIRMS, the combo supersedes this solo lever (promoting a weaker subset is pointless);
- if 296 FAILS confirm, this whole axis is seed-noise and this solo would fail too.
Stays `needs-confirm` as a record of the screen-win, but is deferred indefinitely.
