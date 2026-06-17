---
id: 293-champion-poly-curvature-init-gentle
status: done
round: 1
updated: 2026-06-16T14:54:55Z
transfer-risk: low
plain: Gentle bracket for 292 — same poly-alibi curvature warm-start but POLY_ALIBI_C_SCALE=1.0 (c_h ≈ 0.083 uniform per head) instead of ×3. Tests whether a mild convex far-token prior at step 0 helps without the over-penalization risk of ×3. Env-driven, 0 new params.
---

# 293 — champion + poly-alibi curvature warm-start (×1, gentle)

## Why this, why now
The magnitude companion to [292](../292-champion-poly-curvature-init/idea.md).
292 warm-starts the poly-alibi curvature c_h at ×3 (≈0.249); the d²/L term makes
that a ~510-logit penalty at the farthest token, which may over-suppress distant
attention. This runs the identical lever at ×1 (c_h ≈ 0.083) so the pair brackets
the curvature pre-load strength — the same two-point strategy 288/289 used for β.

## The one change vs 292
`POLY_ALIBI_C_SCALE=1.0` instead of 3.0. Everything else — config, code path,
param count (0 new), seed, A/B — identical to 292.

## Hypothesis
If 292 wins, ×1 reads where the optimum sits (closer to 0 ⇒ curvature wants a
light touch; ×3 better ⇒ wants more). If 292 over-penalizes (wrong-sign), ×1 is
the safer arm that still tests whether ANY curvature warm-start binds. If both
NULL, the curvature-init sub-axis closes alongside any null from 290's slope-init
and the locality-prior warm-start axis is exhausted on the champion.

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then a
paired 3-seed confirm before any promotion). Single seed (42).
