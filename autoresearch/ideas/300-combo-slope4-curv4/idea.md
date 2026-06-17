---
id: 300-combo-slope4-curv4
status: done
round: 2
updated: 2026-06-16T16:06:38Z
transfer-risk: low
plain: Record combo (296) with BOTH halves at ×4 — the interaction probe. Tests whether slope and curvature scales are independently additive past (3,3) or compete for the same locality budget. Stacks on the 296 combo champion.
---

# 300-combo-slope4-curv4 — the interaction corner

## Why this, why now
296 is the (slope ×3, curv ×3) record; 298/299 sweep curvature alone. This is the
joint corner (slope ×4, curv ×4) — both warm-starts one step past 296 at once. It
reads the **interaction** of the two scales, not just one axis.

## Mechanism
296 except `ALIBI_SLOPE_SCALE` 3.0 → 4.0 AND `POLY_ALIBI_C_SCALE` 3.0 → 4.0. Full
env: slope geometric/uniform/×4/learnable, curvature geometric/×4.
Tiny1M3MAlibiConfig + use_deepnet_alpha + use_poly_alibi. Knobs in layers.py, 0 new params.

## Hypothesis
If 298 (curv ×4) wins AND 300 beats 298, the two scales are independently additive
past (3,3) — optimum is at higher joint scale. If 300 is no better than the best
single-axis move, the halves compete for the same locality budget at higher scale
(saturation / diminishing returns). Maps the curvature of the response surface
around the record.

## A/B
Stacks on the **296 combo champion** (activate only after 296 promotes). Judged vs
the combo's confirmed 3-seed mean, SCREEN band 0.02, then paired confirm. Seed 42.

## Status note
Held `draft` (no run.json) until 296 confirms and is promoted. Interaction/control
arm for the 298/299 curvature sweep.
