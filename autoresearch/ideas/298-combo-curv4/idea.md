---
id: 298-combo-curv4
status: done
round: 2
updated: 2026-06-16T15:58:24Z
transfer-risk: low
plain: Record combo (296, slope×3 + curv×3) with curvature pushed one step up to ×4. Curvature is the unexhausted half of the locality kernel (solo ×1 NULL → ×3 WIN), slope is near-saturated. One knob changed. Stacks on the 296 combo champion.
---

# 298-combo-curv4 — push the steep half of the kernel

## Why this, why now
296 fixed the record combo at (slope ×3, curv ×3). The scale data says (3,3) is not
the optimum: **slope** barely moved ×3→×4 (6.2009→6.2000, near-saturated) while
**curvature** climbed steeply ×1→×3 (6.2094 NULL → 6.2006 WIN). The curvature half
still has gradient; this spends it. Hold slope ×3, push curvature ×3→×4.

## Mechanism
Identical to 296 except `POLY_ALIBI_C_SCALE` 3.0 → 4.0. Full env: slope
geometric/uniform/×3/learnable, curvature geometric/×4. Tiny1M3MAlibiConfig +
use_deepnet_alpha + use_poly_alibi. Knob in models/layers.py, 0 new params.

## Hypothesis
WIN (new record) if curvature sits below its optimum inside the combo — the steep
solo gradient implies headroom. NULL if (3,3) already sat at the curvature peak.

## A/B
Stacks on the **296 combo champion** (activate only after 296 promotes). Judged vs
the combo's confirmed 3-seed mean, SCREEN band 0.02, then paired confirm. Seed 42.
Pairs with 299 (curv ×5) to trace the 1-D curvature gradient 3→4→5 at fixed slope.

## Status note
Held `draft` (no run.json) until 296 confirms and is promoted. This is the **lead
arm** of the round — curvature is where the unexhausted gradient is.
