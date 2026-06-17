---
id: 299-combo-curv5
status: done
round: 2
updated: 2026-06-16T16:00:29Z
transfer-risk: low
plain: Record combo (296) with curvature pushed two steps up to ×5 — the curvature-ceiling probe. With 296 (×3) and 298 (×4) it traces a clean 1-D curvature gradient at fixed slope ×3 to locate the optimum / knee. Stacks on the 296 combo champion.
---

# 299-combo-curv5 — find the curvature ceiling

## Why this, why now
The curvature-ceiling probe. With 296 (curv ×3, the record) and 298 (curv ×4), this
(curv ×5) completes a 1-D curvature gradient at fixed slope ×3: **×3 → ×4 → ×5**.
Solo curvature climbed ×1→×3; somewhere the quadratic init dominates the attention
logits and over-suppresses distant tokens at step 0 — this finds that knee.

## Mechanism
Identical to 296 except `POLY_ALIBI_C_SCALE` 3.0 → 5.0. Full env: slope
geometric/uniform/×3/learnable, curvature geometric/×5. Tiny1M3MAlibiConfig +
use_deepnet_alpha + use_poly_alibi. Knob in models/layers.py, 0 new params.

## Hypothesis
299 < 298 < 296 if curvature is still gradient-positive at ×5 (new record); 299
worse than 298 if ×4 was the peak (locates optimum either way). Either outcome is
informative — this is the arm that turns a single win into a calibrated curve.

## A/B
Stacks on the **296 combo champion** (activate only after 296 promotes). Judged vs
the combo's confirmed 3-seed mean, SCREEN band 0.02, then paired confirm. Seed 42.

## Status note
Held `draft` (no run.json) until 296 confirms and is promoted. Diagnostic arm —
expected to bracket the optimum even if it does not itself win.
