---
id: 289-deepnet-beta-init-matched
status: done
round: 1
updated: 2026-06-16T13:12:56Z
transfer-risk: low
plain: Same lever as 288 (DeepNet β init-downscaling of the value/output/FFN projections) but with β = (2·n_layers)^(-1/2) ≈ 0.204 — exactly the champion's forward α, so forward branch and init are conditioned by the SAME residual-depth factor. 288 uses the milder canonical 0.319; together they bracket how hard the init should be down-scaled. Zero new params.
---

# 289 — DeepNet β init-downscaling MATCHED to the forward α

## Why this, why now
Companion point to [288](../288-deepnet-beta-init/idea.md) on the same init-only
conditioning axis — the one family that still binds at 0.94M/92 steps. 288 asks
*whether* the β half of DeepNet helps; 289 fixes the strength question that 288's
canonical constant leaves open.

## The one change vs 288
β = **(2·n_layers)^(-1/2) ≈ 0.204** instead of the canonical (8L)^(-1/4) ≈ 0.319.
This is exactly the champion's forward `use_deepnet_alpha` branch-scale, so the
**forward residual scale and the init down-scale use the identical factor** — the
symmetric reading of Wang 2022's α·β coupling (condition the init by the same
amount you scale the forward branch). Mechanism, code path, param count, and A/B
are otherwise identical to 288.

## Why bracket the strength
β is a down-scale in (0,1); the right value at this tier is unknown. 288's 0.319
is the textbook decoder gain; 289's 0.204 matches the forward branch the champion
already runs. Running both turns "does β help?" into "how much init down-scaling
matches the 0.204 forward branch" — a clean two-point sweep rather than a single
guess. Over-suppression risk is higher here (init AND forward both at 0.204), which
is exactly what the pair is designed to measure.

## A/B
vs champion val **6.2209**, SCREEN band **0.02** → SCREEN-WIN < 6.2009 (then a
paired 3-seed confirm before any promotion). Single seed (42).
