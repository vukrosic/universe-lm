# 017 — Sub-LN / Sandwich block — evidence

**Date**: 2026-06-09
**Tier**: tiny1m3m (0.94M params, 3M tokens)
**Box**: vast-34386 (RTX 3060)
**Seed**: 42 (one seed only, per project rule)
**Queue**: ctrl → 015 → 016 → 017 → ctrl2

## Results

| Run | Final Val Loss | Δ vs ctrl1 | Δ vs ctrl2 |
|---|---|---|---|
| ctrl | 6.4044 | — | — |
| 015 (Moonlight) | 6.3906 | −0.0138 | −0.0185 |
| 016 (QK-Norm) | 6.3906 | −0.0138 | −0.0185 |
| **017** (Sub-LN: `LN_post(Sublayer(LN_pre(x)))`) | **6.4084** | **+0.0040** | **−0.0007** |
| ctrl2 | 6.4091 | — | — |

ctrl-to-ctrl gap: |6.4091 − 6.4044| = **0.0047**.

## Verdict — NULL (slight drift vs ctrl1, inside variance band)

Treatment (6.4084) fails to beat **ctrl1** (Δ +0.0040) and only marginally
beats **ctrl2** (Δ −0.0007) — both deltas are well inside the 0.0047
ctrl-to-ctrl gap. By the two-ctrl rule, this is **NULL** (inconclusive, on
variance).

Per `plan.md` pass bar (Δ ≤ −0.005): fails (Δ +0.0040 / −0.0007 — neither
side beats by the bar). Per `plan.md` drift bar (Δ > +0.01): also fails.
Per the taste review framing ("treat as a depth-stability probe; null is
the more informative outcome"): clean null at 6 layers, as expected for
a lever DeepNet reports firing at 100+ layers.

## Note (composition)
015 and 016 both won by ~−0.015 in the same A/B session. 017 (sub-LN) did
not. Partitions the depth-stability axis: per-tensor RMS rescale on the
optimizer (015) and per-head logit bounding (016) help at 6 layers; per-
sublayer-output re-bounding (017) does not. Consistent with DeepNet's
report that sub-LN's win is at 100+ layers.

## Log files
- `~/arq/logs/ctrl.log` (75 KB)
- `~/arq/logs/017-sub-ln-sandwich.log`
- `~/arq/logs/ctrl2.log`
