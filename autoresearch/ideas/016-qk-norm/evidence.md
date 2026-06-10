# 016 — QK-Norm (LayerNorm on Q,K head-dim) — evidence

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
| **016** (QK-Norm: LayerNorm on Q,K head-dim, γ=1, β=0) | **6.3906** | **−0.0138** | **−0.0185** |
| 017 (Sub-LN) | 6.4084 | +0.0040 | −0.0007 |
| ctrl2 | 6.4091 | — | — |

ctrl-to-ctrl gap: |6.4091 − 6.4044| = **0.0047**.

## Verdict — WIN

Treatment (6.3906) beats **both** ctrls (6.4044 and 6.4091) by more than the
ctrl-to-ctrl gap (0.0047). Δ of −0.0138 and −0.0185 ≫ 0.0047.

Pass bar from `plan.md`: `trt ≤ ctrl − 0.005`. Trt − ctrl1 = −0.0138, trt −
ctrl2 = −0.0185. Both pass.

## Note
Tied exactly with 015 at 6.3906 — different mechanism (per-tensor RMS
rescale on ortho'd update vs per-head LayerNorm on Q/K), same magnitude of
win. Suggests the small-model headroom is hit by either stability lever.

## Log files
- `~/arq/logs/ctrl.log` (75 KB)
- `~/arq/logs/016-qk-norm.log`
- `~/arq/logs/ctrl2.log`
