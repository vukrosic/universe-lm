---
id: 297-combo-geometric-slope-dist
status: draft
round: 2
updated: 2026-06-16T15:05:00Z
transfer-risk: low
plain: The 296 combo (record, 6.1947) used the slope distribution that was NULL solo (uniform, 290). Swap in the slope distribution that WON solo (geometric, 291) inside the combo, holding curvature fixed. One knob changed. Stacks on the 296 combo champion.
---

# 297-combo-geometric-slope-dist — the record combo with its slope half upgraded

## Why this, why now
296 broke the champion era open (slope-warm-start + curvature-warm-start =
**6.1947**, Δ−0.0262, biggest gain all session). But it shipped with a known
sub-optimal half: `ALIBI_SLOPE_DIST=uniform`, the per-head slope distribution that
was a **NULL solo** (290 uniform 6.2019), while the **geometric** distribution
**WON solo** (291 geometric 6.2009). The combo's curvature half masked the weaker
slope half. This asks whether the solo slope-dist ranking survives inside the combo.

## Mechanism
Identical to 296 except one env knob: `ALIBI_SLOPE_DIST` uniform → geometric.
Full env: `ALIBI_SLOPE_INIT=geometric`, `ALIBI_SLOPE_DIST=geometric`,
`ALIBI_SLOPE_SCALE=3.0`, learnable, `POLY_ALIBI_C_INIT=geometric`,
`POLY_ALIBI_C_SCALE=3.0`. On Tiny1M3MAlibiConfig + use_deepnet_alpha + use_poly_alibi.
Both knobs committed in models/layers.py (default off = byte-identical). 0 new params.

## Hypothesis
WIN (new record) if geometric's sharper per-head locality compounds with curvature
the same way it beat uniform solo — the slope and curvature kernels occupy
different frequency bands, so a better slope shape adds on top of curvature. NULL
if the combo's curvature already supplies whatever locality the geometric dist
added over uniform (the dist edge only mattered in the slope-only regime).

## A/B
Stacks on the **296 combo champion** (activate only after 296's paired 3-seed
confirm promotes it). Judged vs the combo's confirmed 3-seed mean, SCREEN band 0.02,
then paired 3-seed confirm before promotion. Single seed (42).

## Status note — DEMOTED to fold-in (not a record bet)
The uniform→geometric slope-dist difference was only **0.001 solo** (290 6.2019 vs
291 6.2009) — below the confirm band (0.018). So 297 will **not** be a confirmable
record on its own; the dist edge is sub-noise. Treat it as a **free fold-in to the
champion config** (use the geometric dist when re-pinning the combo champion), not
as a confirm-slot experiment. The real headroom is curvature scale (see 298/299).
Held `draft`, no run.json; do not flip to needs-run.
