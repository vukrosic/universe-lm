# 013 — CoPE (Content-aware Positional Encoding) — evidence

**Date**: 2026-06-09
**Tier**: tiny1m3m (0.94M params, 3M tokens)
**Box**: vast-34386 (RTX 3060)
**Seed**: 42 (one seed only, per project rule)
**Queue**: ctrl → 013-cope (FIRE+CoPE stacked) → ctrl2

## Results

| Run | Final Val Loss | Δ vs ctrl1 | Δ vs ctrl2 |
|---|---|---|---|
| ctrl | 6.3969 | — | — |
| **013** (FIRE + CoPE, stacked) | **6.4659** | **+0.0690** | **+0.0768** |
| ctrl2 | 6.3891 | — | — |

ctrl-to-ctrl gap: |6.3891 − 6.3969| = **0.0078**.

## Verdict — DRIFT (clear regression)

Treatment (6.4659) is **+0.069 to +0.077 worse** than both ctrls — far
outside the 0.0078 ctrl-to-ctrl gap. Stacking CoPE on FIRE produces a
**large negative effect** (vs in-session plain baseline: +0.069; vs the
closed 009 FIRE-alone WIN at 6.3234: **+0.143** — CoPE added on top of
FIRE is *worse than no positional encoding at all*).

This kills the stacked lever for tiny1m3m. The CoPE bias + FIRE bias
likely interact destructively at this scale (both add per-position bias
to the attention scores; the combined bias is too large).

## Note (composition)
- 009 FIRE alone: 6.3234 (WIN, closed)
- 013 FIRE+CoPE: 6.4659 (DRIFT, this run)
- Difference: +0.143. **CoPE stacked on FIRE ruins the FIRE win.**

## Log files
- `~/arq/logs/ctrl.log`
- `~/arq/logs/013-cope.log`
- `~/arq/logs/ctrl2.log`
