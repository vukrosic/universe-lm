# Hypothesis-driven follow-ups

Screening found 2 wins (value-channel gate −0.0147, residual norm gate −0.0166) and
2 informative failures (attn-output channel gate +0.0247, residual token gate diverged).
Phase 2 stops screening and isolates *why* — each run below tests one hypothesis with
one minimal control. Tier: Tiny1M3M, seed 42, baseline 6.4216, noise ±0.005.

## H1 — VCG wins through per-CHANNEL selectivity, not per-head scale freedom
- **Test:** per-head scalar gate on V (one scalar per head, zero-init) — the coarse version of VCG.
- **Predicts:** scalar-V ≈ null, channel-V wins ⇒ selectivity is channel-level.
  If scalar-V also wins ⇒ the win is just "let heads rescale V", weaker claim.
- **Run:** `vsg` (needs ~10-line code change in sweep_value clone).

## H2 — RNG wins through input-DEPENDENCE, not the extra residual scale parameter
- **Test:** LayerScale (`use_layerscale=True`, exists in base repo) — input-independent
  per-channel residual scaling, zero-init, same step-0 identity.
- **Predicts:** LayerScale ≈ null/worse while RNG wins ⇒ the RMS(x) conditioning is the
  active ingredient. If LayerScale also wins ⇒ RNG is just "scale the writes", weaker claim.
- **Run:** `ls` (zero code, launch immediately).

## H3 — output-side gating hurts at any granularity (input vs output asymmetry)
- **Test:** per-head scalar output gate (`use_attn_output_gate=True`, exists in base repo) —
  the coarse version of the failed AOCG.
- **Predicts:** hurts or null ⇒ asymmetry is about *position* (pre vs post weighted sum),
  not granularity. If it wins ⇒ AOCG's failure was granularity/optimization, not position.
- **Run:** `aog` (zero code, launch immediately).

## H4 — the residual token gate diverged because its signal is UNBOUNDED
- **Test:** bounded version: write *= (1 + g·tanh(⟨x̂, u⟩)), u unit-norm direction, g zero-init scalar.
- **Predicts:** trains stably ⇒ divergence was a stability artifact, the *idea* (token-dependent
  write scale) remains live and is exactly what RNG implements with RMS instead of a projection.
- **Run:** `btg` (small code change in sweep_resid clone).

## H5 — VCG and RNG are INDEPENDENT mechanisms (different causal paths)
- **Test:** both flags on in one model.
- **Predicts:** combined ≈ −0.03 (additive) ⇒ independent. Sub-additive ⇒ shared cause,
  paper must merge the two stories.
- **Run:** `combo` (port RNG diff into sweep_value clone, then both flags on).

## Status — all complete (2026-06-11 evening)
| Run | Hypothesis | Final | Δ | Outcome |
|---|---|---|---|---|
| ls | H2 | 6.4241 | +0.0025 | null → input-dependence was RNG's active ingredient |
| aog | H3 | 6.4269 | +0.0053 | worse → output-side hurts at every granularity |
| vsg | H1 | 6.4231 | +0.0015 | null → channel selectivity is VCG's active ingredient |
| btg | H4 | 6.4250 | +0.0034 | stable but null → boundedness fixed the crash, not the idea |
| combo | H5 | 6.4066 | −0.0150 | non-additive: equals VCG alone → RNG fragile |

## Scale-up (Screen10M20M)
| Run | Final | Δ vs ctrl 4.8020 | Outcome |
|---|---|---|---|
| VCG | 4.6945 | −0.1075 | **confirmed, effect grows with scale** |
| RNG | 4.8739 | +0.0719 | **inverted — tiny-tier artifact** |

## Next
- Full10M200M (~10M params, 200M tokens) record attempt: ctrl + VCG running overnight.
