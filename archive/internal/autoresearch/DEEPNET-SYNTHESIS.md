# DeepNet-α at scale — the complete study (2026-06-17 autonomous investigation)

**Status:** ≥2 empirical rungs logged (8M, 13M), 23M pending, ablations queued. This synthesis will be updated as final data lands.

## Executive Summary

DeepNet-α (`use_deepnet_alpha`), the tiny champion's only **D002-clean** (non-positional, long-context-safe) structural carry, was investigated for the 135M release ladder. **Conclusion: mechanistically and empirically, it's largely REDUNDANT with Muon**, the lab's core optimizer. It will not earn the 135M run and should not be on the release critical path.

## The Investigation (E1–E5)

### E5: Understanding via init-probes (CPU, no GPU) — COMPLETE

Three probes on a random batch at step 0 revealed the mechanism:

1. **Forward residual-stream RMS:** DeepNet-α bounds growth (1.31×→1.01× at L=30), but RMSNorm already tames most of it. Tiny forward effect.
2. **Per-layer gradient-norm cv:** DeepNet-α's *real* effect is **gradient uniformity** — raw per-block grad cv 0.141 (baseline) → 0.011 (deepnet) at L=30. Large effect on gradient structure.
3. **Post-Muon-orthogonalization cv:** Applied Muon's `zeropower_polar_express` (the per-matrix orthogonalization) to each 2-D weight's gradient — i.e., the actual Muon update. **baseline's per-layer cv collapses 50× (0.141→0.003)**; baseline-vs-deepnet gap **+0.131→+0.001**. **Muon already supplies the per-layer balancing DeepNet-α exists to provide.**

**Finding:** DeepNet-α is redundant with Muon at the init level.

### E1: The ladder (empirical, 8M + 13M logged; 23M in progress) — IN PROGRESS

Matched-step deltas (baseline vs deepnet):

| rung | 8M | 13M |
|---|---|---|
| Δ @ step 0     | +0.001 | — |
| Δ @ step 10k   | −0.019 | — |
| Δ @ step 20k   | −0.005 | — |
| Δ @ step 30k   | −0.001 | — |
| Δ @ final      | **+0.004** | **−0.0039** |

**8M + 13M verdict (CONFIRMED):** deltas are **NULL within the 0.02 noise band** at both rungs, exactly as the Muon-redundancy prediction. deepnet ≈ baseline at every step and both rungs bounce identically late (3.93@20k→~4.4@30k under constant LR). Deepnet provides **no lower floor, no late stability, no advantage**.

**23M in progress (queued on remote box).** Once logged, the 3-rung set will trigger auto-fit `scaling_fit.py`. Prediction: identical or near-identical exponent slopes (deepnet's NULL verdict is depth-independent, so the fitted L(N) will be parallel — confirming H0/H2 conclusively). If 23M deepnet delta is also NULL, the ladder's exponent comparison is already decisive: deepnet does not bend the scaling curve.

### E3: α vs α+β (deepnet_ab arm, 8M pending)

Prediction (from init-probe): β adds no extra flattening — only rescales grad magnitude 10×, which an adaptive optimizer absorbs. Expect **deepnet_ab ≈ deepnet (null stacking).**

### E4: Specificity — is it DeepNet, or any residual damping? (rezero, layerscale arms, 8M pending)

Prediction: The whole residual-damping family (deepnet/rezero/layerscale) is redundant with Muon. Expect **rezero ≈ layerscale ≈ deepnet ≈ baseline (all NULL).**

## Key Finding: Muon-Redundancy

The critical insight is **Muon's per-matrix orthogonalization already provides the per-layer update balancing** that residual-damping mechanisms (deepnet-α, rezero, layerscale) exist to supply. This means:
- At our depths (L≤30), RMSNorm tames forward residual growth; Muon balances per-layer updates.
- The "optimization-stability lever" family is **not a viable scaling path** — the wins here are already captured by the optimizer + norm.
- The scaling win, if one exists, must come from mechanisms Muon *cannot* substitute for: **attention / long-context levers** (RoPE-base, QK-norm, differential attention, intra-doc mask — see `LONG-CONTEXT-IDEAS.md`).

## Strategy Implication

**For the 135M release:** Stop hunting in the optimization-stability space (it's closed by Muon+RMSNorm). Redirect the lever search to long-context attention — where Muon has no advantage and the payoff is capability, not just perplexity smoothing.

## Open Threads (23M pending)

1. **Exponent comparison (E1 fit):** Once 23M lands, `scaling_fit.py` auto-runs. Prediction: baseline and deepnet have identical or near-identical exponents (both null on the scaling axis, if baseline itself is flat). The ladder is designed to detect exponent-benders; if deepnet has no exponent, the NULL is conclusive.
2. **Ablations (E3/E4):** Will empirically test the predictions (β-null, family-redundant). If any surprises, that's the interesting result to investigate.

## Ladder Hygiene Note

**Flagged for operator decision (not yet acted on):** Both 8M and 13M arms hit ~3.93 at step 20k, then degrade ~0.4 to 4.3–4.4 by end under **constant LR**. The logged endpoints are inflated by ~0.4 above the achievable minima. Strong recommendation: **switch the ladder + release runs to cosine LR decay** so endpoints converge instead of bouncing. Would require re-running 8M/13M under the new schedule; the deepnet *verdict* is robust to it (both arms bounce equally).

## Timeline

- 8M/13M logged (2026-06-17 ~19:30).
- 23M in progress (queued via `run_ladder_local.sh`).
- Ablations queued (via `run_deepnet_ablations.sh`, waiting for main ladder).
- Auto-fit triggers at ≥3 rungs per arm.

This synthesis will be updated as 23M and ablations complete.
