# RMSNorm ablations — research plan

**For the implementing AI.** Self-contained. The base op is `y = g · x / rms(x)`
with `rms(x) = √(mean(x²) + eps)`. Every lever here is **one extra param or one free
trick** on that op. Promotes + expands the idea bank at
[../../research-plans/rmsnorm-tweaks/plan.md](../../research-plans/rmsnorm-tweaks/plan.md).

---

## The one point we're poking

```text
models/layers.py:  make_norm(dim, norm_type, use_layernorm)
RMSNorm:           y = g ⊙ x / √(mean(x²) + eps)
```

The residual-stream norms are `norm1`, `norm2` per block (and the final `self.norm`).
Every block uses two. A norm change at depth 24 compounds 48× — small per-op effects
can matter. Each lever changes only the norm op; baseline = plain RMSNorm.

## Already in the repo (do NOT reimplement — sweep, don't rebuild)

`make_norm` already accepts these `norm_type` strings — Batch 4 just *runs* them:
`rmsnorm`, `layernorm`, `pnorm<p>`, `clipnorm<k>`, `center`, `centeredl1`,
`channelscale`, `manhattan`, `manifold`, `median`, `peak`, `squash`. Also `use_layerscale`
(per-channel residual gate) exists — distinct from a norm gain, note the overlap.

---

## Implementation contract

- New norm ops → add a class + register in `_NORM_REGISTRY` (or extend RMSNorm with a
  flag), in [models/layers.py](../../../models/layers.py). Selected via `norm_type`
  (residual-stream norm) on the config.
- One `class Screen10M20M<Name>Config(Screen10M20MConfig)` per lever; set `norm_type`
  (and any scalar) there.
- Run: `python train_llm.py --config <name> --seed 42`.
- **Identity-init:** init the new param to the no-op value (g₀=0, b=0, λ=0, τ=1,
  c=0, p=2) → step-0 == baseline. Non-identity levers are flagged → own control.
- **Optimizer routing:** norm gains/biases/scalars are 1D → AdamW. Confirm gradient flow
  (a zero-init scalar under the wrong optimizer can stay zero).
- **Apply consistently:** whatever norm a config selects is used by `norm1`, `norm2`,
  and the final `self.norm`. Don't special-case one site or the A/B is muddy.

## Protocol (what counts)

- Control = clean `Screen10M20MConfig` (plain RMSNorm) → **4.7984** (`s_ctrl_full`).
- tiny → screen 3-seed (42/43/44). "Live" = mean beats control by ≥0.01, seeds don't
  straddle zero. Winners re-run on the full ladder.
- Norm is high-variance across seeds — **3-seed is mandatory** here, single-seed norm
  wins are noise. (The existing `tiny1m norm4/5/6` sweep already shows this.)

---

## Batch 1 — free / 1-param tweaks (the original bank, identity-init)

| # | Name | Op | Extra params | step-0==base | Conf |
|---|---|---|---|---|---|
| N1 | `ReparamGain` | `g = 1 + g₀`, learn g₀ (g₀=0) | 0 | yes | med |
| N2 | `RMSBias` | `y = g·x_norm + b` (b=0) | d_model | yes | med-high |
| N3 | `GlobalTemp` | `y = g·x / (rms·τ)`, scalar τ=1 | 1 | yes | med |
| N4 | `PartialNormMix` | `y = g·x_norm + λ·x`, scalar λ=0 | 1 | yes | med-high |
| N5 | `LearnableFloor` | `rms = √(mean(x²) + c²)`, c≈0 | 1 | ~yes | low-med |
| N6 | `ScaledGainInit` | init g=0.5 not 1.0 | 0 | no — own control | low-med |
| N7 | `SoftplusGain` | `g = softplus(w)`, w s.t. g≈1 | 0 | yes | low |

Cheapest high-signal (run first): N1, N4, N2.

## Batch 2 — NEW structural tweaks (still cheap)

| # | Name | Op | Extra params | step-0==base | Why |
|---|---|---|---|---|---|
| N8 | `PartialNormVector` | `y = g·x_norm + λ⊙x`, **λ a length-d_model vector** (λ=0) | d_model | yes | per-channel version of N4 — which channels want to stay un-normalized? |
| N9 | `GroupRMS` | split d_model into G groups, rms **per group** (GroupNorm-style) | 0 | ~yes | tests whether one global magnitude per token is too coarse |
| N10 | `StopGradRMS` | `y = g·x / detach(rms(x))` — no gradient through the denominator | 0 | yes | isolates whether the *gradient* through rms matters or just the forward scaling |
| N11 | `AsymGain` | separate gain for x>0 and x≤0 (`g₊`, `g₋`, both=1) | d_model | yes | breaks the sign symmetry of a single gain — cheap extra expressivity |
| N12 | `GainClamp` | bound the learned gain to `[1−a, 1+a]` | 0 | yes | stops runaway gains at depth; tests if unbounded gain is a problem |
| N13 | `DepthScaledGainInit` | init gain `= 1/√(layer+1)` (not learned offset) | 0 | no — own control | manual depth taper of branch magnitude via the norm |
| N14 | `LearnableEps` | learn the eps inside the sqrt (init = default eps) | 1 | yes | is the fixed numerical floor leaving signal on the table? |

## Batch 3 — NEW norm-replacement probes (bigger swings, gated)

| # | Name | Op | step-0==base | Why |
|---|---|---|---|---|
| N15 | `DynTanh` | replace norm with `y = g·tanh(α·x)` (DyT, no statistics) | no — own control | recent result: a normless elementwise op can match RMSNorm. big claim, cheap test |
| N16 | `DoubleNorm` | apply RMSNorm twice (`norm(norm(x))`) | ~yes | does iterating the normalization sharpen signal prop, or wash? |
| N17 | `CenterMix` | `y = g·(x − μ·x̄)/rms`, scalar μ=0 interpolating RMS↔LayerNorm | 1 | yes | a *continuous* RMS-to-LayerNorm knob — how much centering does the model want? |

## Batch 4 — existing norm zoo sweep (no new code, just run)

Run the repo's existing `norm_type` options at this tier, 3-seed, vs RMSNorm control:
`pnorm1.5`, `pnorm1.75`, `clipnorm3`, `channelscale`, `manhattan`, `center`,
`centeredl1`, `manifold`, `median`, `peak`, `squash`, `layernorm`. Pure ranking run —
extends the prior `tiny1m norm4/5/6` work to the screen tier. Promote any clear mover.

---

## Run guidance

Batches 1–2 are the cheap core (run all at tiny, promote movers). Batch 3 are
higher-variance probes — run only after the core shows whether the norm axis is alive.
Batch 4 is a free ranking sweep using code that already exists. Two A/Bs to record:
**N4 scalar vs N8 vector** partial-norm (scalar or per-channel?), and **N17 CenterMix**
landing point vs the existing `layernorm` (does partial centering beat full?).

## When a batch finishes

1. Numbers → [tutorial/results.md](tutorial/results.md) — 3-seed mean + std (mandatory here).
2. Status → [tutorial/experiments.md](tutorial/experiments.md).
3. Clear story → draft [tutorial/README.md](tutorial/README.md), house style of
   [../../tutorials/normalization/README.md](../../tutorials/normalization/README.md)
   (the existing norm tutorial — this folder extends it, link back).
4. Commit `metrics.json`, re-run `runs/make_evidence_index.py`.
