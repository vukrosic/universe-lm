## r1 — 2026-06-15 — verdict: revise
- **Pass/fail bar needs formalization with cache reference and bands.** The
  pitch's "Leverage read" paragraph only states "win case is Δ<-0.01 with
  sub-1% param overhead" — there is no explicit cache reference (champion
  val, current baseline mean ± cache band), no formal WIN / NULL / DRIFT
  bands, and no reference to the 0.94M/3M noise envelope. Add a `## Pass/fail
  bar at tiny1m3m (seed 42)` section with: (a) cache reference (champion
  ~6.24 from 175-alibi, cache baseline 6.40±0.04), (b) WIN bar
  `trt_val ≤ ctrl_val − 0.01` AND clears the two-ctrl rule, (c) NULL
  `|trt_val − ctrl_val| < 0.01`, (d) DRIFT `trt_val > ctrl_val + 0.01`. The
  box noise floor is ±0.01 at 0.94M/3M tokens (per the §2 two-ctrl protocol)
  so any narrower bar will not resolve at this tier.
- **`α_b` design is ambiguous between convex blend and unconstrained linear
  combination.** The mechanism block declares `α_b ∈ R, init 0.0` and writes
  `W_O_eff_b = (1 − α_b) · W_O_b + α_b · W_O_shared`. With unconstrained real
  α_b, α_b=2 gives `−W_O_b + 2·W_O_shared` (a non-convex extrapolation), not
  a blend. The bet paragraph and "α_b=1 limit" both implicitly assume the
  formula is bounded to [0,1], so the unconstrained form lets the optimizer
  walk off the convex hull. The in-repo precedent (188, 206) uses
  `sigmoid(α_raw)` with `α_raw init = −10` (sigmoid ≈ 4.5e-5) for both
  init-near-0 AND bounded [0,1] semantics. Pick one and commit:
  - Option A (sigmoid-bounded, matches 188/206): `α_b = sigmoid(α_b_raw)`,
    `α_b_raw init = −10` (sigmoid ≈ 0), still bit-identical at step 0. State
    "convex blend" in the framing.
  - Option B (unconstrained real, current draft): keep `α_b ∈ R` init 0.0;
    state explicitly "linear combination, not convex blend; the α_b=1 limit
    is the only 'shared' point; the optimizer can extrapolate and we let it."
    This is the LoRA-style formulation. Different from 188/206 — call that out
    in the "Distinct from in-repo priors" section so the implementer does not
    copy the 188 sigmoid pattern by accident.
  Whichever you pick, add a one-sentence justification in the mechanism block
  ("why not the other form"). The current draft is silent, so the implementer
  has to guess.
