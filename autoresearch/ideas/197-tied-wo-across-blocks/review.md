# Review log — 197 Tied W_O Across Blocks

## r2 — 2026-06-15 — verdict: approve
- **Both r1 findings closed cleanly.**
  - *Pass/fail bar formalized*: a dedicated `## Pass/fail bar at tiny1m3m (seed 42)` section now cites the cache (`autoresearch/baseline-cache.json`) — champion 6.2403 from 175-alibi (pinned, std 0.0088, band 0.04) and current baseline 6.40±0.04, states the box noise floor ±0.01 per the §2 two-ctrl rule, and gives explicit WIN / NULL / DRIFT bands keyed to that floor (`trt_val ≤ ctrl_val − 0.01` AND clears the two-ctrl bracket for WIN; `|Δ| < 0.01` for NULL; `trt_val > ctrl_val + 0.01` for DRIFT). Numbers tie to a real control, the band is tight enough to resolve at tiny1m3m, and the "no in-between outcomes" clause is the right call.
  - *α_b committed*: the mechanism block is now explicit — `α_b_raw ∈ R, init -10.0` then `α_b = sigmoid(α_b_raw) ∈ [0, 1]`, with the convex-blend interpretation locked in and a paragraph explicitly rejecting the unconstrained-real `α_b ∈ R` form (with the `α_b = 2 ⇒ −W_O_b + 2·W_O_shared` non-convex counter-example). The justification matches the in-repo precedent at `188-cross-block-kv-share` (in queue, needs-plan) and `206-cross-block-ffn-share` (in queue, needs-plan, r1) — both use the same `sigmoid(α_raw)` + `α_raw init = −10.0` pattern for the same "init-near-0 AND bounded [0,1]" semantics. Implementer is pointed at the 188 pattern, not the rejected draft form.
- **Source check passes.** Dehghani et al. "Universal Transformers" ICLR 2019 / arXiv:1807.03819 — real, well-known. Lan et al. "ALBERT" arXiv:1909.11942 (2020) — real, validated at BERT-base/large/xlarge. The 171-dropconnect-wo citation in `closed.md:144` is the actual closed null (`Δ=+0.0478 wrong-sign`), and the discriminator paragraph correctly distinguishes 171's *weight-level regularizer* (training-time multiplicative noise) from 197's *parameter-level sharing* (inference-time structural collapse with learnable α_b).
- **Dedup check passes.** 197 is unique vs the in-repo queue: `188-cross-block-kv-share` shares K/V projections (not W_O); `206-cross-block-ffn-share` shares W_up/W_down (FFN, not attention); `171-dropconnect-wo` (closed) regularizes W_O weights, doesn't share them; closed-tying-axis is the *full* Universal-Transformer tying (stronger than 197's W_O-only soft blend). 197 is the narrowest tying lever in the portfolio and a distinct mechanistic question — it isolates W_O-collapse alone from the {QK, FFN, W_O} decomposition of the full-tying null.
- **Mechanism is a mechanism, not an HP lever.** Structural/architectural: changes the dimension of the parameter space (one global `W_O_shared` + 12 per-block scalars, in lieu of 12 per-block `W_O_b` matrices; the per-block `W_O_b` slots are *kept*, so treatment is param-*superset* of control by 4,108 params = +0.4% overhead, sub-noise). Step-0 byte-identical via `sigmoid(−10) ≈ 4.54e-5`. No LR/schedule/init-constant lever.
- **Tier / seed discipline holds.** Every claim is at tiny1m3m (0.94M · 3M tok, seed 42); no `screen20m` or larger-tier reference; one seed (42) only; no multi-seed protocol. The cache-keyed baseline (6.40±0.04) and the pinned champion (6.2403 from 175-alibi) are the only two reference points, which is the right discipline at this tier.
- **LoC budget is realistic.** Mechanism says "<50 LoC in `models/layers.py`" — matches the 188-cross-block-kv-share scope (config flag, per-block scalar parameter, forward-path blend). Implementable in this repo against the real files.
- **Transfer-risk tag (med) is justified.** Closest published analogs (Universal Transformers, ALBERT) share *all* parameters and validate at <100M-235M; 197 shares *only W_O* and only via a soft blend — narrower and structurally weaker. The med tag captures the "lever is novel; closest analogs (full tying) closed null in-repo" tension honestly. Scale-evidence section cites the right prior work.
- **Plan-level nit (not a reviser finding).** The mechanism block is silent on how `W_O_shared` is initialized. At `α_b ≈ 0` the value of `W_O_shared` is multiplied by `4.54e-5`, so step-0 byte-identity holds for *any* `W_O_shared` init (random Kaiming, zero, or a copy of block 0's `W_O_b`). The frontmatter `plain` says "init at the baseline's W_O" — that choice is fine and matches the plain's claim, but the mechanism block should echo it for self-consistency. This is a plan-level wiring detail, not a definition-gate finding; the code-implementer will pick the init. **No action from the reviser.**
- **Verdict: approve** — pitch is sound, mechanism is well-specified, falsifiable pass/fail bar ties to a real control, dedup vs in-repo priors is clean, sources are real and current, scope is tiny1m3m/seed-42. Routing to `needs-plan` with `round` reset to 1 per the on-approve protocol.

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
