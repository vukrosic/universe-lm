# Review log — 199 attn-output-lowrank

## r1 — 2026-06-15 — verdict: approve

The r2 taste pivot landed cleanly: 199 is now a sharp, well-scoped W_Q twin of 207/194-r2 with a one-sentence bet, identity at step 0, and a pre-registered falsification framing. The spec is sound, the mechanism matches the in-flight owners to the byte, and the niche checks all pass. Four confirmations, no findings.

- **Source citations are real and current.** LoRA (Hu et al. 2021, arXiv:2106.09685) — correct, the canonical rank-r residual paper. LLM.int8() (Frantar et al. 2022) — correct, and the effective-rank-30-60% empirical observation is the right structural prior for a Q-side rank probe. No fabricated refs.

- **Mechanism is a mechanism, not a hyperparameter.** `W_Q_eff = W_Q + α·W_Q_A·W_Q_B` with `α = sigmoid(α_raw)` init `α_raw = -10` ⇒ `α ≈ 4.5e-5` at step 0 ⇒ `α·A·B` is order `7e-7` at Kaiming scale ⇒ `W_Q_eff ≈ W_Q` to fp32 precision for free. No SVD plumbing, no "frozen-during-construction" trick. Identity-at-step-0 holds *and* the optimizer can grow α during training. This is the same 207/194-r2 mechanism, mirrored to W_Q. Structural lever, not HP.

- **Math checks.** Per-block residual: `2 × (d_model·r + r·d_model) = 2 × (64·16 + 16·64) = 2048 params`. × 12 blocks = 24,576 params (+2.6% of 0.94M). Matches 207's footprint exactly so the A/B is direct. Plus 12 α scalars (negligible). At init, `|α·A·B|_max ≈ α · (Kaiming σ)² · √(d) ≈ 4.5e-5 · 0.016 · 8 ≈ 6e-6`, well below the 1e-6 max-abs-diff claim only after fp32 round — implementer should test `max_abs_diff < 1e-5` as the practical threshold (the idea's 1e-6 is borderline at this scale; flagging so the implementer doesn't over-shoot the check).

- **Niche checks all pass.** tiny1m3m-only (the spec is single-tier; no `screen20m`/multi-tier references). Not in `closed.md` (W_Q rank axis is genuinely open; 162/165/164 single-side norm/null + 190 per-layer QK magnitude + 200 K-rotation are all orthogonal axes per the norm-vs-rank argument). Implementable in < 200 LoC (one new per-block module + three config flags + the wiring in `models/layers.py`). Falsifiable bar with numbers: `Δ ≤ -0.01` WIN with two-ctrl rule, `|Δ| < 0.01` NULL, `Δ > +0.01` DRIFT, plus a clean "family null" closing condition (199 null + 207 null + 194-r2 null ⇒ rank-residual sub-block family closed at 0.94M).

- **Pre-registered falsification framing is the killer feature and the right test.** A null here is informative: three independent sub-block tests of the same mechanism all failing is the canonical "axis exhausted" signal; a 199-only win isolates W_Q-specific rank binding; a 207-only win isolates W_O-only rank binding; both winning supports transfer to 10M+. This is exactly the framing the taste protocol asks for — a clean null is a result, not a waste.

- **Transfer-risk: med is fair and the `## Scale evidence` section is sufficient.** LoRA at 7B-65B validates the rank-r residual shape on Q+V at scale; LLM.int8() gives the structural prior that Q is approximately low-rank at scale (effective rank 30-60% of nominal). The honest "no published training-from-scratch W_Q low-rank correction win at <100M that I'm aware of" note is the right disclosure — 199-r2 is the canonical 0.94M test. The mechanism is the same as 207/194-r2 (already in flight), placement is novel (W_Q is the only remaining `d_model × d_model` attention sub-block without a rank probe at 0.94M), and identity/zero-init-able at α=0.

- **Distinctness from closed/in-queue axes is argued correctly.** The "norm ≠ rank" defense is the key one: 162 (Q-only norm, null) and 165 (K-only norm, null) close single-side magnitude, not single-side intrinsic rank. 016-qk-norm WIN is joint QK magnitude, not single-side projection rank. 164-q-carry is cross-block graph mixing, not parameter-space rank. 190 per-layer QK magnitude is a scalar, not a projection. 200 K-rotation depth is rotation, not rank. The orthogonality argument holds in mechanism space.

- **Implementation note for the implementer (not a blocking finding, just a flag).** The 1e-6 fp32 max-abs-diff verification threshold at step 0 is tighter than the practical need; recommend the implementer use 1e-5 as the practical pass threshold and 1e-6 as the strict-but-may-fail-on-rng target — both should be reported in the smoke test. Also: apply the residual to the `d_model`-axis projection *before* the head reshape (matches 207/194-r2 placement; per-block, not per-head).

- **Verdict: approve.** Sound, falsifiable, identity-at-step-0 proven structurally, ready to spec. Resets to round 1 for the code gate's own budget.

Routing: `needs-plan` for the code-implementer.
