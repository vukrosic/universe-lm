# Review log — 003 soap

## r2 — 2026-06-08 — verdict: approve

**All 6 r1 findings addressed. Ready for plan.md.**

- **Routing committed (r1 finding).** Doc now has explicit "Routing (committed)" section: SOAP → `token_embedding.weight` + `emb_proj.weight` + `out_proj.weight`; AdamW → 1D scalars + `*.norm.weight`; Muon → 2D hidden. Mutually exclusive, no overlap. ✓
- **1D-param handling (r1 finding).** Explicitly states 1D scalars (`q_gain`, `k_gain`, `smear_gate`, `output_temp` τ, `vocab_bias` b_v) and `*.norm.weight` stay on plain AdamW with the reason ("eigendecomposition is meaningless on 1D params"). Also notes the `CautiousAdamW` (002) compat: SOAP and the cautious mask are independent levers and can ship in the same config. ✓
- **bf16 pre-flight (r1 finding).** Full pre-flight protocol added: 100 steps on screen20m, 3 abort criteria (NaN/Inf, imaginary > 1e-3, condition > 1e6), ≤5 min wall-clock, must pass before the 19m full run. Memory cost stated (~288 MB for `token_embedding` eigenbasis at d_model=576, vocab=49152). ✓
- **Expected-Δ and seed protocol (r1 finding).** Tightened to "−0.03 to −0.05; lower values are below the single-seed noise floor." 3-seed protocol (42/43/44) committed for the |Δ| ≤ 0.03 case; single-seed pass otherwise. ✓
- **Transfer argument (r1 finding).** Dedicated section: eigenbasis converges in O(1) steps (preconditioner, not a learned feature), scale-invariant, and the bf16 pre-flight is the explicit conditioning check. ✓
- **Flag placement (r1 finding).** Wiring section: "Add `use_soap: bool = False` to `LLMConfig` on the line after `use_cautious_muon: bool = False` (line 360)." ✓

**Hand-off to code-implementer:** promote to `plan.md`. The Wiring section is concrete enough to spec directly: new `optimizers/soap.py` (~190 LoC, copy `Adam._single_tensor_adam` body, prepend eigenbasis update every K steps) + 1 config flag + 4-line trainer gate at `training/trainer.py:142`. Pre-flight must run before the full screen20m and is a hard gate (any of the 3 abort criteria → don't promote to a full run).

## r1 — 2026-06-08 — verdict: revise

**Sound mechanism, falsifiable bar, clean dedup — but the doc has 4 uncommitted decisions that block promotion to plan.md.**

**5-check sweep:**
- **Source real & current:** Vyas et al. 2024, arXiv:2409.11321 (v2 Jan 2025). Authors plausible (Harvard/Meta). Code at github.com/nikhilvyas/SOAP. ✓
- **Mechanism is structural, not HP:** SOAP = Adam in Shampoo's eigenbasis, with periodic basis refresh (preconditioning frequency K). Inherits curvature benefits + Adam's adaptive step. Not a LR/schedule/init-constant lever. ✓
- **Not already closed:** not in `closed.md`; closed axes (V/Q/K/O embeds, SWA, RoPE, NoPE, post-norm, layer tying, MHA/GQA/MLA/Tied QK/dilated/logit softcap, norm zoo, NSA/diff-attn/hybrid, multiscale/parallel/attn sink) do not include Shampoo/SOAP. ✓
- **< 200 LoC:** paper's own implementation is <200 LoC. ✓ (integration with the repo's Muon routing will be the LoC driver, not the optimizer body)
- **Falsifiable bar with real control:** V+q+SWA+HighRoPE 4.6364 baseline. Pass ≤ 4.5887, fail > 4.6364, noise band |Δ| ≤ 0.05. ✓
- **Transfer argument:** paper's gains at 360M/660M; repo's screen20m is ~10M-20M. Mechanism (eigenbasis curvature, scale-invariant if the basis is well-conditioned) transfers; small-scale signal is uncertain. ⚠️ — see finding below.

**Findings (must be addressed before `needs-plan`):**

- **Routing decision not committed.** Doc says "SOAP replaces AdamW only, or does it also take some 2D params from Muon? Default: replace AdamW only." This is a question, not a plan. **Fix:** state explicitly "SOAP replaces the AdamW path (1D scalars, `*.norm.weight`, `token_embedding.weight`, `emb_proj.weight`, `out_proj.weight`); Muon keeps all 2D hidden weights." The default is right — Muon's orthogonalization on 2D hidden is the load-bearing mechanism, and SOAP on hidden would discard it. Commit this as a line, not a question.
- **1D-param handling not addressed.** Eigendecomposition is meaningless on 1D params. SOAP cannot and should not run on `q_gain`, `k_gain`, `smear_gate`, `output_temp τ`, `vocab_bias b_v`, or `*.norm.weight`. **Fix:** state that 1D scalars stay on AdamW (or a sign-masked variant if `use_cautious_adamw` from 002 ships first). Eigendecomposition runs on the 2D params only (`token_embedding`, `emb_proj`, `out_proj`).
- **bf16 stability — pre-flight, not afterthought.** Doc lists "verify on small params first" as a concern. That's a single line, not a commitment. **Fix:** add a pre-flight step: train 100 steps on screen20m with `bf16` enabled, log eigenvalue spectrum stats (any NaN/Inf? any imaginary-part > 1e-3? condition number > 1e6?). If any of these fire, abort and re-promote the idea as an fp32-only variant (or close it). The pre-flight must be ≤ 5 min wall-clock and run before the full 19m screen20m.
- **Expected-Δ at the noise floor.** Doc's expected Δ is −0.02 to −0.05; noise band is |Δ| ≤ 0.05. The lower half of the expected range is unresolvable on a single seed. **Fix:** either (a) tighten to "expected Δ −0.03 to −0.05; lower values unresolvable at this tier" or (b) commit to a 3-seed protocol when |Δ| ≤ 0.03 (single-seed pass, otherwise re-seed). State the decision.
- **Transfer argument — present but weak.** The doc says "paper at 360M/660M" but doesn't argue WHY a small-scale win would survive 25M→135M. **Fix:** add one paragraph: the eigenbasis converges in O(1) steps at any scale (it's a preconditioner, not a learned feature), so the curvature benefit is scale-invariant. The unknown is whether the eigenbasis is *well-conditioned* at small scale — that's exactly what the pre-flight check above measures. If the basis is well-conditioned at screen20m, the same mechanism applies at 25M/135M.
- **Pre-existing flag conflict.** `configs/llm_config.py:360` already mentions `use_cautious_adamw` (from idea 002). The new flag `use_soap` should land next to it, same convention. **Fix:** name the flag `use_soap: bool = False` and place it on the line after `use_cautious_adamw`.

**Hand-off to reviser:** all 6 findings are fillable in <30 min. The structure is sound; the doc just needs to commit its own defaults instead of asking questions.

## r0 — (none)
