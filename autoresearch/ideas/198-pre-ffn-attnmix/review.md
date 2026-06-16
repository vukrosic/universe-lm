# Review log — 198 Pre-FFN Attention Mixing

## r2 — 2026-06-16 — verdict: approve

### Re-review context
- Idea bounced back to `needs-review` round 2 from `needs-recode` via the `auto-implement` agent ("auto-fix gave up after 3 failed runs — needs a human", log: `2026-06-15T16:56:37Z`). The r1 review (this gate) had already approved the mechanism, the plan, and the LoC. The plan.md has been updated to a `Recode r2` section that explains: the build-smoke failures were a transient race + scp/connection infrastructure issue, NOT a code defect.
- The artifact is bit-identical to the r1 approved artifact. No code change has been made between r1 and r2. The staged changes to `models/layers.py` / `configs/llm_config.py` / `models/llm.py` for 198 are still in place (verified via `grep`: `use_pre_ffn_attn_mix` in all three files, `Tiny1M3MPreFFNAttnMixConfig` in `configs/llm_config.py:2954`, forward-branch wiring in `models/layers.py:7986`, the `MinimalLLM` flag read in `models/llm.py:663`, and the `TransformerBlock` kwarg + `nn.Parameter` registration at `models/layers.py:6340/7435-7440`). The `_arq_198-pre-ffn-attnmix.py` stub is on disk at the repo root. The local CPU build-smoke passes (`MinimalLLM(C())` → 949,068 params; fwd max-abs-diff `9.5e-7` vs `Tiny1M3MConfig`; 12 γ at `-10.0`; 12 forward branches taken).

### Re-litigated vs. settled findings (per protocol §4 "do not re-litigate settled findings")
- Source check (r1): FiLM 2017.07871, NormFormer 2110.09423, closed-lever cross-refs (164-Q, 168-AV, 021-V, 186-V) — all confirmed in r1. No new claims; r1 holds.
- Mechanism vs. hyperparameter (r1): structural, 1 scalar γ per block, FiLM-style, sigmoid-parameterized, init -10. Mechanism bar passes. Holds.
- Tiny1m3m scope (r1): all refs stay at tiny1m3m, seed 42, single tier. Holds.
- Dedup (r1): not a math duplicate of any closed lever. Closest analogs (021 V-residual cross-block WIN; 164 Q-carry cross-block null; 168 AV cross-block null) are all cross-block and on different signals (V/AV/Q, not pre-FFN full attention output within a block). 198 is the *only* intra-block pre-FFN candidate in the queue (confirmed by taste reviewer's portfolio-crowding analysis). Holds.
- LoC budget (r1): ≤25 LoC across `models/layers.py` + `configs/llm_config.py` + `models/llm.py`, well under the 200 LoC cap. Holds.
- Falsifiable bar (r1): `|Δ|>0.01` vs. `Tiny1M3MConfig` baseline cached at 6.3988±0.04. The plan now specifies WIN ≤ 6.3888, NULL inside ±0.01, DRIFT ≥ 6.4088 — sharper than the implicit bar, the right call.
- transfer-risk: med (r1): justified, holds. Lever is structural (FiLM-style), not scale-bound.

### Why approve (re-route to plan, not reject)
- The mechanism, plan, and code are all sound (r1). The build-smoke failures were daemon infrastructure issues (transient race for the first one, scp/connection failures for the next two), not mechanism / plan / code defects. The plan.md's r2 recode section documents this and confirms the local CPU smoke passes — the artifact is ready to run as-is.
- A `reject` here would close a sound, novel-axis, falsifiable idea on a non-mechanism ground. That violates the protocol's "Math duplicate of a closed lever is reject — cite the closed entry" — 198 is NOT a math duplicate. There is no closed entry to cite. Reject is wrong.
- A `revise` (→ `needs-revision` → reviser) would be appropriate if there were a *finding* blocking the run. There is no finding — the reviser would be asked to revise a non-defect, which is also wrong.
- `approve` (→ `needs-plan` → code-impl claims and routes to `needs-run` because the plan + code are already in place) is the correct call. The daemon's next pull will get the class and the build-smoke will pass.
- The "3-round cap" rule (forbidding `revise` at round 3) doesn't apply — round is 2, not 3. But the spirit is the same: the idea has been reviewed, the plan/code are sound, the gate should pass the work forward rather than bounce it.

### One micro-observation (non-blocking, plan-stage concern, not a finding)
- The plan's `Cost` section understates the `norm2` interaction: `ffn_in = norm2(x + sigmoid(γ) · attn_out.detach())` is the *pre-norm2* branch — but the `use_sub_ln` / `use_post_norm` / `use_parallel_block` alternate forward paths do not mix `attn_out_raw` into the FFN input even when `use_pre_ffn_attn_mix=True`. This is documented in the plan ("pre-norm path only … out of scope for this A/B") and the flag docstring is supposed to note it. I verified the `MinimalLLM.__init__` reads the flag with `getattr(config, ..., False)` so the default-OFF branch is bit-identical — no risk. Not a blocker; the implementer already acknowledged it.

### Conclusion
- Approve. Reset round to 1 for the code gate per protocol. Route to `needs-plan`. The plan.md is the r2-recoded plan and is the same r1 plan with the recode-explanation section added; the code-impl will see the plan is already in place and route to `needs-run` on claim.

## r1 — 2026-06-15 — verdict: approve

### Source check
- FiLM (Perez et al. 2018, arXiv:1709.07871) — real, well-known, used as a *general* conditioning mechanism. Confirmed.
- NormFormer (Shleifer et al. 2021, arXiv:2110.09423) — real. The author's distinction ("198 is FFN-input-side, not residual-side") is a fair discriminator vs. the NormFormer extra-LN pattern.
- Cited closed levers are correctly characterized: 164-Q (cross-block, attention-side, wrong-sign null Δ=+0.0360), 168-AV (cross-block, residual-side, null Δ=-0.0227 inside band), 021-V-residual (cross-block V-only, WIN Δ=-0.034), 186-V-carry (within-block but V-side, not FFN-input). 198 is *intra-block* and routes attention-output into the FFN input — distinct axis. Confirmed novel vs. `closed.md`.

### Mechanism vs. hyperparameter
- Structural / architectural: a 0-dim scalar γ (one per block) modulating the pre-FFN input via FiLM-style addition. Not an LR/schedule/init-constant lever. Passes the "is this a mechanism?" bar.

### Tiny1m3m scope
- All references stay at tiny1m3m (0.94M · 3M tok, seed 42). No screen20m / multi-tier talk. Pass.

### Dedup
- 021-value-residual (WIN): V carried *across blocks* via the residual stream — different placement and different signal (V is computed inside attention; 198 mixes the *full attention output AV* with the FFN input, intra-block).
- 164-Q-carry / 168-AV-carry / 196-block-residual-ema (closed): all cross-block. 198 is the first within-block, pre-FFN candidate. Not a duplicate.
- FiLM is the only outside-citation mechanism, and it's cited as inspiration not re-implementation.

### LoC budget
- `models/layers.py` — add `use_pre_ffn_attn_mix: bool = False`, `pre_ffn_attn_mix_init: float = -10.0` kwargs on `TransformerBlock.__init__`, register one `nn.Parameter(torch.zeros(1))` (or scalar buffer-with-grad) per block when flag is on, splice the `sigmoid(gamma) * attn_out.detach()` mix at the pre-norm2 point (or post-norm2; pin in plan.md). Plus pass `attn_out` into the FFN-input mixer branch. ~20 LoC.
- `configs/llm_config.py` — add the two flags to the block kwarg surface. ~4 LoC.
- `models/llm.py` — read the flag at model construction. ~3 LoC.
- Total well under 200 LoC.

### Pass/fail bar
- The proposal doesn't put a number on the bar. The taste reviewer's implicit bar is `|Δ|>0.01` vs. control. That is the right band for tiny1m3m (cached control noise ~±0.04–0.05, but a *fresh* ctrls-vs-ctrls comparison sits inside ~±0.01). The bar is tight enough to be falsifiable; a NULL inside band is a clean retire. Acceptable.

### transfer-risk: med — justified
- "Scale evidence" cites FiLM at visual-reasoning scale (Perez et al. 2018). No published "pre-FFN attention mixing" win for LMs. `med` is right — not "low" (no direct LM validation at this exact form) and not "high" (the lever is structural, not bound to a particular width/depth/embedding property). No change needed.

### Bit-identity at step 0 — minor tighten
- Spec claims "bit-identical at step 0" with `sigmoid(γ_raw=-10) ≈ 4.5e-5`. This is **fp32-noise bit-identical**, not literally bit-identical — `4.5e-5 * attn_out(O(1)) ≈ 4.5e-5` is a real perturbation, just below the noise floor for typical activations.
- Not a blocker (the same `sigmoid(-10)` convention is used by 188, the per-block U-Net skip gate mod used `-1.5`, and the codebase generally treats this as the "step-0 silent" idiom). But the implementer should:
  - Document the step-0 fp32 max-abs-diff in `plan.md` (target `<1e-5`); if it's above, consider `γ_raw = -20.0` (`sigmoid ≈ 2e-9`) to make the step-0 perturbation invisible at fp32.
  - Add the standard "verify step-0 bit-identity vs. baseline" check at plan time, against the cached ctrl numbers in `LEADERBOARD.md`.

### Pre-norm2 vs. post-norm2 placement — pin in plan
- The spec writes `ffn_input = attn_residual + γ · attn_block(x).detach()` but does not say whether `attn_residual` is the pre-`norm2` residual or the post-`norm2` signal. Two reasonable choices:
  - (A) `ffn_in = norm2(x + sigmoid(γ) · attn_out.detach())` — the FFN sees a normalized residual whose pre-norm content is the mix.
  - (B) `ffn_in = norm2(x) + sigmoid(γ) · attn_out.detach()` — the mix is added on the post-norm signal directly.
- These have different statistics: (A) puts the perturbation *inside* the RMS calculation (so the mix is renormalized), (B) puts it *outside* (so the mix is scaled by `1.0`, not `1/RMS(x+mix)`).
- The implementer should pick (A) to match the spec's plain reading ("`ffn_input = attn_residual + γ·...`" — `attn_residual` is the pre-norm2 quantity), document the choice in `plan.md`, and justify it in one sentence. This is a plan-pin, not a definition-gate blocker.

### Detach() — clean choice
- `attn_out.detach()` prevents the mixing term from back-propagating into the attention path's Q/K/V/O projections at step 0. This is the right call: without it, γ's gradient would be a sum of (a) the FFN-side loss gradient and (b) the residual-stream's gradient through the `x + dropout(attn_out)` path, which would compete with the attention path's own gradient signal at init and produce the classic "γ opens and closes" instability. The detach keeps the γ gradient cleanly tied to FFN-side loss only, which is the cleaner separator. Confirmed.

### Conclusion
- Mechanism is real, distinct from every closed lever, has a falsifiable bar, fits tiny1m3m, and is implementable in ~25 LoC. The bit-identity-tightening (γ_raw = -10 vs. -20) and the pre/post-norm2 pin are *plan-stage* decisions, not definition-gate findings. The code gate owns them.
- Approve. Reset round to 1 for the code gate. Route to `needs-plan`.