# Review log — 198 Pre-FFN Attention Mixing

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