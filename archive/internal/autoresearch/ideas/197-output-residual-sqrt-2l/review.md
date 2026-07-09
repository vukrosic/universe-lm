## r1 — 2026-06-16 — verdict: approve

**Source.** DeepNet (Wang et al. 2022, arXiv:2203.00555) — well-known, real, validation 200-1000L. The lever is the paper's α = (8N)^(-1/4) ≈ 1/sqrt(2L) fixed depth-conditional scale on sublayer output, distinct from the Sub-LN (per-block LN) sub-mechanism that 017 closed. Citation stands, transfer-risk: low tag is correctly justified.

**Mechanism.** Structural: multiply attn_out and ffn_out by a single global scalar α before the residual add. Zero new params. Step-0 ≠ baseline by construction (the lever's purpose is the bounded regime from step 0), but the idea explicitly flags this and explains why — not an accidental identity fail. The idea also offers the learned-per-block alternative as a step-0-identity preserve; defaulting to the fixed form for the cheapest spec is the right call (the per-block variant just re-pitches the closed 130-rezero axis).

**Scope.** tiny1m3m only, seed 42. Confirmed `n_layers: int = 12` in `Tiny1M3MConfig` (configs/llm_config.py:2475), so α = (2·12)^(-1/2) = 1/√24 ≈ 0.2041 at run time. Plan says L=12, correct.

**Dedup.** Cross-checked closed.md:
- 017-sub-ln-sandwich (null, per-block LN) — different mechanism
- 130-rezero (null, per-block learned scalar init 0) — different (learned, not fixed; per-block, not global)
- 142-layerscale (null, per-channel learned gain init 1e-4) — different (per-channel, learned, init 1e-4 vs fixed global 0.204)
- 111-drop-path, 116-hyper-connections — different axes (regularizer / multi-stream)
- 196-block-residual-ema (taste-reject) — cross-block residual *mixing*, not fixed init scaling

197 is the first fixed-global-scalar lever in the family. The 017 closed entry even cites "DeepNet §3.1" — but for the Sub-LN sub-mechanism, not the α-scaling one. No mathematical duplicate.

**LoC budget.** Trivial: one config flag in `LLMConfig` (≈1 line), one multiply in `TransformerBlock.forward` (≈2 lines guarded by flag), one subclass `Tiny1M3MDeepNetAlphaConfig` (≈3 lines). < 20 LoC, well under the 200 budget.

**Pass/fail bar.** Concrete and resolvable at tiny1m3m: WIN if trt_val ≤ ctrl_val − 0.005 AND clears the two-ctrl rule; NULL if |Δ| < 0.01; DRIFT if trt_val > ctrl_val + 0.01. Cache reference points to 6.24 (current champion = 175-alibi baseline) and 6.40 (pre-alibi) — coherent. The 0.005 bar is ~1/8 of the 0.04 cache noise band, so a single-ctrl miss is expected; the two-ctrl rule is the actual gate. Taste flagged the DRIFT risk (0.204 may starve the residual stream) — that's a clean signal either way, not a code/plan problem.

**Transfer-risk tag.** low — correct. DeepNet validates the form at 200-1000L (paper headline scale); Primer validates a learned analog at 100M-1.5B. The fixed global scalar is the theoretically cleanest instance of the lever (no optimizer freedom → null is unambiguous, not "optimizer didn't use it"). Tag and citation match.

**Findings for the reviser (non-blocking — proceed to plan).**
- Plan should pick the global-scalar form (not the per-block learned alternative) — the latter is a 130-rezero re-pitch and would waste the slot.
- Config flag name is fine (`use_deepnet_alpha: bool`). Subclass name `Tiny1M3MDeepNetAlphaConfig` is consistent with existing naming (`Tiny1M3MReZeroConfig` at 2491, `Tiny1M3MSubLNSandwichConfig` etc.).
- The two-ctrl rule is the only thing that resolves 0.005 inside 0.04 — make sure the plan's "WIN" wording leads with the two-ctrl pass, not the absolute 0.005 bar, to avoid the code-implementer mis-calibrating the screen.

**Verdict.** Approve. The 0-param, init-time-only fixed global scalar is the cheapest possible lever in the pipeline with a real mechanism split from five closed per-block / per-channel depth-conditional forms. Null is unambiguous (no optimizer freedom); win is 0-cost and theoretically motivated. Move to `needs-plan` for the code gate.
