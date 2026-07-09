# Taste log — 170 swiglu-ffn

## r1 — 2026-06-14 — verdict: accept
- **Sharp bet, real lever.** "SwiGLU's gating structure is a stronger inductive bias than plain GELU because it allows per-token soft routing through FFN sub-regions; Δval ≈ -0.01 to -0.04." Mechanism vs HP, identity-able, niche-fit. ✓
- **Information value is high either direction.** Win → gating binds at 1M, strong transfer signal to 135M. Null → gating doesn't bind at 0.94M, mechanistically distinct from 153's activation-shape null (`closed.md:112`). Both outcomes log a clean result. ✓
- **Not in `closed.md` axes.** `closed.md:112` closes only the FFN-**activation-shape** axis (153-relu2-ffn nulled at 0.94M); gating-structure is a different lever. No closed-axis dupe. ✓
- **Transfer risk: low.** Mechanism is per-token soft routing — scale-invariant. Every modern open LM (LLaMA 1/2/3, Mistral, Qwen, Gemma, OLMo, PaLM) ships SwiGLU; Shazeer 2020 validated at T5 1.1B-3B. Mechanism is not tier-locked. ✓
- **Gate-zero-init is clean.** `W_gate = 0` ⇒ `silu(W_gate·x)=0` ⇒ FFN output is exactly zero on step 0. Stable ReZero-style start; the optimizer must learn the gate. No accidental diverging-init concern. ✓
- **Concerns logged, not blocking.** d_ff_swiglu=170 (after 2/3-trim) is small — the gating benefit at LLaMA-scale partly depends on d_ff size. At 0.94M the gate may not have enough hidden units to route meaningfully. That's an honest null to surface, not a reject trigger. The miner flagged this trade-off in the design sketch.
- **Caveat vs 153.** Same FFN-module location but mechanistically distinct: 153 is `act(curvature)` (ReLU² inside a 2-matrix FFN); 170 is `act(W_up·x) ⊙ silu(W_gate·x)` (3-matrix FFN with explicit elementwise gate). Miner's mechanistic distinction is sound; 153's null does NOT pre-close 170.
- **Decision: accept.** Sharp, high-leverage, niche-fit, low transfer risk, clean identity init. Proceeds to definition gate (round reset to 1).
