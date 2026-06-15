# Review log — 196-ffn-glu-mish (MishGLU)

## r1 — 2026-06-15 — verdict: approve

**Context.** Taste accepted r2 with a sharpened bet (inner-activation axis orthogonal to 170's closed outer-GLU axis, 20% origin-gradient advantage, Δ=-0.005..-0.01 win prediction). Round reset to 1 on taste's accept. This is the definition gate's r1.

**Source check ✓.** Shazeer, "GLU Variants Improve Transformer" (2020, arXiv:2002.05202) — the SwiGLU paper, real, foundational. Misra, "Mish" (2019, arXiv:1908.08681) — real, well-cited (3k+ cites). MishGLU itself is a natural compositional variant; no published direct test at scale, but each component is independently validated. Nothing fabricated.

**Mechanism is structural, not a hyperparameter ✓.** Inner-activation swap inside the GLU gate (Mish instead of SiLU) is a structural change to the gating function — not an LR / init / schedule lever. Param count is identical to SwiGLU (3 × d_model × d_ff = 49,152). No new params, no new shape.

**Step-0 byte-identity is exact, not approximate ✓.** `mish(0) = 0 * tanh(softplus(0)) = 0` and `silu(0) = 0 * sigmoid(0) = 0`. Both activations are 0 at the origin ⇒ the gate output `mish(W_gate @ x)` at step 0 (W_gate ~ Kaiming-init, post-Kaiming ~ N(0,1)) is `mish(0) = 0` for every gate input — bit-identical to `silu(W_gate @ x) = silu(0) = 0`. The 2/3-trick semantics hold automatically without an explicit zero-init. The lever's gradient pathway *does* differ (Mish' (0) ≈ 0.6 vs SiLU' (0) = 0.5), and that's the lever — the structural step-0 forward is genuinely bit-identical.

**Math is exact ✓.** `dMish/dx|_{x=0} = tanh(softplus(0)) + 0 · (sech²(softplus(0)) · sigmoid(x)) = tanh(ln 2) ≈ 0.6`. `dSiLU/dx|_{x=0} = sigmoid(0) + 0 · sigmoid'(0) = 0.5`. The 20% advantage is exactly computed. The "~38% of gate inputs in |x| < 0.5" claim for N(0,1) post-Kaiming is `erf(0.5/√2) ≈ 0.383` — exact.

**🔴 tiny1m3m only ✓.** Plan specs `Tiny1M3MMishGLUConfig(Tiny1M3MConfig)` with `use_mish_glu: bool = True`, single seed 42, 0.94M / 3M tokens. No screen20m / full ladder / multi-tier reference. ✓

**Not already closed ✓ — the orthogonal-axis argument holds.**
- **170-swiglu-ffn** (null, Δ=-0.0170, cache-authoritative) closed the *outer* GLU axis at 0.94M with explicit "re-evaluate at >=135M Phase-2" deferral. The question 170 closed: "does the GLU gating mechanism itself bind at 0.94M?" Answer: borderline-no.
- **196 is the orthogonal *inner* axis**: "given a (borderline-engaged) gate, is Mish or SiLU the better *inner gating activation*?" Different hypothesis, different null-falsification criterion, different gate-side mechanism (different activation function applied to the *gate* pre-activation, not the gating pattern itself). 170's null is the strongest available prior against any GLU-family lever at 0.94M, but the inner-activation axis is genuinely orthogonal — it survives even if the outer axis is closed.
- **153-relu2-ffn** (null, Δ=-0.0053) closed the *ungated* FFN-activation axis. 196 is the *gated* inner-activation axis. Different families.
- **157-conv-ffn / 158-gau / 156-moa / 146-sparse-ffn / 117-soft-moe / 118-MoD** all null on the *capacity-injection* axis. 196 is in the *gated-inner-activation* family — orthogonal.
- **196-block-residual-ema** (taste-rejected 2026-06-15) is on the residual-stream axis, not FFN. Not in conflict.

**Implementable in < 200 LoC ✓.** Sketched against the real files:
- `models/layers.py`: new `MishGLUFeedForward` module — structurally identical to `SwiGLUFeedForward` (~30-40 LoC: `__init__` with three `nn.Linear(d_model, d_ff, bias=False)` lines, `forward` doing `mish(W_gate @ x) * (W_val @ x) @ W_out.T`). `mish(x) = x * torch.tanh(F.softplus(x))`.
- `models/llm.py`: dispatch on `use_mish_glu` flag (~5-10 LoC, similar to the existing SwiGLU dispatch).
- `configs/llm_config.py`: `use_mish_glu: bool = False` on `LLMConfig`, plus `Tiny1M3MMishGLUConfig(Tiny1M3MConfig)` subclass with `use_mish_glu = True` (~10-15 LoC).
Total: ~50-65 LoC, well under the 200-LoC budget.

**Falsifiable bar ✓.** WIN = `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule. NULL = `|Δ| < 0.01` (closes the inner-activation axis orthogonally). DRIFT = `trt_val > ctrl_val + 0.01`. Cache reference champion ≈ 6.24, baseline ≈ 6.40. The WIN bar of -0.005 sits at the very edge of the noise band (Δ = -0.005..-0.01 prediction is right at the bar), but the NULL outcome is high-leverage (orthogonal-axis closure combined with 170's outer-axis closure gives a structured menu for 135M).

**transfer-risk: med ✓.** Tag is correctly justified in the `## Scale evidence` section — Shazeer 2020 (1.1B-3.9B T5) and Misra 2019 (30M-200M image classification) are each validated independently; MishGLU is their composition (no published direct scale test). The "med" call is right: high enough that 0.94M is not auto-fatal, low enough not to claim tested-at-scale.

**Findings to fix in plan.md (implementer attention, not blocking).**

1. **`mish(x)` numeric stability.** `mish(x) = x * torch.tanh(F.softplus(x))`. `F.softplus(x)` is well-behaved in fp32 for all x ∈ ℝ (it's `log(1 + exp(x))` with internal stabilization), and `tanh` is bounded in [-1, 1]. Watch only the gradient pathway (which is the lever — fine). No fp16 overflow risk. Suggest `mish = lambda x: x * torch.tanh(F.softplus(x))` as a top-level helper next to `silu` in `models/layers.py` so the smoke test can import it directly.

2. **Step-0 smoke test — forward vs gradient.** The implementer's self-check should verify `max-abs-diff(fwd_MishGLU_step0, fwd_SwiGLU_step0) < 1e-6` (forward is bit-identical because both activations evaluate to 0 at the origin, AND W_gate ~ Kaiming ~ N(0,1) gate inputs ⇒ output ≈ 0). The gradient pathway is *not* bit-identical — that's the lever, not a bug. Document this explicitly in plan.md so the implementer doesn't think the smoke test is failing.

3. **No explicit zero-init needed.** Unlike other GLU variants that require `W_gate.zero_()` to silence step-0, MishGLU's silence falls out of `mish(0) = 0` automatically — the same way SwiGLU's silence falls out of `silu(0) = 0`. The implementer should NOT add a zero-init on W_gate (it's not needed and would actually mask the gradient signal we want to preserve). Kaiming-uniform init (the standard `nn.Linear` default for `weight`) is correct.

4. **Config precedence — `use_mish_glu` overrides SwiGLU.** The dispatch order in `models/llm.py` should be: if `use_mish_glu=True`, use `MishGLUFeedForward`; else use `SwiGLUFeedForward`. Mutually exclusive in practice (only one FFN class per block), but document the precedence in plan.md so a future maintainer doesn't try to set both.

5. **Param-group routing — W_gate, W_val, W_out ride Muon (default).** All three are 2-D matrices ⇒ standard Muon routing per `optimizers/muon.py`. No `nn.Parameter` scalars, no embedding params, no special routing needed. Add one sentence in plan.md so the implementer doesn't reach for the AdamW fallback.

6. **Trajectory dump — gate activation statistics.** Per the r2 taste review's bet (the gain is concentrated in the last 20-30 update steps when the gate is most engaged), dump at end of run a small tensor `gate_activation_stats = {gate_abs_mean, gate_abs_max, val_loss_per_step_last_30}` so the NULL-vs-WIN attribution can read whether the gate ever engaged meaningfully. Cost negligible, no second run. This addresses the r2 bet's specific timing claim — without it, a NULL would just be "the gate didn't bind" without the timing-attribution signal.

**Why this is approved, not revised.** The mechanism is sound, the math is exact, the orthogonal-axis framing is structurally distinct from 170 (genuinely different research question), the step-0 byte-identity claim holds by direct computation, the null is informative (combined with 170's outer-axis closure, gives a structured inner-activation menu for 135M), the LoC is realistic, and the falsification criteria are explicit with a cache reference. The findings above are implementer polish, not definition-gate blocks. Round reset to 1 for the code gate.

<!-- reviewer appends one round per pass; do not hand-edit the frontmatter. -->