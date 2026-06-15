# Taste review — 196-ffn-glu-mish (MishGLU)

## r2 — 2026-06-15 — verdict: accept

**Context.** r1 raised two specific concerns: (a) the FFN-gating family is closed at 0.94M by 170 (with explicit "re-evaluate at >=135M" deferral), and (b) the tiny1m3m bet was unsharp. r1 offered two paths: defer to 135M, or sharpen the bet for 0.94M. The miner chose (2) and re-pitched the inner-activation axis with a specific mechanism and prediction.

**Where the r2 repitch lands.**

- **The math is right.** `dMish/dx|_{x=0} = tanh(softplus(0)) + 0·sigmoid'(0)·... = tanh(ln 2) ≈ 0.6`; `dSiLU/dx|_{x=0} = sigmoid(0) = 0.5`. The 20% origin-gradient advantage is exactly computed, not hand-waved. Both activations are 0 at the origin ⇒ step-0 bit-identity holds (satisfied by `mish(0)=0` and `silu(0)=0`).
- **The orthogonal-axis framing is structurally correct.** 170 closed the *outer* axis — does the GLU gating mechanism bind at this tier? Answer: borderline, gate weights grow slowly, gate values stay small. 196 asks a different question: *given* a borderline-engaged gate, is Mish or SiLU the better inner activation? The lever is in a different sub-family from 170 even though they share the gating structure. Both axes deserve to be tested.
- **The null is informative (the r1 bar was high here).** A clean null at 0.94M (|Δ| < 0.01) closes the *inner-activation axis* — a different orthogonal axis from 170's closed outer axis. After 196 we know: (a) the GLU gating mechanism doesn't bind at 0.94M (170), AND (b) the specific choice of inner activation within the GLU family also doesn't matter at 0.94M (196). That gives the reviewer a structured menu at 135M: where the outer axis binds, the inner-activation sub-choices (Mish, SiLU, GELU, ReLU) become the next variable to test. This is the high-leverage null the screen was asking for.
- **The bet is sharp.** Specific mechanism (20% origin-gradient advantage concentrated in |x| < 0.5 where ~38% of gate inputs live, post-Kaiming N(0,1)). Specific predicted magnitude (Δ = -0.005 to -0.01). Specific falsification criterion (|Δ| < 0.01 closes the inner-activation axis). The bet is testable at 0.94M/12L/92 steps, not deferred to 135M. r1's example of a "real sharp sentence" was more elaborate (tail-distribution argument with 30% gate-saturation reduction), but the r2 bet's math-and-mechanism form is sharp enough to falsify.

**What I'm not ignoring.**

- **Leverage is small.** The predicted magnitude (-0.005 to -0.01) sits inside the ±0.04 noise band; the WIN bar of `trt_val ≤ ctrl_val − 0.005` requires clearing the two-ctrl rule. r1 flagged this as a `safe-but-tiny` lever concern. The counter: the *null* value is high (orthogonal-axis closure), and at 135M where the gate binds, the inner-activation sub-choice becomes a real lever — knowing it doesn't matter at 0.94M rules out a 7th-FFN-variant re-test, and a win at 0.94M (rare-but-possible per the 20% gradient argument) would carry to 135M.
- **Family is crowded.** 170, 153, 157, 158, 156, 146, 117, 118 are all null on the FFN-side / capacity-mixing axis. The miner correctly distinguishes 196 as *gated-inner-activation*, not capacity-injection or un-gated-activation. The orthogonal-axis argument holds.
- **Predicted magnitude is at the edge of the bar.** Δ = -0.005 is exactly the WIN bar; the two-ctrl rule adds friction. Acceptable for a `safe-but-tiny-but-informative-null` bet.

**What I'm not blocking on.**

- Mechanism is real; step-0 bit-identity is exact; param count is identical to SwiGLU. Definition gate can implement as-is.
- Source citations clean (Shazeer 2020 + Misra 2019); no dup/axis-collision issue.
- Transfer-risk: med is correctly tagged.
- File-level plan (MishGLUFeedForward + `use_mish_glu` flag, structurally identical to SwiGLU) is correct.

**Verdict.** The r2 repitch earned the slot. The orthogonal-axis framing is structurally correct, the math is right, the bet is testable and falsifiable, and the null is informative at the high end of the r1 bar. Round 1 reset; definition gate's budget starts now.

## r1 — 2026-06-15 — verdict: revise

**Context.** The FFN family is densely closed at 0.94M:
- `170-swiglu-ffn` — null (Δ=-0.017, cache-NULL inside band). The note in `closed.md` is explicit: "closes FFN-gating axis at 0.94M alongside FFN-activation axis (153) — re-evaluate at >=135M Phase-2 where FFN capacity is the binding bottleneck."
- `153-relu2-ffn` — null (Δ=-0.0053, inside band). Closes the FFN-activation axis.
- `157-conv-ffn`, `158-gau`, `156-moa`, `146-sparse-ffn`, `117-soft-moe`, `118-MoD` — all null on the FFN-side / capacity-mixing axis.

196's own self-summary concedes the point: *"FFN-activation family is closed at 0.94M (153 + 170). 196 extends the family with Mish."*

**Taste-gap.** This is a structurally *real* variant (Mish's non-monotonic region is a genuine shape difference from SiLU; step-0 byte-identity is exact; the param count is identical) — that part is sharp. The problem is *portfolio fit* and *transfer*. 170 closed the **FFN-gating axis** (not just "SiLU specifically") at this tier with an explicit "wait for 135M" deferral. MishGLU is a *variant* of that closed axis (different inner activation, same gating mechanism, same 2/3-trick init, same bit-identical step-0). Running it at tiny1m3m does not advance the state of the screen — the most likely outcome is the 6th-FFN-variant null, which logs as `closed: 196 — null: trt=X vs ctrl=Y, FFN-gating axis re-confirmed at 0.94M, Mish activation also doesn't bind` and joins 170/153/157/158/156 on the closed pile.

The author's own bet is informative here: *"if 170 didn't bind, 196 may also not bind."* The 170-NULL is the strongest available prior, and 196's mechanism is a sub-component variation of 170's.

**Two ways to make this `accept`-sharp.** Pick one:

1. **Defer to Phase-2 135M.** Frame 196 as `MishGLU: a specific FFN-gating variant to re-evaluate when FFN capacity binds at 135M`. The miner has the time, and the tier actually exercises the gating shape (L=24+, d_model=512+, d_ff=2048+ gives the gate tens of thousands of params/block to develop non-trivial statistics, vs 4096/block for 92 steps at tiny1m3m). This honors 170's "re-evaluate at 135M" deferral and turns the family into a structured menu of activation choices *at the right tier*.

2. **Sharpen the bet for tiny1m3m specifically.** Name one *testable, non-rounding-error* prediction that MishGLU should make at 0.94M and SiLU-GLU cannot. The current bet ("Mish's non-monotonic region provides a stronger gradient signal on the negative-gate axis") is mechanistically true but at 0.94M the gradient is dominated by the few hundred FFN-gate updates, not by the shape of the activation in the small-negative region. If the bet is "SiLU was *unlucky* and a smoother-derivative activation will release 0.005-0.01 val", that's a `safe-but-tiny` lever, not a `big-if-true` one — and per the bar, the screen should prefer `big-if-true` even at higher null risk. A real sharp sentence would be testable at our scale: e.g., "MishGLU's non-monotonic gate reduces the worst-case gate-saturation rate (|gate| > 2) by 30%, which at our noise band (Δ=±0.04) predicts a +0.01 advantage on the *upper tail* of the val-loss distribution" — but the miner must show the math.

**What I'm not blocking on.**
- The mechanism is real and the file-level plan is correct (MishGLU module + `use_mish_glu` flag, structurally identical to SwiGLU, step-0 bit-identical). The definition gate can implement as-is once the bet is sharpened.
- Source citations are clean (Shazeer 2020 + Misra 2019). No citation/duplicate-axis issue.
- Transfer-risk: med is correctly tagged.

**Bottom line.** Not a `reject` — MishGLU is a real lever, not a duplicate. Not an `accept` — the family is closed at this tier with an explicit "wait for 135M" deferral, and the tiny1m3m bet is unsharpened. `revise`: pick (1) or (2) and re-pitch.

On receipt, the `flip.sh` will move the status to `needs-repitch` so the miner can re-pitch.
