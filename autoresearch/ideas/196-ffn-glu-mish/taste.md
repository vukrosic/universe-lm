# Taste review — 196-ffn-glu-mish (MishGLU)

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
