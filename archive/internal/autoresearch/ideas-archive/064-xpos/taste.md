## r1 — 2026-06-11 — verdict: revise

- **Comparative framing, not predictive.** "Narrower question than PI/YaRN" tells me XPOS is *different*, not what it *does*. A taste-worthy bet predicts a delta: "XPOS lowers val loss by Δy because it cuts q/k norm divergence by Δz% at training length 2048." Right now the bet is a vibe, not a hypothesis.
- **Crowded family — portfolio fit bar.** 061-ALiBi, 062-PI, 063-YaRN, 065-BiPE are already in `needs-taste`. XPOS is the 3rd RoPE-side variant (alongside PI, YaRN) in a 5-idea length-extrap cluster. The protocol's "5th variant in a row is a revise" applies. Differentiate, or argue XPOS strictly subsumes PI/YaRN.
- **In-distribution effect is the open question, not extrapolation.** The paper's headline is "100x context extension"; the in-distribution effect at 1k–2k training length is real but secondary. tiny1m3m trains at seq_len=2048, SWA=512 — extrapolation is not the slot's lever. The miner must argue the in-distribution mechanism (q/k norm rebalancing at every position) helps at our scale, not at hypothetical 100k context.
- **Rotary juice already extracted.** RoPE base sweep is closed with 500k winner (closed.md). XPOS is a *new* rotary tweak on top. The marginal cost of another rotary mechanism vs the marginal information is unfavorable.
- **Info value of a null.** A null at tiny1m3m is partially informative (rules out in-distribution XPOS benefit) but doesn't distinguish "XPOS doesn't help" from "XPOS only fires at extrapolation." That's a wasted slot in a crowded family.

### How to close the gap
1. Pick a **falsifiable in-distribution signal**: per-layer q/k norm divergence, attention logit entropy, or loss curvature. Predict a magnitude.
2. State the **delta** in one sentence: "we expect val loss to drop by ≥0.005 because XPOS bounds q/k norm growth across the 2048 positions."
3. **Differentiate from PI/YaRN** explicitly — either "XPOS is the only RoPE variant that handles norm drift *during* training, so it subsumes the others" or "XPOS is orthogonal to PI/YaRN and tests a different knob." Crowded queue means re-pitching without a clear contrast is auto-`reject` on round 2.
4. Make the **mechanism** crisp: at training length 2048, the per-position scale factor `g(pos, dim)` ranges from 1 (pos=0) to near-0 (pos=2047, low-freq dims) for high RoPE bases. Predict whether this attenuation helps or hurts and why.

## r2 — 2026-06-11 — verdict: accept
- The in-distribution signal is now explicit: logit-norm variance / q-k norm divergence at seq_len=2048.
- The mechanism is clearly distinct from PI/YaRN/ALiBi/BiPE because it is the only norm-modulating rotary lever.
- The predictive delta is now concrete and tied to the tiny1m3m control.
