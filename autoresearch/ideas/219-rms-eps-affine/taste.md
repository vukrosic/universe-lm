---
idea: 219-rms-eps-affine
status: needs-review
round: 1
---

## r1 — 2026-06-16 — verdict: accept

- **Novel axis in a crowded neighborhood.** Closed at 0.94M: 017-sub-ln-sandwich, 183-pre-lm-head-rmsnorm, 181-cross-head-rmsnorm, 190-per-layer-qk-norm, 160-rms-gain-per-head, 169-qk-norm-depth all touched the norm/attention-shape family and closed null. But none of them touched the *interior* of RMSNorm — both the per-feature bias axis and the learnable-eps axis are genuinely new in the queue. The pitch correctly separates this from the closed family.
- **Sharp bet, despite the FFN-absorption caveat.** The pitch honestly flags that FFN-path bias is absorbed by the next Linear's bias term. The lever fires in the *residual stream* addition path, where `bias` is a per-feature constant added across 12L × 2 norms = 24 sites — a different axis than the post-AV gain or the per-head knobs. Win = "post-norm per-feature constant shift is a binding axis at 0.94M"; null = "LLaMA design is also right at 0.94M." Both are informative.
- **Clean init, cheap lever.** Step-0 byte-identical (eps at 1e-6 clamp, bias at 0). +1560 params (+0.17%), ~50 LoC. The pitch's *mechanism sketch* is straightforward and contains the one real footgun (eps parameterization: it self-corrects mid-sketch to use `eps.abs() + 1e-9` with eps initialized at 1e-6 — note that the sketch's initial `softplus(eps_raw)` tangent was abandoned for the cleaner raw-eps parameterization, which is the right call).
- **Niche fit is good.** Mechanism-shaped, identity-init-able, runs at tiny1m3m. Transfer-risk=low is justified: purely architectural, scale-agnostic (LLaMA-vs-LayerNorm bias choice is empirically a wash at 7B-540B per cited norm-baselines studies, and learnable eps is a single scalar per norm that scales freely).
- **Information value is high in both directions.** A null still logs (rules out bias+eps axes at this tier, complements the 017/190/183/181 residual-stream nulls with a *different* knob). A win would mean the residual stream at 0.94M has a per-feature constant offset that the gain-only RMSNorm cannot express efficiently — a real design choice, not a rounding error.
- **Handoff to definition gate:** please keep the `eps.abs() + 1e-9` parameterization (raw scalar, init 1e-6, clamped positive) and the per-feature `bias` init at 0. Bit-identical step-0 is the whole point. Run the 4-ctrl cluster and compare against the cached baseline band (per Vast runner harness memory); null is allowed and informative.
