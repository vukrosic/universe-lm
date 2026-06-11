# Taste log — 072 T5 Relative Position Bias

## r1 — 2026-06-11 — verdict: accept
- Sharp bet: isolates the bucketed-distance prior from FIRE's smooth decay kernel. Either outcome is informative — a win says quantized buckets are the active ingredient in learned PE; a null says the smooth kernel matters. Clean counterfactual against the current PE WIN (009-fire-pe, Δ −0.064/−0.082).
- Niche fit holds: mechanism not HP, zero-init bias table makes step-0 identity exact, the lever fires at any context length so tiny1m3m (1024 ctx) is fine. transfer-risk: low is well-justified — T5 ships the bias to the 11B checkpoint, so the mechanism's value is mainstream at scale, not a tiny-scale artifact.
- Not in `closed.md` and not in `LEADERBOARD.md`; closed PE work covers RoPE base sweep, NoPE, post-norm, layer tying, MHA/GQA/MLA — but the T5 learned-bucket form is untested in this project. 013-cope (drift when stacked with FIRE) is content-conditional, distinct mechanism.
- Portfolio note: the PE family is crowded (061-alibi, 062-pos-interp, 063-yarn, 064-xpos, 065-bilevel, 073-deberta-disentangled all queued at needs-taste). That argues *for* running the canonical learned-bucket reference first to anchor the rest — not against it.
- Routing to definition gate at round 1.
