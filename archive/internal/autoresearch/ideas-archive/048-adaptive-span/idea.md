---
id: 048-adaptive-span
status: needs-plan
round: 1
updated: 2026-06-11T01:34:07Z
transfer-risk: high
---

# 048 — Adaptive Attention Span (per-head, on top of SWA-512)

## Source
Sukhbaatar et al., "Adaptive Attention Span in Transformers" (arXiv:1905.07799), 2019. Re-purposed for the SWA-uniform baseline rather than the full-context regime the paper targeted.

## Mechanism
Per head, learn a softplus-bounded scalar `s_h ∈ [s_min, sliding_window_size]` that parameterises a head-local causal mask on top of the existing SWA-512 mask. Composition rule (one line, code-gate-disambiguating): the effective mask for head `h` is `SWA-512_mask · head_span_mask(s_h)`, where `head_span_mask` keeps positions within `s_h` of the query and zeros the rest.

- `s_h` is initialised at **512** (the fire-ctrl's `sliding_window_size`), so the head-span mask is the *all-ones* over SWA-512's window at step 0 → identity relative to the SWA-512 ctrl.
- As training proceeds, `s_h` can **shrink** (a head decides it needs ≤16 context, sink-like) or stay near 512 (a head that the global cap is already right for). `s_h` is **explicitly capped at `sliding_window_size` (512)** — a head that "wants more" has no signal the model can express, because the underlying SWA mask does not widen. This isolates the per-head *parameterisation* claim (does heterogeneity within the existing SWA budget beat the uniform cap?) from a much bigger "heads vote to widen the global cap" lever.
- 1 scalar per head × 4 heads = **4 extra params**; cost is the mask build (computed once per forward, cached). Total LoC ≤ 60.

The lever is therefore *additive* to SWA-512, not a replacement. The A/B is `Tiny1M3MVQGainSWAHighRoPE250KConfig` (SWA-512, `sliding_window_size: int = 512`) ctrl vs `same + per-head learnable span (init at 512, cap ≤ 512)`.

## Scale evidence
The original paper reports SOTA on text8 and enwiki8 up to 8k context — strong long-context char-LM evidence but no modern LM pretrain at 100M+. Transfer-risk stays **high** for the *numerical* claim, but the *mechanism* is plausibly general: head-level span heterogeneity (a small set of sink-like heads plus a small set of broad-context heads, the rest concentrated near the working window) has been reported across scales in follow-up probing work on attention-head specialisation. The bet is therefore not a "long-context-only artifact" — if heterogeneity shows up at tiny1m3m, the *same inductive bias* (per-head cutoff as a learnable structural prior) carries forward into the 135M recipe; a null that re-confirms uniform-512 is also a slot worth spending (see below).

## Why it's worth a slot
**The bet (distribution claim, not cutoff claim):** per-head learned spans will *spread* — at least one head shrinks to ≤16 (sink-like, attends only to recent tokens) and at least one head stays near 512 (broad, covering the full SWA window) — and that spread beats the uniform-512 baseline by ≥0.005 val loss.

Both branches are informative:
- **WIN** (spread ≥ span² variance, val loss gap ≥ 0.005): head heterogeneity > uniform window — a mechanism for the 135M recipe (per-head structural priors beat a hand-picked global cap).
- **NULL** (spans concentrate within ±64 of 512): re-confirms SWA-512 as the right uniform cap, and **kills the #97 multi-scale-heads follow-up** (a closed sibling in `closed.md`) as worth pursuing — we'd already know learned per-head cutoffs collapse to a single value at this scale.

The current SWA-512 closed-win (uniform cutoff wins over 256/384/512/768/1024/2048 — `closed.md:21`, with the fire-ctrl `Tiny1M3MVQGainSWAHighRoPE250KConfig` baking in `sliding_window_size: int = 512`) pre-empts the original "is tiny1m3m wasting far-history compute?" framing — that question is *already* answered, the null cannot fire. The live question is whether the per-head *parameterisation* of that cutoff beats the global one, which is the bet above.
