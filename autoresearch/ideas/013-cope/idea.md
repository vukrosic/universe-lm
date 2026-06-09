---
id: 013-cope
status: needs-taste
round: 1
updated: 2026-06-09T09:35:00Z
---

# 013 — CoPE (Contextual Position Encoding)

## Source
Golovneva et al., "Contextual Position Encoding: Learning to Count What's Important" (arXiv:2405.18719, 2024, Meta). Position is computed *per-head* from the *content* of nearby tokens: the position offset between token i and j is the number of "important" tokens (those with high dot-product to a learned probe) between them — not the literal index distance. Drop-in alternative to additive/relative position encodings.

## Mechanism
- For each head, learn a small "importance" probe `p ∈ R^D`.
- Position offset between i and j = count of k in [i, j] where `dot(x_k, p) > threshold` (or a soft sigmoid variant).
- Add this offset to the relative-position bias term in attention.
- Implementation: a `CoPE` module (~40 LoC) that replaces the RoPE application in `models/attention.py`. RoPE removed when CoPE is on.
- Distinct from FIRE (009): FIRE has a fixed decay kernel on absolute position; CoPE has a *content-conditional* position. Different inductive bias.

## Why it's worth a slot
- **Orthogonal to 009 (FIRE) and to RoPE (closed).** If 009 wins, we get a fixed-decay lever; if 013 wins, we get a content-conditional lever. Run both → clean ablation.
- **Paper result**: matches/exceeds RoPE on language modeling, wins on counting and selective-copy tasks.
- **Identity-safe** (probe-init near 0 → near-uniform positions → RoPE-like behavior at init).
- **Risk**: introduces an extra probe parameter and a sigmoid threshold; small LoC impact but one more thing that can go wrong at init.

## Hypothesis
Δ in [−0.005, −0.02] val loss on tiny1m3m. Mechanism: better position signal for in-context structure (e.g. rare word spacing, list boundaries).

## Wiring
- New file: `models/cope.py` — `CoPE` module + integration in attention.
- `LLMConfig.use_cope: bool = False` (replaces RoPE when True).
- Probe init: small random, sigmoid threshold `τ` as a learnable scalar (init at 0).
- Pass/fail: PASS ≤ −0.005 vs V+q+SWA+HighRoPE ctrl. NULL = |Δ| < 0.005. DRIFT > +0.01.

## Notes
- 009 (FIRE) is the closest analog. If 009's result lands first, this idea should reference 009's evidence in its plan section.
- Threshold τ should be tuned but with a sensible default (τ=0, midpoint). A small sweep over τ in {−1, 0, +1} could be the secondary axis if the primary one lands.
