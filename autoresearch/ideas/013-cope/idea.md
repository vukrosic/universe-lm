---
id: 013-cope
status: needs-run
round: 1
updated: 2026-06-09T13:49:25Z
---

# 013 — CoPE (Contextual Position Encoding)

## Source
Golovneva et al., "Contextual Position Encoding: Learning to Count What's Important" (arXiv:2405.18719, 2024, Meta). Position is computed *per-head* from the *content* of nearby tokens: the position offset between token i and j is the number of "important" tokens (those with high dot-product to a learned probe) between them — not the literal index distance. Drop-in alternative to additive/relative position encodings.

## Mechanism
- For each head, learn a small "importance" probe `p ∈ R^D`.
- Position offset between i and j = count of k in [i, j] where `dot(x_k, p) > threshold` (or a soft sigmoid variant).
- Add this offset to the relative-position bias term in attention.
- Implementation: a `CoPE` module (~50 LoC, bumped from the original 40 — see RoPE-audit below) that replaces the RoPE application in `models/attention.py`. RoPE removed when CoPE is on.
- Distinct from FIRE (009, WIN): FIRE has a fixed decay kernel on absolute position; CoPE has a *content-conditional* position. Different inductive bias.

## Why it's worth a slot
- **Orthogonal to 009 (FIRE WIN) and to RoPE (closed).** FIRE won at 6.3234 vs ctrl 6.3875 (Δ −0.064), the current best baseline. If 013 wins on top of FIRE, we get a *stacked* content-conditional lever over a content-aware base. The A/B is "FIRE + CoPE vs FIRE" — not "CoPE vs last week's ctrl."
- **Paper result**: matches/exceeds RoPE on language modeling, wins on counting and selective-copy tasks.
- **Identity-safe** (probe-init near 0 → near-uniform positions → RoPE-like behavior at init).
- **Risk**: introduces an extra probe parameter and a sigmoid threshold; small LoC impact but one more thing that can go wrong at init.

## Hypothesis
Δ in [−0.01, −0.02] val loss on tiny1m3m. Mechanism: better position signal for in-context structure (e.g. rare word spacing, list boundaries). (Range tightened from [−0.005, −0.02] so a real effect must clear the ~±0.01 run-to-run noise floor.)

## Wiring
- New file: `models/cope.py` — `CoPE` module + integration in attention. ~50 LoC (was 40; RoPE-audit found more than one call-site, see below).
- `LLMConfig.use_cope: bool = False` (replaces RoPE when True).
- Probe init: `p ~ N(0, 0.02)` (mirrors FIRE PE's per-head content projection init in `models/fire_pe.py:54`, `nn.init.normal_(self.phi, mean=0.0, std=0.02)`). Concrete, pinned, no implementer judgement.
- Threshold τ pinned at 0 (midpoint sigmoid) for the primary A/B. **No τ sweep** — pipeline rule is one seed only, seed 42; sweeps are out of scope. A follow-up idea can probe one alternative τ as a single-experiment secondary if the primary lands.
- Pass/fail: PASS ≤ −0.01 vs **FIRE-equipped ctrl** (the current best baseline at 6.3234 per closed.md, not the V+q+SWA+HighRoPE reference). NULL/INCONCLUSIVE = |Δ| < 0.01. DRIFT > +0.01.

### RoPE call-site audit (per finding)
When `use_cope=True`, the plan must bypass every RoPE touchpoint. Confirmed sites from grep:
- `models/layers.py:12` — `self.rope = RotaryPositionalEmbeddings(...)` in attention block init.
- `models/layers.py:20` — `return self.rope(x_BTHD)` (the application call).
- `models/layers.py:548` — `self.rotary = Rotary(self.d_k, max_seq_len, base=rope_base)` inside `MultiHeadAttention`.
- `models/llm.py:207` — `use_qk_norm_post_rope` flag (downstream of RoPE — must be off when CoPE is on, or its meaning changes).
- `models/llm.py:217` — `self.rope_base` (still used by Q-per-token-RoPE branch — no change).
- `models/llm.py:322,346,347` — RoPE base / per-head base / partial rotary passed into `TransformerBlock`.
- `models/llm.py:210-213` — NoPE-style bypass comment (a precedent for "skip rotary entirely" exists).

Plan must gate the `Rotary(...)` construction and the `rope(...)` call on `not use_cope`, and the CoPE forward returns the relative-position bias consumed by attention in the same slot where the rotary bias is added. Multi-site edits push the LoC estimate from ~40 to ~50 — still under the 200 LoC cap.

## Notes
- 009 (FIRE) already closed as WIN (Δ −0.064 / −0.082) and is the current best baseline. The "wait for 009" dependency is dead — anchor against the FIRE-equipped ctrl directly.
- 009 (FIRE) is the closest analog. The A/B is "FIRE + CoPE" vs "FIRE alone" — a stacked lever, not a replacement for FIRE. Plan states this explicitly.
- 014-sigmoid-loss closed as a *rejected* idea for misusing arXiv:2405.18719; that arXiv ID is the *correct* CoPE citation (Golovneva et al., Meta, 2024). No collision — different idea, different mechanism.
