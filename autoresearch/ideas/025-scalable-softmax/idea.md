---
id: 025-scalable-softmax
status: needs-run
round: 1
updated: 2026-06-09T22:45:29Z
---

# 025 — Scalable-Softmax (SSMax, length-aware attention temperature)

## Source
Nakanishi, "Scalable-Softmax Is Superior for Attention" (arXiv:2501.19399), Jan 2025.

## Mechanism
Replace the softmax over attention logits with SSMax, which scales the logits by `s · log(n)` before the exponential, where `n` is the number of keys the query attends to (its causal context length) and `s` is a single learnable scalar **per head** (init 1.0 — the paper's natural starting point and an effective identity on the operator, not bit-identical to vanilla softmax at n>1): `attn = softmax( (s_h · log n) · QKᵀ/√d )`. Because the effective temperature grows with `log n`, the attention distribution does not flatten toward uniform as context length increases — each query can still concentrate mass on the few relevant keys at long range. Drop-in: compute a per-query `log(n)` vector (position index + 1) and multiply into the logits before softmax. ~20 LoC, `n_heads` extra learnable scalars (negligible at 0.94M, e.g. 4 at the tiny1m3m head count), no schedule.

## Why it's worth a slot
We expect a val-loss drop because at `max_seq_len=2048` later-position queries attend over hundreds-to-thousands of keys, and vanilla softmax provably flattens (logit variance fixed, denominator grows), wasting the limited concentration budget of a tiny model's few heads. SSMax restores per-position sharpness with one scalar per head. This is distinct from every filed/closed lever: logit-softcap (closed) *clamps* the logit range with tanh, 020-FoX applies a *content decay* on probabilities, 025 instead applies a *length-dependent temperature* on logits — an orthogonal axis (sharpening vs decaying vs clamping). It is parameter-near-free and fires every step. A null tells us softmax flattening is not the binding constraint at 2048/0.94M; a win is a ~20-LoC transferable sharpening lever stackable on FIRE.

### Pass / fail bar (numeric, tied to box noise floor)
Reference noise floor: the most recent in-session vast-34386 ctrl-pair is 6.3875 / 6.4050 (Δ=0.0175 across two ctrls, single-seed gap inside the bracket ≈ 0.006 when measured against a single ctrl); the leaderboard ctrl for the prior batch was 6.4287, giving a typical ctrl↔leaderboard drift of ~0.02. The bar for this idea:

- **WIN:** Δ vs the in-session ctrl ≤ **−0.01** (clearly outside the ctrl-pair bracket, mirroring the 011-cautious-lion WIN margin of −0.0312 ≫ gap 0.0009 and the 009-FIRE WIN margin of −0.064 / −0.082).
- **Informative NULL:** −0.01 < Δ ≤ 0 (sharpening lever is real but not binding at 2048/0.94M — still a *result*; logged to `closed.md` so it isn't re-mined).
- **Regress / box-drift:** Δ > 0 means SSMax hurts *or* the box drifted; runner re-runs ctrl to disambiguate before calling a clean null (per `PIPELINE.md` box-validation rule).
- **Anti-cheat:** the reviser and reviewer both treat the +0.0053 / −0.0053 inside-bracket POLYLOSS-style outcome as NULL, not WIN — the bar is the −0.01 threshold, not "any negative number."

## A/B scope (primary vs follow-up, vs FIRE and vs 016-qk-norm)
- **Primary A/B (the one that gets `needs-run`):** `ctrl = tiny1m3m baseline` vs `trt = tiny1m3m baseline + ssmax`, seed 42, single run each. This is the only A/B the runner executes.
- **Follow-up #1 (stack-with-FIRE):** `ctrl = baseline + fire-pe` vs `trt = baseline + fire-pe + ssmax`, seed 42. Only runs *after* the primary lands a non-regress result (WIN or informative NULL). This is the bet the idea's "stacks on FIRE cleanly" line refers to — it is not the primary.
- **Follow-up #2 (stack-with-016-qk-norm):** `ctrl = baseline + qk-norm` vs `trt = baseline + qk-norm + ssmax`, seed 42. Both ops are per-tensor multiplies on the same `scores` tensor pre-softmax, so order is irrelevant and they compose cleanly. Same gate as follow-up #1 — only after the primary clears.

## Stackability notes
- vs **009-FIRE** (additive position bias on scores): SSMax is multiplicative length-scaling on scores. Different operator at the same site; both can co-reside on the `scores` tensor from different ops. Per-query `log n` is a (T,) vector broadcast over (H,); FIRE's per-position bias is (T,) (or (H,T) for headwise variants). No interference.
- vs **016-qk-norm** (pre-softmax norm bound on QKᵀ): both act on the `scores` line; qk-norm bounds magnitude, SSMax rescales by length. Both are per-tensor multiplies so order is irrelevant. Composition is well-defined and is follow-up #2 above.
- vs **020-FoX** (content probability decay, post-softmax): different site (post-softmax), no interaction.
- vs **022-softpick / 024-gated-attention** (replace distribution / post-AV gate): different families, no interaction.

## Hard rules
tiny1m3m-only, one seed (42), ~20 LoC drop-in, mechanism not HP, per-head `s` init 1.0 (justified non-bit-identical step-0 by the paper's mechanism), no schedule, no auxiliary loss, no trainer plumbing beyond a flag.
