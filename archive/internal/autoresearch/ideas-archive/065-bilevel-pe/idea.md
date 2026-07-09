---
id: 065-bilevel-pe
status: needs-codereview
round: 1
updated: 2026-06-11T01:19:20Z
transfer-risk: med
---

# 065 — Bilevel positional encoding (fixed-stride two-band RoPE, stacked on FIRE)

## Source
He et al., "Two Stones Hit One Bird: Bilevel Positional Encoding for Better Length Extrapolation" (arXiv:2401.16421). Jan 2024. We borrow only BiPE's *two-scale index decomposition*; the paper's length-extrapolation framing is dropped (OOS at tiny1m3m's fixed 2048 ctx, same posture as 009-fire-pe's "## Transfer argument").

## Mechanism
Replace the single RoPE rotation `R(pos·θ_k)` with a **two-band rotation on a fixed-stride segmentation** of the same position, while leaving FIRE's additive attention bias untouched:

- Pick stride `S = 64` (deterministic — no text/newline metadata required, no learned segmenter). For position `p`, define `intra = p mod S` and `inter = p // S`. With `max_seq_len = 2048` this yields 32 segments of 64 positions each.
- Split the rotary frequencies into two disjoint bands of equal size. With `d_head = 16` (4 heads, `d_model = 64`), the upper-half frequencies (fast-rotating, high θ) encode `intra`; the lower-half frequencies (slow-rotating, low θ) encode `inter`. Per-channel, the composed rotation is `R(intra · θ_k)` on upper bins and `R(inter · (S · θ_k))` on lower bins.
- Because `intra + S · inter = p` and the lower-band frequencies are pre-scaled by `S`, the composed rotation per channel equals `R(p · θ_k)` exactly at init — **bitwise identical to plain RoPE**.
- A pair of **per-band scalar gates** `g_intra, g_inter` (init = 1.0 each) multiply the rotation magnitude per band; opening/closing them is how the lever fires. Gradients can shrink `g_inter` (suppress slow cross-segment phase, sharpening within-segment attention resolution) or shrink `g_intra` (the inverse experiment).
- Drop-in for the existing RoPE path in `models/layers.py`; **stacks on top of FIRE** (FIRE's additive logit bias is unchanged). Implementation budget: ~80 LoC (band split, pre-scaled inter freqs, index decomposition, two scalar gates).

## Scale evidence
Paper's headline win is length extrapolation — OOS for our fixed-ctx tier, not tested. The *mechanistic* claim we keep — that a two-scale frequency partition lets the model resolve intra-segment order more sharply than a single-band RoPE — has not been screened at sub-200M in the public literature. `transfer-risk: med`: the construction is structurally simple and identity-safe, but at tiny1m3m the win depends on whether 64-position segments are a useful inductive bias for 0.94M-param attention; no published evidence at this scale supports or refutes it.

## Pass / fail bar (tiny1m3m, FIRE-equipped ctrl from 009-fire-pe = 6.3234)
- pass: tiny1m3m val ≤ 6.3184 (Δ ≤ −0.005 vs FIRE-only ctrl)
- fail: tiny1m3m val > 6.3234
- noise: |Δ| ≤ 0.005 (single-seed, tiny1m3m) — treat as inconclusive
- expected Δ ≈ −0.005 to −0.015 (the band gates have to actively move; if they stay pinned near 1.0 the result is a designed null and still informative)

## Why it's worth a slot
**The differentiating bet (one sentence):** of every PE idea in the live queue (061-ALiBi monotone decay, 062-PosInterp phase compression, 064-XPOS norm-balanced rotary, 072-T5-RPE bucketed bias, 073-DeBERTa content-position disentangle, 009-FIRE content-aware bias), **only 065-BiPE tests a two-scale structural partition of the position index itself** — whether the model wants disjoint rotary frequency bands to encode "where in the segment" vs "which segment", rather than one uniform-frequency band. A null tells us tiny1m3m attention is indifferent to two-scale position factoring (closes a band of PE designs cheaply); a win tells us the "segments matter" inductive bias is worth 80 LoC even without text-derived boundaries.

**Stacking story vs FIRE (closed winner) and CoPE (closed destructive):**
- We *stack on FIRE*, not replace it. FIRE adds an input-dependent **additive bias on attention logits**; BiPE only changes how **RoPE rotates Q/K**. The axes are orthogonal — there is no shared scalar these two levers compete over.
- This is the inverse of 013-CoPE+FIRE (closed.md: trt 6.4659 vs ctrls ≈6.39, +0.069 — destructive). CoPE made *position itself* input-dependent, so it collided with FIRE's content-aware bias on the same "content × position" axis. BiPE keeps positions **deterministic** (fixed stride, no learned segmenter, no content gating), so there is no double-content-dependence to interfere.
- Why not replace FIRE? FIRE is the standing PE winner with Δ −0.064 to −0.082; the stronger test is "does an orthogonal positional axis stack on it" rather than re-litigating it.

**Segmenter rule (committed):** fixed stride `S = 64`. No newline detection, no text metadata, no learned boundary — those are confounders we cannot afford at tiny1m3m. A null with `S = 64` is interpretable as "the lever didn't fire", not "the segmentation failed".

**Identity / zero-init pathway:** band frequencies are pre-scaled so the composed two-band rotation per channel equals `R(pos · θ_k)` exactly at init; both per-band gates init at 1.0. Step 0 is bitwise identical to plain RoPE, and when stacked on FIRE is bitwise identical to 009-fire-pe's step 0. A null is therefore "the gates stayed near 1.0" — not "init pulled the model off the baseline manifold".
