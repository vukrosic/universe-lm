---
id: 232-poly-logit-stack
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T02:06:11Z
transfer-risk: low
plain: Stack the two most-negative single levers this session — 230 poly-alibi (convex distance bias on attention scores, Δ−0.0111) + 216 logit-scale (learnable output-logit temperature, Δ−0.0141). They act on opposite ends of the network (attention scores vs lm-head output), so they're orthogonal and should compose. Tests whether the two right-direction effects are additive (~−0.025 if so). No new model code — both flags already on the box.
---

# 232 — poly-alibi + logit-scale stack (do the two best right-direction levers add?)

## Motivation
This session's positional-curvature sweep found the productive direction (convex/sharper locality), but every single lever sits inside the 0.04 band:
- 230 poly-alibi (convex quadratic distance): Δ−0.0111
- 216 logit-scale (output temperature): Δ−0.0141 (closed-batch)
- 217 mix-norm: Δ−0.0030
- 231 kerple (concave): Δ+0.0449 (worse — confirms sharper-is-better)

To clear the gate you need a bigger effect than any single sub-band lever. The cheapest path: **compose** the two largest right-direction levers. They're orthogonal by construction:
- **poly-alibi** shapes the ATTENTION scores (per-head convex distance bias).
- **logit-scale** rescales the lm-head OUTPUT logits (softmax temperature).
Different ends of the network, no shared axis → expected additive if both gains are real.

## Mechanism
`@dataclass C(Tiny1M3MPolyAlibiConfig): use_logit_scale = True` — poly-alibi (use_alibi_bias off, use_poly_alibi on) + the +1-param output logit temperature. No `models/` edits: both flags are already wired and on the box (f360b33).

## A/B design
- **Bar**: champion `Tiny1M3MAlibiConfig` val 6.2539, band 0.04.
- **PASS/WIN**: val < 6.2003. **NULL** |Δ| < 0.04. Single seed (42).
- **Caveat**: deliberately a 2-lever stack — the question IS composition, so this is not a clean single-lever attribution. If WIN, a follow-up isolates which lever carries it.

## Pre-run verification
- **SMOKE_OK** ✓ — `_box_smoke.py _arq_232-poly-logit-stack.py` (both flags build + stack).
- Step-0: poly-alibi is step-0 identical (zeros init) and logit-scale inits to identity, so the stack is step-0 ≈ champion; both are few-param / fast-moving.
