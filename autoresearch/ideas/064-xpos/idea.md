---
id: 064-xpos
status: needs-plan
round: 1
updated: 2026-06-11T01:34:07Z
transfer-risk: med
---

# 064 — XPOS (length-dependent rotary scale, in-distribution bet)

## Source
Sun, Dong, Patra, Ma, Huang, Majumder, Wei, "A Length-Extrapolatable Transformer" (arXiv:2212.10554). Dec 2022.

## Mechanism
Standard RoPE multiplies q and k by a position-dependent rotation `R(pos, θ_d)`. XPOS adds a **per-position, per-dim magnitude scale** on top of the rotation:

```
q_xpos = R(pos, θ_d) · q   *   γ(pos, d)
k_xpos = R(pos, θ_d) · k   *   γ(pos, d)
γ(pos, d) = β ^ ( pos · (2d/D) / seq_len )    with β ∈ (0, 1)
```

So the rotated vectors are then **attenuated** — and the attenuation is harsher for higher-frequency dims and for larger positions. Norms are no longer preserved; that is the point. The lever trades RoPE's uniform-norm property for a *recency-biased rotary code*. Drop-in for the existing RoPE path in `models/layers.py` (~30 LoC, no new params, no new buffers beyond the precomputed `γ` table).

## Scale evidence
Paper's headline is 100× context extension on causal LM (extrapolation is OOS at tiny1m3m). The *secondary* claim — that the per-position attenuation bounds q/k norm divergence and stabilizes attention logits — is mechanism-level, not scale-bound, and has not been separately validated at sub-200M in the public literature. `transfer-risk: med`: structurally simple, identity-safe at step 0, but the win depends on whether the rotary-magnitude knob is helpful at fixed 2048 context, where extrapolation pressure is absent.

## Pass / fail bar (tiny1m3m, seed 42; ctrl = best RoPE-base baseline from closed.md = 6.4287)
- pass: tiny1m3m val ≤ 6.4237 (Δ ≤ −0.005 vs RoPE-only ctrl)
- fail: tiny1m3m val > 6.4287
- noise: |Δ| ≤ 0.005 (single-seed tiny1m3m) — treat as inconclusive
- expected Δ ≈ −0.005 to −0.015 — the lever is small in absolute terms because it only modulates existing rotary vectors; null at the "fail" line is informative (closes the rotary-magnitude family)

## Why it's worth a slot

**The predictive bet (one sentence):** we expect val loss to drop by **≥0.005** at tiny1m3m because XPOS's per-position scale `γ(pos, d)` bounds the *logit-norm variance across positions* (predict ~30% reduction) — RoPE alone keeps every position's q/k norm identical, so high-freq dims dominate the late-position logits unevenly; XPOS's `β^(pos·d/D/seq_len)` attenuation brings the late-position high-freq contribution back into the dynamic range of the early positions, stabilizing softmax entropy across the sequence.

**Falsifiable in-distribution signal:** compute `var_pos(‖q_rope[pos]‖_D)` across the 2048 positions at layer 0 mid-training (we already have hooks). XPOS should reduce this variance by ~30% vs plain RoPE. The signal is observable at any scale — no extrapolation needed to test it.

**Crisp mechanism at our scale (`seq_len=2048`, `d_head=16`):**
- At `pos=0`: `γ(0, d) = 1` for all `d` → step-0 output is bitwise identical to plain RoPE (zero-init property holds).
- At `pos=2047`, lowest-freq dim (`d=0`): `γ ≈ β^0 = 1` → no attenuation.
- At `pos=2047`, highest-freq dim (`d=D-1=15`): `γ ≈ β^(2047·1/2048) ≈ β ≈ 0.5` (with the standard `β=0.5` choice) → ~50% attenuation.
- Net effect on `‖q_rope[pos]‖` over `pos ∈ [0, 2047]`: with RoPE alone the norm is constant; with XPOS it drops monotonically by a factor that reaches ~`β^0.5 ≈ 0.71` at the last position (averaged over dims).

**Differentiation from the live PE queue (the crowded-family finding):**
- **PI (062, rejected)** — rescales the position *index* (`pos' = pos · α`) before RoPE. Pure **phase compression**. Doesn't touch vector magnitudes.
- **YaRN (063, in queue)** — piecewise phase rescaling + attention-logit scaling at extension. Pure **phase compression with logit correction**. Doesn't touch vector magnitudes.
- **ALiBi (061, needs-repitch)** — additive `−m_h · |i−j|` bias on attention logits. Pure **logit bias**, lives in attention-score space, not vector space.
- **BiPE (065, needs-taste r2)** — two-band frequency partition of the rotary index. Pure **frequency structure**, preserves norms.
- **XPOS (this)** — only lever in the queue that **modulates the magnitude of the rotated q/k vectors**. It is the *norm-balancing* knob, not the *phase-compression* knob, not the *frequency-band* knob, not the *additive-bias* knob.

So XPOS tests an axis the queue has not yet exercised. PI/YaRN subsumption argument is *not* needed; the family is not redundant on this axis.

**Why not just say "RoPE is already winning on this baseline":** 009-FIRE (closed WIN, Δ −0.064) is *not* a RoPE variant — it's an additive attention bias, completely orthogonal to rotary mechanics. The 500k RoPE-base winner is the *phase tuning* of RoPE; XPOS is a *magnitude post-process* on top of that. The two are independent dials on the same overall RoPE machinery, and the magnitude dial has never been tried in this queue.

**Stacking story:** identity at step 0 (γ=1 at pos=0), so it stacks cleanly on FIRE (009), QK-Norm (016), and Moonlight Muon (015) without disturbing any closed winner's step-0 manifold. The mechanism is *purely positional* — no content-dependence, no input gating — so it cannot collide with FIRE the way 013-CoPE did (closed.md: trt 6.4659, +0.069 destructive).

**Info value of a null at this bar:** a null at `Δ > −0.005` tells us *the rotary-magnitude knob does not help at fixed 2048 context* — the queue's only mechanism-level claim about RoPE's magnitude dimension is then closed cheaply. A win at `Δ ≤ −0.005` validates an axis none of PI/YaRN/ALiBi/BiPE touch. Both are informative; this is the lowest-regret slot in the current PE cluster.

## Open question we will not try to answer here
Extrapolation behavior is dropped (same posture as 065-BiPE's "## Transfer argument"). The lever is *only* tested on the in-distribution axis at our fixed `seq_len=2048`. The bet is specifically about logit-norm stabilization, not about length extension.
