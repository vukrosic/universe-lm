---
id: 205-per-head-mult-logit-scale
status: needs-taste
round: 2
updated: 2026-06-15T08:49:32Z
transfer-risk: low
plain: Per-head convex interpolation between softmax(·) output and uniform 1/T (init m_h≈0 so step-0 is byte-identical); bounded on the softening side, allowing some heads to flatten toward uniform but no head can sharpen.
---

# 205 — Per-Head Post-Softmax Convex Interpolation Toward Uniform

## Source
- 155-per-head-temp (closed null Δ=−0.0063 inside band) — per-head *pre-softmax* temperature τ_h in `scores / (τ_h · √d_k)`. Pre-softmax axis; lever is absorbed by Q/K updates at 0.94M/12L/4H.
- 152-attn-logit-bias (closed null) — per-head *pre-softmax* additive logit bias. Same family / same axis; same null pattern.
- 160-rms-gain-per-head (closed null |Δ|<0.005 inside band) — per-head *post-AV* gain (rescale `out_h` magnitude). Post-attention axis; null because the post-AV axis is plausibly redundant with W_O.
- 184-logit-scale (in-repo, needs-run) — *global* (not per-head) multiplicative logit scale at the LM head output. Different placement; OUTPUT-side lever, cannot be absorbed by per-head Q/K.
- 025-scalable-softmax (in-repo WIN with caveat) — *global* (not per-head) attention temperature. Global version won, per-head pre-softmax version nulled (152/155).
- Pattern: three per-head-attention-shape levers (152/155/160) all closed null at 0.94M. 184 won specifically because OUTPUT-side is not absorbable by Q/K. 205 is *post-softmax* (between QK and AV) and per-head.

## Mechanism

> **Pre-softmax form dropped (r1 finding 1).** A previous draft included the pre-softmax form `scores / (τ_h · √d_k)` alongside this one; that is algebraically a reparameterization of closed 155-per-head-temp (same math, different name). The only mechanism in this pitch is the post-softmax form below.

Standard attention: `scores = Q · K^T / √d_k` then `attn = softmax(scores)` then `out = attn @ V`.

**Per-head convex interpolation toward uniform** (post-softmax, bounded):
```
# per head h, per layer l: m_{h,l} ∈ [0, 1], init raw_h = -4 ⇒ m_h ≈ 0.018  (≈ identity)
m_h = sigmoid(raw_h)
attn_h_post = (1 − m_h) · attn_h + m_h · (1/T)
out_h = attn_h_post @ V_h
```
- At init `m_h ≈ 0`: `attn_h_post ≈ attn_h` to floating-point precision (max deviation ≈ 0.018 · 1/T ≈ 9e-6, well below fp32 noise).
- As `m_h` grows: head h's attention softens (interpolates toward uniform 1/T).
- As `m_h` shrinks: bounded in [0, 1] via sigmoid ⇒ head h's attention can NEVER be more peaked than the softmax output. **No sharpening path.**
- Parameters: H × L = 4 × 12 = 48 `raw_h` scalars (+0.005% of 0.94M).
- **Asymmetric boundedness vs 155**: 155 (`scores / (τ_h · √d_k)`) lets τ_h grow or shrink — both directions available, no natural bound on the soften path (τ_h → 0 blows up scores → softmax saturates → gradients vanish). 205 restricts the optimizer to soften-only, and the soften path itself is bounded (m_h ∈ [0, 1], output stays a valid distribution).

## Sharp mechanism claim (option c from r1 review — picked and quantified)

**Claim**: at convergence, the per-head mixing values will satisfy `m_h ≪ 1` for *most* heads and `m_h > 0.05` for *at most* a small minority — i.e., the dominant direction the lever takes is *soften-only, weakly*. Predicted distribution: `|{h : m_h > 0.1}| ≤ 2/4 heads per layer` (i.e., at most half the heads use the lever meaningfully), and for those that do, `m_h ≤ 0.3`.

**Why** (in one sentence): **asymmetric boundedness vs 155** — 205 can only soften toward uniform, never sharpen, so the optimizer never pays the saturation cost that drags 155 into null; 155's null came from τ_h moving in BOTH directions and being absorbed by Q/K updates, while 205 restricts motion to the cheap, second-order soften direction.

**Mechanistic argument**: boundedness makes the cost of "soften a head" essentially free at init — `∂L/∂raw_h ≈ 0` when `m_h ≈ 0`, so the optimizer can move any head a small amount with negligible cost. In contrast, 155's pre-softmax form pays a real cost for any non-trivial τ_h because it perturbs the QK-magnitude scale the rest of the model was trained against (the closed 155 entry says the lever was "absorbed by Q/K gradient updates" — that's the cost). The bet is that 205 escapes the absorption failure mode not by changing the math but by changing the *cost gradient*: a small `m_h` perturbation to the attention output is a second-order effect on the LM loss surface, while a small `τ_h` perturbation to the QK scale is a first-order effect that gets immediately cancelled by Q/K updates.

**Prediction summary**:
- WIN: at least one head per layer ends with `m_h > 0.1`, AND trt beats ctrl by Δ ≤ −0.01 (tighter pass-bar per r1 finding 5).
- PARTIAL WIN: heads use the lever (m_h > 0.1) but Δ inside band — the lever binds but the loss landscape doesn't reward it at 0.94M. **This is the most likely 0.94M outcome** given the three-null prior in the family.
- NULL: m_h stays near 0 for all heads AND Δ inside band — the lever fails to bind at all. Treat as: extends 152/155/160 to the post-softmax axis.

## Why it's worth a slot
This 0.94M A/B is a **screen**, not a verdict. The per-head-attention-shape family (152/155/160) is widely understood to fire at larger scale (H ≥ 12, L ≥ 24) where head specialization gives each head a non-trivial axis to exploit. The closed entries all carry the note "re-evaluate at >=135M Phase-2 with deeper stacks (L=24+) and more heads (H=12+)". A 0.94M result tells us:
- WIN: bounded per-head attention-shape is the missing axis; carry to 135M Phase-2 as a top candidate.
- NULL: per-head attention-shape is dead at 0.94M regardless of placement (pre-softmax 152/155, post-softmax 205, post-AV 160 all null) — saves the 135M queue from burning budget on this family.
- PARTIAL WIN (lever binds, loss doesn't reward): the most useful 0.94M signal — tells us the boundedness claim (option c) is supported but the loss surface doesn't care at this tier, so 135M is the right test.

The 135M Phase-2 follow-up plan if WIN or PARTIAL WIN: ship as `per_head_post_softmax_uniform_mix` with `m_h = σ(raw_h)` init `raw_h = -3` (so m_h ≈ 0.047 at init, almost-but-not-quite identity) and the tighter pass-bar `Δ ≤ ctrl − 0.01`.

## Design sketch
- **File**: `models/layers.py` — modify the manual attention path to apply per-head convex mix toward uniform on the post-softmax attention weights.
- **Config flags**:
  - `use_per_head_post_softmax_mix: bool = False`
  - `per_head_post_softmax_mix_init_raw: float = -4.0` (gives m_h ≈ 0.018 at init, close to identity)
- **Compute**: per head h, compute `attn_h_post = (1 − m_h) · attn_h + m_h · (1/T)`, then `out_h = attn_h_post @ V_h`.
- **Bit-identical at step 0**: m_h ≈ 0.018 ⇒ `attn_h_post ≈ attn_h` to floating-point precision (max deviation ≈ 0.018 · 1/T ≈ 9e-6, well below fp32 noise).

## Scale evidence
- Global pre-softmax temperature (025) won at 0.94M with caveat.
- Per-head pre-softmax (152, 155) and per-head post-AV (160) closed null at 0.94M.
- Per-head post-softmax (this idea, 205) is mathematically distinct from all three: it sits between pre- and post-softmax and is bounded on the soften side only.
- Transfer-risk: low (lever is well-defined; family-wide null pattern at 0.94M is documented; the 135M Phase-2 motivation is consistent with the closed entries' "re-evaluate at larger scale" notes).

## Pass-bar (tighter than default, per r1 finding 5)
Given the three-null prior in the per-head-attention-shape family (152, 155, 160 all null inside band at 0.94M):
- **WIN**: `Δ = val_loss(trt) − mean(val_loss(ctrl)) ≤ −0.010` AND two-ctrl rule passes (trt beats BOTH ctrls by ≥0.005) AND post-softmax m_h > 0.1 for at least 1 head per layer on average (lever actually binds).
- **PARTIAL WIN (informative)**: lever binds (m_h > 0.1 for some heads) but Δ inside `|Δ| < 0.010` band. **Counts as WIN for the 135M Phase-2 carry decision** — the loss surface at 0.94M is too tight to reward the lever even when it binds, but the binding is the signal.
- **NULL**: lever does not bind (m_h ≈ 0 for all heads) OR Δ inside band without binding. Closes per-head attention-shape on the post-softmax axis; extends 152/155/160 to the placement axis; do NOT re-pitch per-head attention-shape at 0.94M again.
- Default `|Δ| < 0.01` NULL band is replaced by the tighter `Δ ≤ −0.010` WIN bar — a 0.005 win on top of three null priors in the same family is not a confident signal.

## Why this is different from 184-logit-scale (in-repo, needs-run)
- 184 is *global* and at the LM head *output* (logit side of the LM, not attention). The output side cannot be absorbed by per-head Q/K updates — that's why 184 was accepted.
- 205 is *per-head* and *post-softmax in attention*. The placement is on the attention-side (interior of the network), not the output side. The lever sits between QK and AV — a different gradient path. 184's win does NOT predict 205's win; they test different axes.
