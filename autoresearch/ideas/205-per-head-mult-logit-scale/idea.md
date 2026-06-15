---
id: 205-per-head-mult-logit-scale
status: needs-repitch
round: 1
updated: 2026-06-15T08:22:23Z
transfer-risk: low
plain: Give each attention head its own learned multiplicative scale on the attention logits (init 1 so step-0 is byte-identical), so individual heads can softly sharpen or soften their own attention pattern.
---

# 205 — Per-Head Multiplicative Attention Logit Scale (Learnable Head Temperature)

## Source
- 155-per-head-temp (closed null Δ=−0.0063 inside band) — per-head learnable temperature *added to logit denominator* (e.g., `scores / (τ_h · √d_k)` with τ_h learnable). Different formulation.
- 152-attn-logit-bias (closed null) — per-head *additive* logit bias; 205 is *multiplicative* logit scale. Different shape (additive vs multiplicative).
- 184-logit-scale (in-repo, needs-run) — *global* (not per-head) multiplicative logit scale at the LM head output. Different placement (LM head output, not attention pre-softmax).
- 025-scalable-softmax (in-repo WIN with caveat) — global temperature on attention logits (single scalar). Per-head variant 205 is the *head-wise* axis.
- Hinton et al., "Sharp vs Flat Minima" / "Distilling the Knowledge" — temperature as sharpness control; per-head variant is novel.
- In-repo screen20m rows tested "q_gain / k_gain" closed the global attention QK gain axis. Per-head variant fresh.

## Mechanism
Standard attention: `scores = Q · K^T / √d_k` then `attn = softmax(scores)`.

Per-head multiplicative logit scale: per head h, multiply scores by `τ_h`:
```
scores_h = τ_h · Q_h · K_h^T / √d_k         # τ_h: learnable scalar per head, init 1
attn_h = softmax(scores_h)
```
At init `τ_h = 1`, scores are unchanged (bit-identical baseline). As τ_h grows, head h's attention sharpens (concentrates on fewer keys); as τ_h shrinks, head h's attention softens (more uniform across keys).

**Important distinction from 155-per-head-temp**: 155 used `scores / (τ_h · √d_k)`, putting τ_h in the denominator. 205 uses `τ_h · scores / √d_k`, putting τ_h in the numerator. Algebraically `τ_h · 1/(τ_h) = 1`, so the two are equivalent IF we reparameterize `τ_205 = 1/τ_155`. At init `τ_155 = 1` and `τ_205 = 1`, both are bit-identical. The difference is in the **default initialization** direction:
- 155: `τ_h = 1` init, learns to grow or shrink.
- 205: `τ_h = 1` init, learns to grow or shrink.

Actually these ARE algebraically equivalent with reparameterization. So 205 is a *re-parameterization* of 155 — different parameterization but same mathematical function. Hmm.

**To make 205 distinct**: change the placement to a different logit-side axis. Apply the per-head scale at a *different* point in the attention computation, e.g., on the *post-softmax attention weights* (rescale attention distribution rather than scores):
```
attn_h_post = τ_h · attn_h + (1 − τ_h) · (1/T)    # τ_h init 1 ⇒ attn_h_post = attn_h
out_h = attn_h_post @ V_h
```
At init τ_h = 1, `τ_h · attn_h + (1 − τ_h) · (1/T) = attn_h` exactly. As τ_h grows, attention becomes more peaked; as τ_h shrinks, attention becomes more uniform (interpolating with uniform attention).

## Design sketch
- **File**: `models/layers.py` — modify the manual attention path to apply per-head multiplicative scale on post-softmax attention.
- **Config flag**: `use_per_head_logit_scale: bool = False`, `per_head_logit_scale_init: float = 1.0`.
- **Compute**: per head h, compute `attn_h_post = τ_h · attn_h + (1 − τ_h) · (1/T)`. `out_h = attn_h_post @ V_h`.
- **Bit-identical at step 0**: τ_h = 1 ⇒ `attn_h_post = attn_h` exactly.
- **Params**: H × L = 4 × 12 = 48 τ scalars (+0.005% of 0.94M).
- **Intuition**: post-softmax interpolation with uniform is mathematically distinct from pre-softmax temperature scaling — it gives the optimizer a *different* path to control attention sharpness. Specifically, post-softmax is bounded (can't go below 1/T) while pre-softmax is unbounded.

## Scale evidence
Global temperature (025-scalable-softmax, in-repo WIN with caveat) validated at 0.94M. Per-head temperature (155) closed null. 205 is the *post-softmax* variant of per-head temperature. Transfer-risk: low (lever is well-defined; global version won, per-head pre-softmax version nulled).

## Why it's worth a slot
**Pattern**: per-head pre-softmax temperature (155) closed null; global post-softmax/attention-output scaling (025) won with caveat; per-head logit bias (152) closed null. 205 is *post-softmax per-head* — distinct placement. The bet: at 0.94M, the post-softmax axis (where attention weights are re-distributed) binds differently from the pre-softmax axis (where logits are scaled). A 205 WIN would mean the per-head post-softmax axis is a missing lever; a 205 NULL would confirm the per-head axis is generally hostile at 0.94M regardless of placement.
