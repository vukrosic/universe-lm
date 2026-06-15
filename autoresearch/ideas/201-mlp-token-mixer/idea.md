---
id: 201-mlp-token-mixer
status: needs-run
round: 1
updated: 2026-06-15T16:47:41Z
transfer-risk: med
plain: gMLP-style spatial gating unit (SGU) added alongside attention on the attention output, in 4 of 12 blocks (stochastic); α=sigmoid(α_raw) init -10 ⇒ bit-identical baseline at step 0; tests whether global per-channel post-attn mixing binds at 0.94M once 163's local-conv result is in.
---

# 201 — gMLP Spatial Gating Unit on Attention Output (Per-Block Stochastic)

## Source
- Liu et al., "Pay Attention to MLPs" (gMLP, NeurIPS 2021, arXiv:2105.08050) — uses a *spatial gating unit* (a global per-channel gate applied after an MLP block) as the sole token mixer (no attention). 79.4% top-1 on ImageNet.
- Tolstikhin et al., "MLP-Mixer" (NeurIPS 2021, arXiv:2105.01601) — alternative MLP-only token/channel mixing architecture. ImageNet scale.
- 163-v-mix-conv (in-repo, `needs-implement`) — depthwise Conv1d kernel=3 on attention output pre-W_O. 163 is **local** mixing; 201 is **global** mixing on the same axis. Different bet.
- 143-shortconv (closed borderline, Δ=−0.0134 not passing WIN bar) — pre-attention depthwise Conv1d on residual. Different placement.
- 157-conv-ffn (closed null) — depthwise conv post-FFN-activation. Different placement.

## Mechanism
Committed mechanism: **gMLP-style spatial gating unit (SGU)** on the attention output, pre-W_O, **added alongside attention** (not as a replacement, unlike gMLP proper).

The SGU computes a global per-channel summary of the attention output, applies a learned per-channel linear, and broadcasts the result back across the token axis:

```
attn_out = softmax(QK^T/√d) @ V                    # [B, T, d_model]
# Spatial Gating Unit (gMLP §3.1)
z = attn_out.mean(dim=T, keepdim=True)              # [B, 1, d_model]  (global avg-pool per channel)
z = gelu(z)                                         # nonlinearity
z = z @ W_g                                         # [B, 1, d_model]  per-channel learned linear
z = z.expand(-1, T, -1).contiguous()                # broadcast over T
attn_out_post = attn_out + α · z                    # α = sigmoid(α_raw), α_raw init -10 ⇒ α ≈ 4.5e-5
out = W_O(attn_out_post)
```

W_g has shape `[d_model, d_model]` = 4,096 params per block. To stay in the fair-by-param regime (compare 163 at +0.25%, baseline ~0.94M), apply the SGU in **4 of 12 blocks (per-block stochastic)**:

- 4 blocks × 4,096 = **16,384 params (+1.74% of 0.94M)**
- Plus 4 α scalars (negligible)
- **Bit-identical baseline at step 0**: α ≈ 4.5e-5 ⇒ SGU contribution is zero in fp32; construction via raw `Parameter(shape, requires_grad=True)` keeps RNG state aligned with the no-flag path.

**Dropped formulations** (round 1 muddled three shapes; r2 commits to one):
- Per-channel-across-T linear `W: [d_model, T]` — 262k params/block, infeasible (+335%).
- Kernel=T depthwise conv — 131k params/block, infeasible (+167%).
- `W_1, W_2 ∈ R^{d_model × d_model}` applied per-channel — that's the standard FFN, not a token mixer (no T-axis mixing).

## Design sketch
- **File**: `models/layers.py` — add `gmlp_sgu` module to `MultiHeadAttention`. Config flag `use_gmlp_sgu: bool = False`, `gmlp_sgu_block_stride: int = 3` (apply to block index 0, 3, 6, 9 by default).
- **Construction**: `self.sgu_W = nn.Parameter(torch.empty(d_model, d_model))` only when `block_idx % stride == 0` (4 of 12 blocks). Init via raw `.data` so RNG state matches the no-flag path.
- **Forward**: after `attn_output` is computed (post-SDPA, post-reshape) and before `W_O` projection, compute the SGU as above. `α` is a `nn.Parameter` shape `[]` scalar (one per SGU-enabled block).
- **Cost**: 4 × 4,096 = 16,384 params (+1.74% of 0.94M); forward: one global avg-pool + one matmul + one broadcast add per SGU-enabled block per step.
- **Identity-init-able ✓**, **mechanism (not HP) ✓**, **no data/infra needed ✓**, **fits in <100 LoC ✓**.

## Scale evidence
gMLP validated at ImageNet (Liu et al. 2021, up to ~73M params, 79.4% top-1) and the SGU formulation is the published primitive. **No in-LM validation known** — gMLP was ImageNet-only and did not test language modeling. The lever is a known architectural primitive applied to a fresh placement (alongside attention, post-attn pre-W_O, on a 0.94M LM), which is the standard recipe-side novelty for this pipeline. Transfer-risk: **med** — same as the original.

## Why it's worth a slot

**Sharp bet (one sentence)**: We expect 201 to be a clean null or small WIN (|Δval| ≤ 0.02, inside the ±0.04 noise band) at 0.94M regardless of 163's result, because attention's softmax already provides global content-based token mixing and a per-block-stochastic *position-based* gate has no clear axis the optimizer can exploit at 0.94M / 12L / 3M tokens; a WIN at Δval ≤ −0.02 would mean the post-attention output axis binds a global gate the softmax wasn't saturating on, which is plausible but rare at this scale.

**Conditioned on 163's outcome** (163 is still `needs-implement` as of this re-pitch):
- **If 163 (kernel=3 local conv) wins**: 201 tests whether the *range* of post-attn mixing matters — wider (global) vs 163's narrower (local). A 201 WIN-on-top-of-163-WIN would mean local was just a weak proxy for global; a 201 NULL-on-top-of-163-WIN would mean local was sufficient and the global SGU has no extra binding.
- **If 163 nulls**: 201 tests whether a *global* post-attn path (different mechanism from 163's local conv) is the binding constraint. A 201 WIN-on-top-of-163-NULL would mean the post-attn mixing axis wants global, not local; a 201 NULL-on-top-of-163-NULL would close the post-attn mixing axis at 0.94M.

**Testable prediction**: the optimizer's choice of α is the load-bearing diagnostic. If α stays near 0 throughout training, the lever didn't bind (null result, regardless of val-loss Δ). If α grows, by how much, and does val-loss follow? The α-trajectory is logged per-block at every checkpoint — that's the cheap post-hoc diagnostic that distinguishes "lever didn't bind" from "lever bound but in a different axis than expected."

**Why the position-based vs content-based framing matters**: attention is a *content-based* token mixer (softmax(QK^T) weights V by key similarity). The SGU is a *position-based* token mixer (global avg-pool ignores content, learns a per-channel scalar). Together they cover both axes; the question is whether the per-channel global signal carries information attention wasn't already providing.

**Why this slot and not more**: a 201 NULL closes the *global* post-attention mixing axis at 0.94M (a previously open axis, distinct from 163's local axis). A 201 WIN unlocks a cheap architectural primitive (16k params) that scales to the ladder. A NULL is informative even if the val-loss Δ is unobservable, because α-trajectory tells us whether the lever was even reachable for the optimizer.
