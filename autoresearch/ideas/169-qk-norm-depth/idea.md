---
id: 169-qk-norm-depth
status: needs-review
round: 1
updated: 2026-06-14T06:23:48Z
transfer-risk: low
plain: Keep the QK RMS-norm from 016 (which won) but give each block its own learnable scale on top, starting at one — tests whether different blocks want different normalization strengths.
---

# 169 — Depth-Conditional QK-Norm (Per-Block Learnable Scale on the 016 Winner)

## Source
- 016-qk-norm (WIN, tiny1m3m) — applied RMSNorm to *both* Q and K with a single shared per-head learnable scale. Δ -0.014 vs both ctrls; pass-bar -0.005 cleared by ~3×.
- 162-q-only-norm (in pipeline) — applies RMSNorm to Q *only*; tests whether the Q-side is the binding axis.
- 165-k-only-norm (165, just filed) — applies RMSNorm to K *only*; the K-mirror of 162.
- 169 is a different axis: the *scale* parameter of the RMSNorm is *per-block* (one per layer) rather than per-head-shared. The bet: different blocks may want different normalization strengths (e.g. shallow blocks want less, deep blocks want more, or vice versa).
- "NormFormer" (Shleifer et al. 2021) introduced per-layer learnable gains on attention output; the depth-conditional norm family is well-validated in the FFN / attention output space.
- Per-block QK-norm gain is the same idea applied to the 016 winner.

Distinct from 016 (per-head-shared scale), from 152/155/160 (per-head per-layer scalars on different tensors), and from 161 (per-layer temperature — closed null because per-layer temperature fights the canonical QK scale prior).

## Mechanism
Apply RMSNorm to Q and K (as in 016), but give each block its own learnable `weight` parameter on the RMSNorm:
```
for each block l:
    Q_l = RMSNorm(Q_l, weight=alpha_l)     # alpha_l is per-block scalar, init 1
    K_l = RMSNorm(K_l, weight=alpha_l)     # same alpha_l shared between Q and K within a block
    logits_l = Q_l @ K_l^T / sqrt(d_head)
```
At init, `alpha_l = 1.0` for all l ⇒ identical to 016's per-head-shared-1.0 init ⇒ logits are bit-identical to 016's step-0 forward (and approximately bit-identical to the no-norm baseline modulo the RMSNorm rescaling, which 016 already accepts). The optimizer can then adjust per-block scales. ~15 LoC; +12 scalar params (one per block, +0.001% of 0.94M).

## Design sketch
- **File**: `models/layers.py` — `MultiHeadAttention.__init__` adds `use_qk_norm_depth: bool = False` kwarg; when on, replaces the per-head-shared `nn.RMSNorm(d_head)` with a per-block scalar parameter `self.qk_norm_scale = nn.Parameter(torch.ones(1))` *or* a per-block `nn.RMSNorm(d_head)` (one per block rather than per head). In `forward`, *after* projecting Q and K and *before* the QK matmul, apply `q = self.qk_norm(q) * self.qk_norm_scale` and `k = self.qk_norm(k) * self.qk_norm_scale`. The per-head RMSNorm stays (so the per-head shape is normalized) but the post-RMS scale is now per-block.
- **Config flag**: `use_qk_norm_depth: bool = False` (default off on `LLMConfig`).
- **Step-0 identity**: `qk_norm_scale = nn.Parameter(torch.ones(1))` init ⇒ scale is 1.0 ⇒ RMSNorm output is unchanged from per-head-shared 016 at step 0. Spec-allowed `fp32 max-abs-diff < 1e-3` tolerance for the RMSNorm rescaling (same as 016).
- **Intuition**: 016's WIN was on a *shared* per-head scale (= 1.0 init, single weight per head). The hypothesis: different blocks have different attention statistics (shallow blocks have broader attention, deep blocks have sharper attention), so a *single shared* scale may not be optimal. Per-block learnable scales let the model adjust the normalization strength per block. If 169-WIN > 016-WIN, the depth-conditional axis is binding. If 169 ≈ 016, the per-head-shared scale is sufficient.
- **Why now**: 016 is the strongest QK-side win. 162/165 are the *which side* tests (Q vs K vs both). 169 is the *depth-conditional* test: does the per-block scale matter once we have a shared scale? The data point we don't have is whether *block-specific* normalization strength compounds with depth at 0.94M.

## Scale evidence
RMSNorm family is well-validated at 1B+ (LLaMA 3, Qwen 2.5, Mistral). Per-block learnable scales on attention-internal tensors are a sub-claim but the primitive is well-tested. NormFormer's per-layer gains (Shleifer et al. 2021) is the closest direct analog, validated at 100M+ on long-document tasks. Transfer risk is **low** (well-validated primitive, narrow extension of 016's WIN).

## Why it's worth a slot
A win (or marginal Δ on top of 016) would tell us *depth-conditional normalization strength* is a binding axis at 0.94M, suggesting future QK-norm variants should consider per-block scales. A null would tell us 016's shared scale is sufficient at this tier and the depth-conditional axis is not the binding one. The lever is cheap (~15 LoC, ~12 params) and provides a clean attribution test for the 016 win.
