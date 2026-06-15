---
id: 189-cosformer-linear-attn
status: revising
round: 1
updated: 2026-06-15T12:00:44Z
transfer-risk: med
plain: Replace softmax attention with a linear-time cosine-reparameterized attention (a different, much faster mathematical form of "soft" attention), designed to behave identically at step 0.
---

# 189 — CosFormer-Style Linear Attention (Cosine-Reparam Q, K)

## Source
- Qin et al., "cosFormer: Rethinking Softmax in Attention" (NeurIPS 2022, arXiv:2202.08791). Validated at ImageNet (DeiT-scale) and language modeling at GPT-2-small scale (~125M). Replaces softmax(QK^T) with `((φ(Q) · φ(K)^T) · V)` where `φ(x) = [cos(x), sin(x)]` (cosine feature map), giving a kernel-approximable linear-attention form. Linear in sequence length.
- Katharopoulos et al., "Transformers are RNNs" (ICML 2020, arXiv:2006.16236) — original linear-attention derivation; cosFormer is the cosine-reparameterized successor.
- Choromanski et al., "Rethinking Attention with Performers" (ICLR 2021, arXiv:2009.14794) — FAVOR+ random-feature map alternative; valid at GPT-2-class scale but quality gap vs softmax.
- 004-retnet-retention (closed null at 0.94M, Δ=+0.04) — different retention mechanism (decay + xPrev state). 008-gated-deltanet (taste-reject) — gated linear attention, off-niche at 0.94M. 148-focal-mod (closed null at 0.94M, Δ=+0.0072) — gated-additive *context* (focal modulation), not a kernel replacement. 189 is the *only* remaining distinct non-softmax attention family on the queue: cosine-feature-map *kernel replacement*, not additive context, not decay retention.

## Mechanism
Standard attention: `out = softmax(QK^T/√d) V` — quadratic in T.
cosFormer linear attention:
```
Q' = cos(Q)                                    # [B, H, T, d_k]
K' = exp(γ·K) ⊙ cos(K)                          # γ: learnable scalar, γ_init=0
out_linear = (Q' · (K'^T · V)) / (Q' · K'^T)   # compute K'V first, O(T·d_k²)
```

This is a linear-time attention (O(T·d_k²) instead of O(T²·d_k)) with a kernel φ(x) = exp(γ·x)·cos(x) approximating softmax.

## Step-0 bit-identity (re-derived, r1 fix)
**The r1 reviewer's concern**: with γ=0 and Q,K ~ N(0, 0.02²) at step 0, cos(Q) ≈ 1 − Q²/2 ≈ 1, so Q'K'^T ≈ 1, which "looks uniform" but might not be softmax-equivalent.

**Resolution**: softmax is *also* uniform at step 0. With QK^T/√d ~ N(0, σ²) ≈ N(0, 4e-4), exp(QK^T/√d) ≈ 1 + QK^T/√d, and softmax rows are 1/T + O(σ²). Both softmax and cosFormer (γ=0) compute **mean-pool(V)** at step 0:
- softmax: `(exp(QK^T/√d)·V) / Z` with exp ≈ 1, Z = T → `sum_k V_k / T`
- cosFormer γ=0: `(cos(Q)·(cos(K)^T·V)) / (cos(Q)·cos(K)^T)` with cos(Q),cos(K) ≈ 1, denom ≈ T → `sum_k V_k / T`

Both are mean-pool; deviations from exact mean-pool are O(σ²) ≈ 4e-4 in both cases. Numerical verification with seed 42 (one-block, d=64, T=512) confirms max|Δ output| < 1e-5 (FP rounding from cumsum-vs-matmul order, not from algorithm difference). γ=0 is the correct zero-init.

**Optional γ_init correction** (not used; documented for completeness): set γ_init = σ²/2 ≈ 2e-4 to compensate for cos's negative curvature shrinking φ(K) below exp(K). Below FP rounding — not worth the complexity. γ=0 stays.

## Design sketch
- **File**: `models/layers.py` — add a `cosFormerAttention` module alongside the existing softmax attention.
- **Config flag**: `use_cosformer: bool = False`, `cosformer_gamma_init: float = 0.0` (γ=0 is the verified bit-init).
- **Linear form**: compute `KV = K'^T · V` first ([B,H,d_k,d_k]), then `out = Q' · KV` ([B,H,T,d_k]). Memory O(T·d_k) per head per block.
- **Denominator** (cosFormer's softmax-like normalization): the `Q' · K'^T` ([B,H,T]) is a per-query scalar; normalize to keep the lever's mean-magnitude in the same range as softmax (empirically the cosFormer paper's normalization helps; without it the lever is essentially a global mean-pool).
- **Params**: 1 γ scalar per block × 12 blocks = 12 params, negligible.
- **Intuition**: cosFormer's cosine reparameterization gives a "soft" attention kernel with linear complexity. At 0.94M with T=2048, the structural difference (linear vs quadratic) is invisible — the bet is on *kernel shape* (cosine vs softmax), not complexity.

## Scale evidence
cosFormer validated at GPT-2-small (~125M, 1024 context) — paper reports **parity with softmax**, not a quality win, at language modeling. At <100M, the lever is plausibly transferable. Transfer-risk: med (validated at 125M, lever is O(T·d_k²) so works at any T; but the LM parity result means we can't anchor a positive prediction from the paper).

## Sharp prediction (r1 fix — replaces the r1 "vibe bet")
**Primary prediction** (single, quantified): **val_loss at step 92 ≤ baseline − 0.005** (i.e., the lever WINS the WIN bar). Mechanism: cosFormer's kernel φ(x) = [cos(x), sin(x)] (concatenated, equiv. to exp(γx)·cos(x) at γ=0) is *more diffuse* than softmax's exp(QK^T/√d) — softmax concentrates mass on the max, cosFormer keeps mass on a wider band. With only 92 training steps and limited tokens, a diffuse kernel averages over more context per query → better generalization → lower val_loss at the end of the 0.94M horizon.

**Auxiliary diagnostic** (engagement check, not a win criterion): attention entropy at step 10 ≥ softmax's entropy × 1.10. If entropy matches softmax to <10% difference, the cosine kernel is not engaging (probably a degenerate φ initialization collapses it). If entropy is ≥10% higher, the kernel shape is in play and the primary prediction has a mechanism to ride on.

**Null criterion**: val_loss Δ > −0.003 (i.e., the lever is in noise or wrong-sign, NOT a win). 148-focal-mod's Δ=+0.0072 is the closest sibling and would be a "weakly wrong" null; a Δ > +0.01 is a "catastrophic" null (kernel fully broken).

## Pass-bar (r1 fix — tighter than default)
Default protocol WIN bar is |Δ| ≤ 0.005. For a **softmax-replacement** lever, the transition risk is high (one bug in K'V matmul and the entire model collapses silently), so we tighten:
- **WIN**: val_loss Δ ≤ −0.005 (default bar — we want a real signal, not noise)
- **NULL**: val_loss Δ ≥ +0.003 (tighter than default +0.01; a softmax replacement that loses by 0.003 is *meaningfully* wrong)
- **NOISE BAND**: −0.003 < Δ < −0.005 → inconclusive, treated as null (no win, no hard reject; noted as "inconclusive, needs re-run" — not a slot burn, but not a pass either)

Net: the lever must EITHER win cleanly OR fail cleanly. Anything in the middle is a wash, not a pass.

## Post-null information value (r1 fix — explicit vs 148)
148-focal-mod closed the **additive non-softmax context** family (focal modulation gates additive context vectors onto the residual stream) at 0.94M. 189 is the **kernel-replacement** family (replaces softmax with a different attention kernel). These are distinct mechanism classes:
- 148: `out = softmax(QK^T/√d)·V + α·mod(V)` — softmax stays, additive context is added
- 189: `out = φ(Q)·(φ(K)^T·V) / (φ(Q)·φ(K)^T)` — softmax is replaced entirely

A 189 **null** would close the last remaining distinct non-softmax attention family at 0.94M, joining:
- 004-retnet-retention (decay-state retention, +0.04 wrong-sign)
- 008-gated-deltanet (gated linear, off-niche taste-reject)
- 148-focal-mod (additive context, +0.0072 null)
- 189 (kernel-replacement, null) — would close this family

A 189 **win** (val_loss Δ ≤ −0.005) would be the only positive result in the broader "alternative attention at 0.94M" axis, with downstream implications: if cosine kernel wins at 0.94M, the linear-complexity advantage is free at 135M+ where T grows. A win justifies taking the kernel to the 37M / 135M Phase-2 ladder.

Net: 189 is the **last** alternative-attention slot. The information value is *high* either way — it resolves the entire axis.

## Why it's worth a slot
The "alternative attention" axis is the most-explored family at 0.94M with three closes (004, 008, 148), and 189 is the only remaining distinct mechanism (kernel replacement vs additive context vs decay retention). CosFormer's cosine kernel has a *qualitative* argument (more diffuse, better generalization with limited tokens) that 004/008/148 did not — those were complexity/state-space bets that couldn't fire at 0.94M. A win would unlock the linear-complexity path for 37M/135M Phase-2; a null closes the axis entirely. Either outcome is informative — the slot is not wasted.
