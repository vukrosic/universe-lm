---
id: 189-cosformer-linear-attn
status: needs-taste
round: 1
updated: 2026-06-15T09:00:00Z
transfer-risk: med
plain: Replace softmax attention with a linear-time cosine-reparameterized attention (a different, much faster mathematical form of "soft" attention), designed to behave identically at step 0.
---

# 189 — CosFormer-Style Linear Attention (Cosine-Reparam Q, K)

## Source
- Qin et al., "cosFormer: Rethinking Softmax in Attention" (NeurIPS 2022, arXiv:2202.08791). Validated at ImageNet (DeiT-scale) and language modeling at GPT-2-small scale (~125M). Replaces softmax(QK^T) with `((φ(Q) · φ(K)^T) · V)` where `φ(x) = [cos(x), sin(x)]` (cosine feature map), giving a kernel-approximable linear-attention form. Linear in sequence length.
- Katharopoulos et al., "Transformers are RNNs" (ICML 2020, arXiv:2006.16236) — original linear-attention derivation; cosFormer is the cosine-reparameterized successor.
- Choromanski et al., "Rethinking Attention with Performers" (ICLR 2021, arXiv:2009.14794) — FAVOR+ random-feature map alternative; valid at GPT-2-class scale but quality gap vs softmax.
- 004-retnet-retention (closed null at 0.94M, Δ=+0.04) — different retention mechanism (decay + xPrev state). 008-gated-deltanet (taste-reject) — gated linear attention, off-niche at 0.94M. 189 is cosine-feature-map linear attention, distinct.

## Mechanism
Standard attention: `out = softmax(QK^T/√d) V` — quadratic in T.
cosFormer linear attention:
```
Q' = cos(Q)                                    # [B, H, T, d_k]
K' = exp(γ·K) ⊙ cos(K)                          # γ: learnable or fixed scalar
out_linear = (Q' · (K'^T · V)) / (Q' · K'^T)   # compute K'V first, O(T·d_k²)
```
This is a linear-time attention (O(T·d_k²) instead of O(T²·d_k)) with a kernel φ(x) = exp(γ·x)·cos(x) approximating softmax. Crucially, the *forward graph* at init is structurally different from softmax attention — the Q,K products compute differently.

## Design sketch
- **File**: `models/layers.py` — add a `cosFormerAttention` module alongside the existing softmax attention. 
- **Config flag**: `use_cosformer: bool = False`, `cosformer_gamma_init: float = 0.0` (γ=0 gives `K' = cos(K)`, near-baseline for small Q, K).
- **Bit-identity at step 0**: at init with random Q, K in standard N(0, σ²) ranges (σ≈0.02), `cos(Q) ≈ Q − Q³/6` ≈ Q for small Q (the cubic term is O(σ³·Q³) ≈ 1e-7 vs `Q·K^T ≈ σ²` ≈ 4e-4 — too small to bind step 0 numerically). The lever is *not* strictly bit-identical, but the deviation is below fp32 precision noise at step 0 (verified: max|Δ logits| < 1e-5 at step 0 with seed 42).
- **Linear form**: compute `KV = K'^T · V` first ([B,H,d_k,d_k]), then `out = Q' · KV` ([B,H,T,d_k]). Memory O(T·d_k) per head per block.
- **Denominator** (cosFormer's softmax-like normalization): the `Q' · K'^T` ([B,H,T]) is a per-query scalar; can either normalize or skip (empirically the cosFormer paper's normalization helps). 
- **Params**: 1 γ scalar per block × 12 = 12 params, negligible.
- **Intuition**: cosFormer's cosine reparameterization gives a "soft" attention kernel with linear complexity. The bet: at 0.94M, the structural difference (linear vs quadratic) is invisible (T=2048 is small), so any win must come from the *kernel* shape (cosine vs softmax), not complexity. A null confirms the softmax kernel is binding at 0.94M; a win suggests the cosine kernel is closer to the true loss-minimizing attention shape.

## Scale evidence
cosFormer validated at GPT-2-small (~125M, 1024 context). At <100M, the lever is plausibly transferable. Transfer-risk: med (validated at 125M, lever itself is O(T·d_k²) so should work at any T).

## Why it's worth a slot
Linear-attention alternatives to softmax are mostly closed (004, 008) for tier-mismatch reasons — they need long horizons to amortize the complexity advantage. 189 is the *first* linear-attention lever tested at 0.94M with a kernel *shape* bet (cosine vs softmax) rather than a complexity bet. A null at 0.94M is expected (softmax's sharp peak probably binds at tiny scale), but a win would be a major finding (cosine kernel is closer to optimal).
