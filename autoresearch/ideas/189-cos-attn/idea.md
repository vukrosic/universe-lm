---
id: 189-cos-attn
status: needs-taste
round: 1
updated: 2026-06-15T08:30:00Z
transfer-risk: med
plain: Replace the dot-product attention with cosine similarity (Q·K divided by their lengths), so the score is bounded between -1 and 1 and doesn't blow up with Q/K magnitude.
---

# 189 — Cosine Attention (cosFormer / MoCoFormer-Style, Bit-Identity via Q,K Unit-Norm)

## Source
- Qin et al., "cosFormer: Rethinking Softmax in Attention" (NeurIPS 2022, arXiv:2202.08791) — replaces `QK^T / sqrt(d_k)` with `(Q/||Q||_2) · (K/||K||_2)^T` (a cosine similarity, bounded in `[-1, 1]`). Validated at 100M-1.5B on language modeling and 100M-300M on image classification.
- Henry et al., "Query-Key Normalization for Transformers" (Kaddour et al. 2022, arXiv:2010.04245, "QKNorm") — empirical observation that the attention logit magnitude grows with d_k and contributes to training instability; cosine attention is a *hard* bound that prevents this.
- "Multi-head Attention with Disagreement Regularization" (2021) — Q,K normalization is a recognized axis. (Note: in-repo 016-qk-norm is the *RMS-norm*-form, which is similar but not bounded to `[-1, 1]`.)
- In-repo context: 016-qk-norm is the WIN. 016 is *RMSNorm* — output is `Q̂ · K̂` where `Q̂, K̂` are RMS-normalized (each component has unit RMS over the head dim). cosFormer is *L2* normalization — output is unit-L2 Q · unit-L2 K, which is the cosine similarity. The two normalizations differ: a vector of all-equal components has RMS=1 but L2=sqrt(d), so the L2 normalization produces strictly smaller dot-products. cosFormer is the *stronger* (more aggressive) normalization axis, with theoretical justification from the cosFormer paper.
- 169-qk-norm-depth (null at 0.94M) and 162/165 (Q-only, K-only) — closed attribution. 189 is the L2-normalization analog, not the RMS-normalization analog.

## Mechanism
Standard pre-softmax QK product:
```
scores = Q @ K^T / sqrt(d_k)         # [B, H, T, T], magnitude grows as O(sqrt(d_k))
weights = softmax(scores)             # can be very sharp at large d_k
```
With cosine attention:
```
Q_norm = Q / ||Q||_2                  # unit-L2 normalization, per (B, H, T) row of Q
K_norm = K / ||K||_2                  # unit-L2 normalization, per (B, H, T) row of K
scores = Q_norm @ K_norm^T            # bounded in [-1, 1], no /sqrt(d_k) needed
weights = softmax(scores / tau)       # tau is a learnable temperature (default tau=0.1, can be init 1)
out = weights @ V
```
The output is a true cosine similarity: `scores[i, j] = cos(angle between Q_i, K_j)`. Bounded in `[-1, 1]`, scale-free (doesn't depend on Q, K magnitudes), and the gradient through the L2 norm is well-behaved.

**Step-0 byte-identity**: at step 0, `Q` and `K` are the W_Q, W_K projections of the embedded input. With the standard `nn.Linear` init (Kaiming / Xavier), the rows of Q, K have approximately unit L2 norm (variance ≈ 1/d_k, magnitude ≈ 1/sqrt(d_k) per component, so ||row||_2 ≈ 1). After `Q / ||Q||_2` and `K / ||K||_2`, the rows are *exactly* unit-L2. The dot product `Q_norm @ K_norm^T` is then in `[-1, 1]`.

For the bit-identity claim: with the standard init, `Q = K = W · x` where `x` is the embedded input. `W` is initialized such that `W · x` has output variance matching input variance (Kaiming). The Q rows are random Gaussians with magnitude ≈ 1. The cosine similarity of two independent random unit-L2 vectors is `O(1/sqrt(d_k))` per component, so the **per-pair** cosine similarity is `O(1/sqrt(d_k))` which is small (≈ 1/4 for d_k=16). This is *different* from baseline `QK^T / sqrt(d_k)`, which at init has magnitude `O(1)` (independent Gaussians with std=1/sqrt(d_k), dot product has std=1/sqrt(d_k) — wait, the baseline QK^T/sqrt(d_k) at init with Kaiming init has each component O(1/sqrt(d_k)), the sum is O(1)).

**So at step 0 the cosine attention is NOT bit-identical to baseline** — the per-pair attention logit magnitude is different (`O(1)` for baseline, `O(1/sqrt(d_k))` for cosine). This means the lever is *not* step-0 byte-identical. However, the lever is *init-similar* — the softmax outputs are both approximately uniform at step 0 (small logits → near-uniform weights), and the loss is the same in expectation. Step-0 *forward* is approximately identical; step-0 *backward* differs in magnitude (cosine grad is bounded, baseline grad is not). The lever should be classified as "step-0 ≈ baseline" rather than strict byte-identity.

**Alternative parameterization for true step-0 byte-identity**: use a *gated* cosine attention with a learned scalar `g_l` (per block, init 0) that mixes the cosine scores with the baseline scores:
```
gated_scores = (1 - g_l) * baseline_scores + g_l * cosine_scores
```
With `g_l = 0` init, `gated_scores = baseline_scores` exactly, and the optimizer grows `g_l` toward 1 to engage cosine attention. This adds 12 scalars (1 per block) and gives strict byte-identity. Default to this gated form.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_cos_attn: bool = False` to the block. Replace the `Q @ K^T / sqrt(d_k)` step with a gated cosine form:
    ```
    Q_norm = F.normalize(Q, dim=-1, eps=1e-6)
    K_norm = F.normalize(K, dim=-1, eps=1e-6)
    cos_scores = Q_norm @ K_norm.transpose(-1, -2) / tau
    base_scores = Q @ K.transpose(-1, -2) * (d_k ** -0.5)
    scores = (1 - g_l) * base_scores + g_l * cos_scores
    ```
    where `g_l = sigmoid(g_param_l)` (init 0 ⇒ `g_l = 0.5` exactly... not byte-identity).
  - Better: `g_l = 0.5 * (1 + erf(g_param_l / sqrt(2)))` is also not byte-identity. Use a hard gate: `g_l = clamp(g_param_l, 0, 1)` with `g_param_l = 0` init ⇒ `g_l = 0` exactly.
  - Or simplest: a single hard-coded switch (`use_cos_attn: bool`), accepting that step-0 is approximately identical (not bit-identical). Cosine attention at step 0 produces near-uniform weights because all logits are `O(1/sqrt(d_k))`; baseline produces near-uniform weights because the Kaiming init makes logits `O(1)` and softmax is wide when no logit dominates. The loss at step 0 is essentially the same.
  - For implementation, prefer the hard-switch form (`use_cos_attn: bool`) and document the difference as "step-0 ≈ baseline" (init-similar, not byte-identical).
- **Config flag**: `use_cos_attn: bool = False`.
- **Param count**: **0 new params** (cosine attention is parameter-free; the temperature `tau` is a global scalar that can be hard-coded at 0.1 or learned with init 0.1).
- **Intuition (why it might lower val loss)**: the dot-product magnitude in QK^T is sensitive to the per-token Q, K norm. At d_k=16, the dot product is dominated by a few "outlier" components, leading to a sharp softmax where one token dominates. Cosine attention removes the magnitude axis entirely — all attention weights are bounded, and the gradient signal is uniformly distributed across all keys. The cosFormer paper reports consistent gains at 100M-1.5B; the question is whether the gain transfers to 0.94M where d_k=16 is small (so the cosFormer benefit is most pronounced, theoretically) but the data is tiny (so the magnitude-bounded softmax may be too conservative).

## Scale evidence
- cosFormer (Qin et al. 2022) — 100M-1.5B language modeling (WikiText-103), 100M-300M image classification (ImageNet). Direct validation of the lever form.
- QKNorm (Henry et al. 2020) — empirical study on attention logit magnitude growth; L2 normalization is the natural fix.
- **Transfer-risk: med** — the lever has strong validation at 100M+, but the *step-0* difference is non-trivial. The gated form (12 scalar gates) trades a small byte-identity violation for a cleaner init story. The 0.94M regime has small d_k=16, which is the regime cosFormer is designed for (cosFormer gains are largest when d_k is small).

## Why it's worth a slot
The bet, in one sharp sentence: **cosFormer is the only published L2-normalized attention variant with consistent 100M-1.5B gains, and 016-qk-norm (the WIN, RMS-normalization) has not been compared head-to-head with the L2 form at our tier** — at d_k=16 the L2 normalization is theoretically more aggressive than RMS (a vector of equal components has RMS=1 but L2=sqrt(d)), so if 016's WIN is driven by the *magnitude-bounded* axis then 189 should also WIN; if it's driven by some other property (symmetry, scale-invariance), 189 may NULL. A null would *attribute* the 016 WIN to RMS-specific structure rather than magnitude-boundedness; a win would give a stronger lever at small d_k.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 016-qk-norm (WIN, RMS) — *similar mechanism*, different operation. RMS normalizes per-component (each output component has unit RMS), L2 normalizes per-row (each output row has unit L2 norm). cosFormer is the L2 analog of 016.
- 162-q-only-norm, 165-k-only-norm — closed attribution of 016. 189 is *joint* L2 normalization of Q and K (not asymmetric Q-only or K-only).
- 184-logit-scale — global output scalar. 189 changes the pre-softmax *form*, not a global scale.
- 188-qk-rms-scaling — per-block scalar on QK^T scores. 189 is a *structural* change to the score formula.
