---
id: 192-topk-attn
status: needs-plan
round: 1
updated: 2026-06-15T12:03:26Z
transfer-risk: med
plain: Hard top-k sparse attention — keep only the k largest pre-softmax scores per row, zero the rest, renormalize. Default k=512 (T=2048, 75% sparsity). 0 new params. Step-0 non-identical (structural lever, same category as 173-entmax / 022-softpick).
---

# 192 — Top-K Sparse Attention (Hard Structural Sparsity, k=T/4)

## Source
- Touvron et al. "Going Deeper with Image Transformers" (2021) — top-k attention in vision transformers, validated at 100M-300M on ImageNet. The lever is the *attention-mask form* of top-k (pre-softmax, per-row hard sparsification).
- Martins & Astudillo, "From Softmax to Sparsemax" (ICML 2016, arXiv:1602.02068) — sparsemax projects to the simplex (learned support size). Top-k is the *fixed-support* cousin: support size is a hyperparameter, not learned. Closed in-repo analog: 173-entmax-15 (learned support, exhausted recode cap).
- Closed/won in-repo priors that 192 must engage:
  - 173-entmax-15 (closed, recode cap) — *learned* sparse softmax. 192 is *fixed*-support hard top-k.
  - 022-softpick (closed) — sparse softmax via ReLU(exp(x)−1). 192 is hard sort, not soft.
  - 182-per-head-window (null) — *windowed* attention (contiguous block). 192 is *score-sorted* (non-contiguous).
  - 154-rebased-attn (WIN) — *soft locality* prior (avg-pool K/V before softmax). 192 is *hard global* sparsity. **Opposite priors on the same axis.**
  - 177-talking-heads (DRIFT, Δ=+0.9509) — H×H *output* mixing. 192 is *input-side* hard sparsification, not output mixing.

## Mechanism (COMMITTED: option (a) — hard-fixed top-k only)

Standard softmax attention:
```
scores = Q @ K^T / sqrt(d_k)         # [B, H, T, T]
weights = softmax(scores)             # all T positions get non-zero weight
out = weights @ V
```

With hard top-k sparse attention (k=T/4=512 for T=2048, 75% sparsity):
```
scores = Q @ K^T / sqrt(d_k)         # [B, H, T, T]
topk_vals, topk_idx = scores.topk(k, dim=-1)              # [B, H, T, k]
sparse_scores = torch.full_like(scores, float("-inf"))
sparse_scores.scatter_(-1, topk_idx, topk_vals)           # -inf everywhere else
weights = softmax(sparse_scores, dim=-1)                  # renorm over k keys
out = weights @ V
```

**Why option (a), not the learnable-`k_l` alternative**: the taste reviewer flagged that bundling both is two levers, not one. Pick one. Option (a) (hard-fixed) is the cleaner null-info result: it tests "is *fixed* structural sparsity useful at 0.94M," with no Learn-vs-Hard confound. The learnable-`k_l` variant is a different A/B (smooth per-block learn schedule) and would have its own plan; it's *not* in this pitch.

**Param count**: 0 new params. **Step-0 identity**: NOT byte-identical to baseline (different key set is attended to). This is a *structural lever*, same category as 173-entmax / 022-softpick / 154-rebased. Frame it as such in the recode plan.

**Why it should bind at 0.94M where 173 didn't — mechanism (a) from taste findings**: the lever's mechanism is **strictly more restrictive than 173's, with strictly cleaner gradient flow**:
- 173-entmax-1.5 searches a 2-D surface: per-row support size (via bisection on a Lagrange multiplier) AND the per-token weights. The Lagrange multiplier flickers near the support boundary (the well-known entmax-1.5 gradient noise problem — Peters et al. 2019 document this), so at 0.94M the optimizer is fighting two coupled moving targets.
- 192 (option (a)) eliminates the support-size dimension entirely. The k=512 budget is fixed. The optimizer only has to learn the per-token weights *within* the fixed top-k. The gradient through the top-k boundary is still discontinuous, but it's a *single* discontinuity (membership in the top-k set), not a *coupled pair* (support size + weight).
- The bet: at 92-step tiny1m3m, dropping the support-size dimension removes one source of variance in the search, and 92 steps × 0.94M is exactly the regime where 1-D search beats 2-D search. This is the *only* way the lever can win: 192 wins by *being a more restricted problem* than 173, not by *adding* a new prior.

**Engagement with 154-rebased (competing, not complementary)**: 154-rebased WIN says *locality* prior helps. 192 is the *opposite* prior — global, non-contiguous, top-k. The two levers test opposite ends of the "what kind of sparsity helps?" axis:
- 154: "soft locality, K/V averaged into 256 rebasins" → captures positional continuity
- 192: "hard global, pick 512 of 2048 by score" → captures non-contiguous relevance

If 154 wins and 192 wins, *both* sparsities help and a combined lever (e.g., top-k within a rebased window) is the next step. If 154 wins and 192 nulls, the prior is "locality-specific" not "sparsity-generic." If both null, the sparsity axis is closed at this tier. **They are competing for the same axis-slot**, not stacking.

**Engagement with 177-talking-heads (DRIFT +0.95)**: 177 mixes attention *output* H×H with soft weights, allowing one head to flip another head's mass. At d_k=16 the H=6 head count and small per-head dim make soft H×H mixing structurally hostile (one head's signal swamps another's). 192 is structurally different: it operates *pre-softmax* on the scores, doesn't cross heads (topk is per-row within a head), and is *hard* (binary mask) not *soft* (continuous mixing). The d_k=16 hostility of 177 does not transfer to 192 — the levers share an attention-score modification but not the head-cross-mixing dynamic that caused 177's blowup.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_topk_attn: bool = False` and `topk_k: int = 512` config flags. In the attention forward, after computing `scores = Q @ K^T / sqrt(d_k)` and applying the causal mask:
    ```python
    if self.use_topk_attn:
        k = min(self.topk_k, scores.size(-1))
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores.scatter_(-1, topk_idx, topk_vals)
        weights = F.softmax(sparse_scores, dim=-1)
    else:
        weights = F.softmax(scores, dim=-1)
    ```
  - `configs/llm_config.py` — add `use_topk_attn: bool = False` and `topk_k: int = 512` to `LLMConfig`. Add `Tiny1M3MTopKAttnConfig(Tiny1M3MConfig)` with `use_topk_attn: bool = True, topk_k: int = 512`.
- **Config flag**: `use_topk_attn: bool = False, topk_k: int = 512`.
- **Param count**: **0 new params**.
- **Causal-mask interaction**: the scatter must be applied *after* the causal mask is added (causal positions outside the visible past should be `-inf` before topk so the topk never picks a future token). Implementation: topk on the already-masked scores. Verified mental check: with `scores = -inf` for future positions, `scores.topk` will never select them.

## Scale evidence
- Top-k attention (Touvron et al. 2021) — validated at 100M-300M on ImageNet. transfer-risk: **med** (the *forced sparsity ratio* matters at scale — k=512/T=2048 is 75% sparsity, but k=512/T=8192 is 94%; sweet spot may shift at 135M where T grows. Flag for 135M-stage re-test of `k ∈ {256, 512, 1024}` keeping ratio constant, but do not lock in now).
- Sparsemax (Martins & Astudillo 2016) — validated at 30M-100M on multi-label classification (different task, related mechanism).
- In-repo at tiny1m3m: closed sparse-softmax family (173, 022), null windowed (182), win locality-prior (154), drift H×H-mixing (177). 192 is the *hard-fixed-support* variant — not yet tested at this tier.

## Why it's worth a slot
The bet, in one sharp sentence: **hard-fixed top-k attention is the *strictly more restricted, strictly cleaner-gradient* cousin of the already-closed 173-entmax-1.5, and the *opposite* prior of the 154-rebased WIN — at 0.94M a 1-D search over weights within a fixed 512-key budget should beat 173's coupled 2-D search (support + weights), and the result distinguishes "sparsity-generic" from "locality-specific" priors on the same axis that 154 won**.

**Predicted magnitude with mechanism (COMMITTED)**:
- **Primary prediction: NULL (|Δval| < 0.01).** The mechanism is: 173 already closed the *learned*-support axis at 0.94M with 3 recode rounds. 192's only advantage over 173 is dropping the support-size dimension, but that drops *expressivity* too (173 could in principle pick support size 50 or 1500; 192 is pinned at 512). On a 92-step / d_k=16 / 12L regime the expressivity loss probably balances the gradient-cleanliness gain. Mechanism: 173's 2-D search was the wrong tool for 0.94M; 192's 1-D search is *less wrong* but still not the right tool.
- **Long-shot WIN (Δval ∈ [-0.005, -0.015]).** Mechanism: 154-rebased WIN says locality prior helps; top-k as a *strict* structural prior (forced 512-budget) plus the *cleaner gradient* over 173 both fire. Magnitude bounded by the fixed-k rigidity — it cannot beat 154's WIN by more than ~half because 192 is a *harder* constraint than 154's soft pool.
- **DRIFT risk (Δval ∈ [+0.01, +0.05]).** Mechanism: d_k=16 / H=6 with a forced 75% sparsity means each head has 512 of 2048 keys to work with, and at 12L the model may not be able to learn the *right* per-head 512 selection in 92 steps. Bound by the same d_k=16 that 177-talking-heads hit (but 192 doesn't cross heads, so DRIFT is bounded, not catastrophic like 177's +0.95).

**Primary prediction = NULL.** The slot is justified by the clean axis-disambiguation against 154 (locality vs global hard sparsity) and the *strict* relationship to 173 (which already nulled). A null confirms 173's null on a different parameterization (1-D vs 2-D) — strictly more informative than re-running 173.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24 (175-alibi), cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 173-entmax-15 (closed, recode cap) — *learned* sparse softmax. 192 is *fixed*-support hard top-k. **192 is the strictly-more-restricted cousin of 173**; a 192 null confirms 173's null on a different parameterization (1-D vs 2-D search).
- 022-softpick (closed) — sparse softmax via ReLU(exp(x)−1). 192 is hard sort, not soft.
- 182-per-head-window (null) — *windowed* attention (contiguous block). 192 is *score-sorted* (non-contiguous).
- 154-rebased-attn (WIN) — *soft locality* prior. 192 is *hard global* sparsity. **Opposite priors, same axis — competing for the axis-slot, not stacking.**
- 177-talking-heads (DRIFT) — H×H *output* mixing, soft, cross-head. 192 is *input-side* hard sparsification, per-head, no cross-head mixing. d_k=16 hostility does not transfer.
- 148-focal-mod (null) — gated additive context, not attention replacement.
- BigBird (closed lever family) — different sparsity pattern (random + window + global).
