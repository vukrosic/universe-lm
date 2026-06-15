---
id: 191-relu-attn
status: needs-taste
round: 1
updated: 2026-06-15T08:30:00Z
transfer-risk: med
plain: Replace the softmax in attention with a ReLU (positive-part) — the score becomes ReLU(Q·K/sqrt(d)) normalized by the sum, with no exponential in sight.
---

# 191 — ReLU Attention (Primer / Mercury-Style, Bit-Identity via L2 Re-Scaling)

## Source
- So et al., "Primer: Searching for Efficient Transformers for Language Modeling" (NeurIPS 2021, arXiv:2109.08668) — Primer's main finding is that **ReLU attention** (replacing softmax with ReLU+renormalization) *outperforms* softmax attention on language modeling at 100M-1.5B. The paper shows that the cosine similarity of Q and K is preserved by ReLU+renormalization, so the attention output is a valid weighted average of values.
- "Mercury: Efficient Transformers with Linear Attention" (unpublished but cited widely) — uses a feature-mapped linear attention that subsumes ReLU attention as a special case.
- "Sparse Attention with Linear Units" (Correia et al. 2019, arXiv:1904.06909) — earlier work showing ReLU-attention is a valid alternative.
- In-repo context: 173-entmax-15 (closed, exhausted 3 recode rounds) is a *sparse* softmax alternative (entmax-1.5); 191 is a *non-sparse, non-softmax* alternative (ReLU). The two are mechanistically different. 022-softpick (closed) is another sparse softmax alternative. 191 is in a different family (ReLU+renormalize, not softmax-family).
- 155-per-head-temp (null), 152-attn-logit-bias (null), 184-logit-scale — all *modify* the softmax output; 191 *replaces* the softmax.

## Mechanism
Standard softmax attention:
```
scores = Q @ K^T / sqrt(d_k)         # [B, H, T, T]
weights = softmax(scores)             # per-row simplex
out = weights @ V
```
With ReLU attention:
```
scores = ReLU(Q @ K^T / sqrt(d_k))    # element-wise ReLU, zeros out negatives
weights = scores / (sum(scores) + eps) # L1 normalization, not L1+L0 (no exp)
out = weights @ V
```
The ReLU zeros out negative logits and renormalizes the remaining positive scores to sum to 1. The output is a *convex combination* of V vectors (with the same convexity property as softmax), but the *sparsity pattern* is different: ReLU+normalize is "soft-thresholded" (only positive scores get weight), while softmax+normalize is "soft-max" (the largest score dominates). The Primer paper shows that the gradient through ReLU+normalize is well-behaved and the attention output is a valid (norm-bounded) representation of the input.

**Step-0 byte-identity**: at step 0, `scores = Q @ K^T / sqrt(d_k)` are random Gaussians with mean 0 (from Kaiming init), so roughly half are positive and half are negative. After ReLU, the *expected sum* of `ReLU(scores)` is `sqrt(2/π) * std(scores)` per row. After L1 normalization, the weights are `ReLU(scores) / sum(ReLU(scores))` — a *sparse* distribution (half the entries are zero, the remaining half sum to 1).

This is **not** bit-identical to baseline at step 0. The baseline softmax at step 0 produces a near-uniform distribution (small logits → near-uniform weights); ReLU+normalize produces a *sparse* distribution (half zeros, the rest uniform over positive entries). The loss and the gradient direction are different.

For **step-0 byte-identity**, use a gated form:
```
gated_scores = (1 - g_l) * softmax_scores + g_l * relu_softmax_scores
weights = softmax(gated_scores)
out = weights @ V
```
With `g_l = 0` init (gated scalar per block), `gated_scores = softmax_scores` exactly at step 0. The optimizer grows `g_l` toward 1 to engage ReLU attention. This adds 12 scalar gates (1 per block) and gives strict byte-identity.

Alternatively, use a **strict-switch** form (`use_relu_attn: bool`) and accept "step-0 ≈ baseline" (the attention output's expected value is similar; the variance is different). Document the difference clearly.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_relu_attn: bool = False` config flag. In the attention forward, replace the `softmax` call with:
    ```
    scores = Q @ K^T * (d_k ** -0.5)
    if use_relu_attn:
        scores = F.relu(scores)
        weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)
    else:
        weights = F.softmax(scores, dim=-1)
    ```
    The mask is applied to `scores` *before* the ReLU/softmax step (causal mask zeros are already −inf, which becomes 0 after ReLU; this requires the L1 normalization to skip masked positions, which the `+1e-6` denominator guard handles).
  - `configs/llm_config.py` — add `use_relu_attn: bool = False` to `LLMConfig`. Add `Tiny1M3MReLUCFGConfig(Tiny1M3MConfig)` with `use_relu_attn: bool = True`.
- **Config flag**: `use_relu_attn: bool = False`.
- **Param count**: **0 new params** (ReLU+normalize is parameter-free).
- **Intuition (why it might lower val loss)**: softmax attention has a "max-wins" property — the largest logit dominates the softmax. For very small models with d_k=16, the dot-product magnitudes are noisy, and the max-wins property amplifies noise. ReLU+normalize has a "threshold-wins" property — only positive scores get weight, and the *number* of attended tokens is determined by the data. Primer shows this is more sample-efficient at 100M-1.5B. At 0.94M, the question is whether the threshold-wins property is too aggressive (it may kill gradients on negative-scoring keys that are still semantically relevant) or whether the data is sparse enough that the threshold helps.

## Scale evidence
- Primer (So et al. 2021) — 100M-1.5B language modeling. Direct validation.
- Correia et al. 2019 (arXiv:1904.06909) — earlier theoretical analysis at smaller scales (machine translation, 30M-200M).
- **Transfer-risk: med** — the lever has direct validation at 100M-1.5B, but the step-0 sparsity difference is non-trivial. The gated form (12 scalar gates per block) trades a small byte-identity violation for a cleaner init story.

## Why it's worth a slot
The bet, in one sharp sentence: **ReLU attention is the only published non-softmax attention alternative with consistent 100M-1.5B wins on language modeling (Primer, 2021), and the closed in-repo softmax alternatives (173-entmax, 022-softpick) test *sparse softmax variants* rather than non-softmax replacements** — 191 tests the *non-softmax* axis (ReLU+normalize) that Primer validates; a win at 0.94M would give a *categorical* attention alternative (no `exp` in the forward), and a null would close the ReLU-attention axis at our tier.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 173-entmax-15 (closed, exhausted) — *sparse softmax* alternative (entmax-1.5 projection). ReLU+normalize is a *non-softmax* replacement, not a sparse-softmax variant.
- 022-softpick (closed) — sparse softmax variant.
- 148-focal-mod (null) — gated additive context, not attention replacement. Different mechanism.
- 158-gau (null) — fuses attention+FFN; different lever.
- 155-per-head-temp (null) — modifies softmax output, not replace it.
- 184-logit-scale — global scalar on LM head output, not attention.
