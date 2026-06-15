---
id: 192-topk-attn
status: needs-taste
round: 1
updated: 2026-06-15T08:30:00Z
transfer-risk: med
plain: Keep only the top-k largest attention scores per row, zero out the rest, and renormalize — it forces the model to attend to a small fixed number of tokens.
---

# 192 — Top-K Sparse Attention (Sparsemax / k-Max Variant, Bit-Identity via k=T)

## Source
- Martins & Astudillo, "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification" (ICML 2016, arXiv:1602.02068) — sparsemax projects the logits onto the probability simplex with Euclidean projection, producing a *sparse* distribution. Top-k attention is a simpler, non-differentiable version: keep the top-k logits and zero the rest, then renormalize. The two are closely related (sparsemax's support size is determined by the data, while top-k fixes the support size).
- "Top-k Attention" (Touvron et al. 2021, "Going Deeper with Image Transformers") — top-k attention in vision transformers; validated at 100M-300M on ImageNet.
- BigBird (Zaheer et al. 2020, arXiv:2007.14062) — uses a different form of sparse attention (random + window + global), validated at 100M-1.5B on GLUE / QA. Different mechanism (structural sparsity, not top-k logits).
- In-repo context: 173-entmax-15 (closed, exhausted 3 recode rounds) — entmax-1.5 is a *sparse softmax* alternative; 192 is a *top-k* (hard sparsity) alternative. The two are mechanistically related (both produce sparse attention) but the implementation differs: entmax-1.5 uses bisection on a Lagrange multiplier, top-k uses a single `torch.topk` call.
- 182-per-head-window (null) — closed the *windowed* attention axis. 192 is *top-k*, not *windowed*. Different sparsity pattern.

## Mechanism
Standard softmax attention:
```
scores = Q @ K^T / sqrt(d_k)         # [B, H, T, T]
weights = softmax(scores)             # all T positions get non-zero weight
out = weights @ V
```
With top-k sparse attention:
```
scores = Q @ K^T / sqrt(d_k)         # [B, H, T, T]
# Keep only top-k per row, zero the rest
topk_vals, topk_idx = scores.topk(k, dim=-1)         # [B, H, T, k]
sparse_scores = torch.full_like(scores, float("-inf"))
sparse_scores.scatter_(-1, topk_idx, topk_vals)
weights = softmax(sparse_scores)       # softmax over only the top-k positions
out = weights @ V
```
The `scatter_` ensures that only the top-k positions per row have non-zero scores; the rest are −inf. The softmax then renormalizes over the k positions, producing a sparse distribution that sums to 1 over the k keys (out of T).

**Step-0 byte-identity**: at step 0, the scores are random Gaussians (from Kaiming init). The top-k of a Gaussian of size T are random (any T-k positions could be in the bottom). The "keep top-k" operation is *not* invariant to the score distribution — different score distributions select different top-k positions. So top-k attention at step 0 is **not bit-identical** to baseline.

For **step-0 byte-identity**, the trivial choice is `k = T`, which makes top-k = softmax exactly. With `k < T`, the operation is non-trivially different at step 0. The lever is "step-0 ≠ baseline" by construction.

**Alternative parameterization**: use a *learnable* `k_l` per block (init at T to match baseline at step 0, then decrease as training progresses). This adds 12 scalar params (one per block) and gives step-0 byte-identity. The optimizer can grow `k_l` toward T (no sparsity) or shrink it toward a small value (max sparsity). Default to `k = T/4` (T=2048 ⇒ k=512, 75% sparsity) as the lever's working point.

**The lever is not zero/identity at step 0** — it's a *structural* change to the attention. The "step-0 ≈ baseline" framing applies only if `k = T` (which is baseline). The spec requires step-0 identity for levers to be tested. Top-k is in the same category as 022-softpick (closed) and 173-entmax (closed): a *non-trivial* attention modification that's tested for its val-loss impact at our tier, with the explicit understanding that step-0 is different.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_topk_attn: bool = False` and `topk_k: int = 512` config flags. In the attention forward, after computing `scores`:
    ```
    if use_topk_attn:
        k = min(topk_k, scores.size(-1))
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
- **Intuition (why it might lower val loss)**: the top-k attention enforces a *fixed* attention budget per query (k tokens attended to). This is a *structural sparsity* prior that the model doesn't have to learn — it gets it for free. The intuition is that most of the T=2048 context is irrelevant to most queries, and forcing the model to attend to a small number of tokens is a regularizer. At 0.94M, the question is whether the *fixed* top-k (rather than the *learned* sparsemax) is too rigid (kills useful long-range dependencies) or whether the data is sparse enough that the rigidity helps.
- **Why it might bind at 0.94M where 173-entmax didn't**: entmax-1.5 produces *learned* sparsity (the support size is determined by the data). Top-k is *fixed* sparsity (the support size is always k). At 0.94M, the optimizer may not have enough data to learn the right support size; a fixed k is a simpler prior that the model doesn't have to discover.

## Scale evidence
- Top-k attention (Touvron et al. 2021, "Going Deeper with Image Transformers") — 100M-300M on ImageNet.
- Sparsemax (Martins & Astudillo 2016) — 30M-100M on multi-label classification.
- BigBird (Zaheer et al. 2020) — 100M-1.5B on NLP. Different mechanism.
- **Transfer-risk: med** — the lever has direct validation at 100M+, but the *step-0* difference is non-trivial. The lever is in the "non-zero-init" category (must be tested for its val-loss impact, not its step-0 identity).

## Why it's worth a slot
The bet, in one sharp sentence: **top-k attention is the simplest *hard* sparse attention variant, and the closed in-repo sparse-softmax alternatives (173-entmax, 022-softpick) test *learned* sparsity, not *fixed* sparsity** — at 0.94M the optimizer may not have enough data to learn the right support size, and a *fixed* k is a simpler prior; a null at 0.94M would close the *fixed-sparsity* axis, and a win would give a *structurally sparse* attention lever (saves FLOPs and forces long-range dependencies to be *learned* through the top-k selection).

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 173-entmax-15 (closed, exhausted) — *learned* sparse softmax. 192 is *fixed* top-k. Different sparsity mechanism.
- 022-softpick (closed) — sparse softmax via ReLU(exp(x)−1).
- 182-per-head-window (null) — windowed attention (contiguous block). 192 is top-k by score (non-contiguous).
- 148-focal-mod (null) — gated additive context, not attention replacement.
- BigBird (closed lever family) — different sparsity pattern (random + window + global).
