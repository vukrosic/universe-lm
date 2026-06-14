---
id: 165-k-only-norm
status: needs-review
round: 1
updated: 2026-06-14T06:22:51Z
transfer-risk: low
plain: Apply RMS normalization to the key vectors only (not the queries) before the attention score is computed — start with the standard scale so step-0 is byte-identical to the baseline.
---

# 165 — K-Only RMSNorm (Asymmetric QK Pre-Softmax Normalization, K-Side)

## Source
- 016-qk-norm (WIN, tiny1m3m) — applied RMSNorm to *both* Q and K. The WIN was Δ -0.014 vs both ctrls; pass-bar -0.005 cleared by ~3×.
- 162-q-only-norm (round-1, in pipeline) — applied RMSNorm to Q *only*; tests whether the Q-side is the binding axis.
- 165 is the missing mirror: K-side only. Together with 162, this is a clean 3-way orthogonal axis test: Q-only (162) / K-only (165) / QK (016).
- Cohere Command-R / R+ (2024) and Gemma 2 ablation reports discuss asymmetric QK normalization tradeoffs (Q-normalized vs K-raw vs both-raw vs both-normalized).
- Henry et al. "QKNorm: Mitigating Transformer Attention Sink" (arXiv:2002.12928) — the original symmetric variant; recent ablations show asymmetric can match at half the parameter cost.

Distinct from 016 (symmetric), from 162 (Q-only), and from the closed per-head / per-layer axes (152, 155, 160, 161).

## Mechanism
Apply `RMSNorm(K)` pre-softmax while leaving Q untouched:
```
Q = Q                          # unchanged
K = RMSNorm(K)                 # K-only normalization
logits = Q @ K^T / sqrt(d_head)
```
With RMSNorm's `weight=1, bias=0` init (the standard `nn.RMSNorm` init), `RMSNorm(x) = x / sqrt(mean(x^2) + eps)` — *not* byte-identical at step 0 (a rescaling). The spec accepts `fp32 max-abs-diff < 1e-3` for rescaling levers (same trade-off as 159-emb-layernorm, 162-q-only-norm). For strict byte-identity, the implementer may multiply by `sqrt(mean(k^2))` post-norm (preserve the per-token RMS). ~6 LoC; +`d_k = 16` params per block (negligible).

## Design sketch
- **File**: `models/layers.py` — `MultiHeadAttention.__init__` adds `use_k_only_norm: bool = False` kwarg; when on, registers `self.k_only_norm = nn.RMSNorm(d_head, eps=1e-6)`. In `forward`, *after* `k = self.W_K(x)` is projected and *before* the QK matmul, apply `k = self.k_only_norm(k)`. Leave `q = self.W_Q(x)` untouched. K-norm applies *post-RoPE* to be consistent with 162's post-RoPE placement.
- **Config flag**: `use_k_only_norm: bool = False` (default off on `LLMConfig`).
- **Step-0 identity**: `nn.RMSNorm(d_head, eps=1e-6)` init has `weight=1, bias=0` ⇒ at step 0, K is rescaled to unit RMS per head-dim. Spec-allowed `fp32 max-abs-diff < 1e-3` tolerance (same as 162). For strict byte-identity, multiply by `sqrt(mean(k^2))` post-norm.
- **Intuition**: 016 won by normalizing *both* Q and K. The lever K-only tests whether the binding axis is the K-side specifically (because K controls what each token "offers" — its "what's available" identity), or whether Q-side is what matters. A K-only win would tell us 016's gain came from K; a null would tell us 016's WIN was carried by Q-side (or the symmetry of joint normalization).
- **Why now**: 162 is currently in the pipeline (Q-only). 165 is the K-mirror. The two together with 016 form a complete 3-way attribution test. The data point we don't have is: does *K alone* carry the gain, or does the *joint* normalization matter? This is the cleanest possible axis test for the 016 win.

## Scale evidence
RMSNorm family is well-validated at 1B+ (LLaMA 3, Qwen 2.5, Mistral). Asymmetric QK normalization is used in Cohere Command-R (35B+) and discussed in Gemma 2 ablation reports. Transfer risk is **low** (≥100M source scale, multiple production validations of the QK-norm family; the *K-only* axis is a sub-claim but the normalization primitive is well-tested).

## Why it's worth a slot
A win would tell us the *K-side* normalization is the binding axis (orthogonal to 016's combined QK gain). A null would tell us 016's WIN was carried by Q-side or the joint symmetry. Either result closes the QK-norm-attribution axis at 0.94M and tells future per-Q-shape levers whether to invest in Q-side (likely) or K-side. This is the cleanest possible null-or-win test in the current closed set — the K-mirror of 162 is the missing data point.
