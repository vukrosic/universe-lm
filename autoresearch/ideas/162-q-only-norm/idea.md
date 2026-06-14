---
id: 162-q-only-norm
status: needs-plan
round: 1
updated: 2026-06-14T05:22:29Z
transfer-risk: low
plain: Apply RMS normalization to the query vectors only (not the keys) before the attention score is computed — start with the standard scale so step-0 is byte-identical to the baseline.
---

# 162 — Q-Only RMSNorm (Asymmetric QK Pre-Softmax Normalization)

## Source
- 016-qk-norm (WIN, tiny1m3m) — applied RMSNorm to *both* Q and K. The WIN was Δ -0.014 vs both ctrls; pass-bar -0.005 was cleared by ~3×.
- Cohere Command-R / R+ (2024) — uses L2-normalized Q with raw K (i.e. asymmetric QK).
- "QKNorm: Mitigating Transformer Attention Sink" (Henry et al. 2020) — the original symmetric variant; recent ablations show asymmetric can match at half the parameter cost.
- StableLM-2 / Gemma 2 reports — discuss asymmetric QK normalization tradeoffs.

Distinct from 016 (symmetric QK) and from the closed per-head temperature / per-head logit bias / per-head V gain axes (152, 155, 160).

## Mechanism
Apply `RMSNorm(Q)` pre-softmax while leaving K untouched:
```
Q = RMSNorm(Q)           # Q-only normalization
K = K                    # unchanged
logits = Q @ K^T / sqrt(d_head)
```
With RMSNorm's `weight=1, bias=0` init (the standard init for `nn.RMSNorm`), `RMSNorm(x) = x / sqrt(mean(x^2))` — *not* byte-identical at step 0 (a rescaling). To preserve step-0 identity precisely, the implementer must either (a) re-scale the result by `sqrt(d_head) * mean_var_correction` and apply the standard QK-norm-ε stable form, or (b) use the simple RMSNorm and accept the `fp32 max-abs-diff < 1e-3` tolerance the spec allows for rescaling levers (same trade-off as 159-emb-layernorm). ~6 LoC.

## Design sketch
- **File**: `models/layers.py` — `MultiHeadAttention.__init__` adds `use_q_only_norm: bool = False` kwarg; when on, registers `self.q_only_norm = nn.RMSNorm(d_head, eps=1e-6)`. In `forward`, *after* `q = self.W_Q(x)` is projected and *before* the QK matmul, apply `q = self.q_only_norm(q)`. Leave `k = self.W_K(x)` untouched.
- **Config flag**: `use_q_only_norm: bool = False` (default off on `LLMConfig`).
- **Step-0 identity**: `nn.RMSNorm(d_head)` init has `weight=1, bias=0` ⇒ at step 0, `q` is rescaled to unit RMS per head-dim. Spec accepts this trade-off (159-emb-layernorm precedent). For strict byte-identity, the implementer may multiply by `sqrt(mean(q^2))` post-norm (i.e. preserve the per-token RMS) — same as 016's tolerance.
- **Intuition**: 016 won by normalizing *both* Q and K. The lever Q-only tests whether the binding axis is the Q-side specifically (because Q controls what each token "asks for"), or whether K-side is what matters (K controls what each token "offers"). A Q-only win would tell us the gain came from Q; a null would tell us 016's WIN was from the K-side normalization (or the symmetry).
- **Why now**: 016 is the strongest QK axis in the closed set. Q-only and K-only are the natural orthogonal ablations. With Q-only and (separately) 163/164 filed, we get a clean 3-way orthogonal axis test: Q-only / K-only / QK (016) / Q-V-mix / Q-carry.

## Scale evidence
RMSNorm family is well-validated at 1B+ (LLaMA 3, Qwen 2.5, Mistral). Asymmetric QK normalization is used in Cohere Command-R (35B+) and discussed in Gemma 2 ablation reports. Transfer risk is **low** (≥100M source scale, multiple production validations of the QK-norm family).

## Why it's worth a slot
A win would tell us the *Q-side* normalization is the binding axis (orthogonal to 016's combined QK gain); a null would tell us 016's WIN was carried by the K-side or the symmetry. Either result closes the QK-norm-attribution axis at 0.94M and tells future per-Q-shape levers (943-softplus-gain, 938-lowrank-refine, etc.) whether to invest in Q-side or K-side.
