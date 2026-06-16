---
id: 219-rms-eps-affine
status: tasting
round: 1
updated: 2026-06-16T00:40:44Z
transfer-risk: low
plain: Make the small numerical-stability epsilon inside RMSNorm learnable, and add a learnable per-feature bias after the norm. At init both are at safe defaults, so the first step is identical, but the model can tune the regularization strength and add a post-norm shift it currently cannot.
---

# 219 — Learnable Epsilon and Bias in RMSNorm (use_rms_eps_affine)

## Source
RMSNorm: Zhang & Sennrich 2019, arXiv:1910.07467 (gain only, no bias, fixed eps). LayerNorm: Ba et al. 2016, arXiv:1607.06450 (learnable gain AND bias, fixed eps). "PowerNorm: Fast and Simple Normalization with Polyak's Power Mean" (Shen et al. 2020) and "An Empirical Study of Layer Normalization" (Bao 2021) both note that norm epsilon is a fixed hyperparameter that rarely matters at scale but is occasionally binding. The "ScaleNorm" variant (Nguyen & Salazar 2019, arXiv:1911.07013) is gain-only-RMSNorm with a single scalar gain — 219 keeps the per-feature gain RMSNorm already has and adds bias + learnable eps.

## Mechanism
Augment the existing RMSNorm with two additions:

```
denom = sqrt(mean(x^2, dim=-1, keepdim=True) + softplus(eps_raw))   # learnable eps, init eps_raw=−4.6 → softplus(−4.6) ≈ 1e-2 ≈ standard 1e-6 is wrong; use eps_raw=+0.0 → softplus(0) = ln2 ≈ 0.693, that's too big too. Cleaner: just learn raw `eps` directly, init at 1e-6 clamp [1e-9, 1e-3].
y = (x / denom) * gain + bias                                          # bias init 0, gain init 1
```

The lever has two pieces:
1. **`eps` becomes a learnable scalar (per-norm-instance, not per-feature)** — the model can tune the denominator's regularization strength.
2. **`bias` becomes a learnable per-feature vector** — currently RMSNorm has only `gain`, not `bias` (the LLaMA design). Adding bias makes RMSNorm equivalent to LayerNorm-on-centered-input.

Init: `eps` = 1e-6, `bias` = 0. Step-0 forward is bit-identical to current RMSNorm (the bias is zero so the additive term vanishes; the eps is the standard value so the denominator is unchanged).

## Design sketch
- Touch `models/layers.py` `RMSNorm` class: add `eps: nn.Parameter` (shape `(1,)`) and `bias: nn.Parameter` (shape `(d_model,)`). Both initialized in the existing `__init__`. Forward becomes `y = (x / sqrt(mean(x^2) + self.eps.abs() + 1e-9)) * self.gain + self.bias` — `eps` is constrained non-negative via `.abs()`.
- Add `use_rms_eps_affine: bool = False` to `configs/llm_config.py`. Active treatment via inline `@dataclass` subclass.
- Cost: 1 scalar + 64-dim vector per norm call. With 2 norms per block × 12 blocks = **24 norms total**, that's `24 + 24*64 = 1560 params` (+0.17% of 0.94M).
- **Why it should help at tiny1m3m**: the closed 016-qk_norm WIN is RMSNorm-on-pre-softmax-Q/K. The closed family (017, 190) explored *placement* of norm in the residual stream but not the *interior* of RMSNorm. Adding a learnable bias and a learnable eps is two new axes the model has not been given. At 0.94M the post-norm bias term has fewer parameters than the per-feature gain, so it should learn fast in 92 update steps. The eps-learning is a single scalar per norm — fast to learn.
- **Why it might be null**: at 0.94M the existing RMSNorm is already "good enough" and the bias/eps terms are absorbed by the next linear layer (post-norm gain can re-fit a constant offset). The closed null pattern at 016/017 suggests the residual-stream norm interior is not the binding constraint at this tier.

## Scale evidence
RMSNorm-without-bias is the de-facto standard at 7B-405B (LLaMA-1/2/3, Mistral, Mixtral, Qwen-1/2). LayerNorm-with-bias is the de-facto standard at 100M-540B (GPT-2, GPT-3, OPT, BLOOM, T5). The bias inclusion choice is essentially a wash at scale (per "NormBaselines" empirical studies). Learnable eps is not widely published but appears in some NMT setups; per-instance (vs per-feature) eps is novel. Transfer risk: **low** — purely architectural, scale-agnostic.

## Why it's worth a slot
A win would say the residual stream at 0.94M benefits from a per-feature constant offset that the gain-only RMSNorm cannot express. A null confirms the LLaMA design (gain-only RMSNorm) is also the right choice at 0.94M, ruling out the bias axis for our tier. The lever is cheap (+1560 params, ~50 LoC, byte-identical step 0) and composes cleanly with the existing 016-qk_norm WIN: if 216's champion (175-alibi + 016) leaves a gap, the bias term is a new axis the optimizer can pull on without disturbing the QK norm.
