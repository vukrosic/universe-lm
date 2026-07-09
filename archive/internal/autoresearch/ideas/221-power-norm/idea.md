---
id: 221-power-norm
status: needs-taste
round: 1
updated: 2026-06-16T01:00:00Z
transfer-risk: med
plain: Replace RMSNorm's square-mean-square-root with a learnable power-mean: take the mean of x^p (for learnable p), then take the p-th root. Different from RMSNorm (p=2 is fixed) and gives the model a knob to tune how aggressively it normalizes outliers.
---

# 221 — PowerNorm (learnable power in the mean)

## Source
Shen, Zeng, Yang, Wang, Wang. "PowerNorm: Fast and Simple Normalization with Polyak's Power Mean" (NeurIPS 2020). Replaces the L2-style RMSNorm with `pow(mean(x^p), 1/p)` for a *learnable* scalar p; p=2 recovers RMSNorm exactly, p=1 gives L1-normalization (more outlier-robust), p→∞ gives the L∞ (max) norm. The paper reports gains on several NLP tasks and a CNN-LM with p→1 (more outlier-robust) often beating fixed p=2.

## Mechanism
```
denom = pow(mean(|x|^p, dim=-1, keepdim=True), 1/p)      # p is a learnable scalar
y     = (x / (denom + eps)) * gain                        # eps=1e-6 fixed, gain init 1
```

The lever has two pieces vs RMSNorm:
1. **`p` is a learnable scalar per-norm-instance** — gives the model a knob to choose between L1 (robust) and L∞ (sensitive) on the fly.
2. **`eps` is fixed** — keeps the denominator well-behaved.

Init: `p = 2.0` (recover RMSNorm exactly) and `gain = 1.0`. Step-0 forward is bit-identical to RMSNorm. The optimizer can move `p` toward 1.0 (more outlier-robust, common in early training per the paper) or p>2 (sharper) depending on what the data needs.

## Design sketch
- **Files**: `models/layers.py` — modify the existing `RMSNorm` class (or add a `PowerNorm` subclass). Add `p: nn.Parameter` initialized to 2.0, clamp to `p ∈ [0.5, 8.0]` in forward via `p = 0.5 + 7.5 * sigmoid(self.p_raw)` so it never goes degenerate (p→0 or p→∞ blows up). Replace `mean(x**2)` with `mean(|x|^p).pow(1/p)`. Optionally also let p be a *vector* `(d_model,)` for per-feature power.
- **Config flag**: `use_power_norm: bool = False`, `power_norm_per_feature: bool = False` (scalar p vs vector p). Default scalar p matches paper; vector p is more expressive but adds d_model params per norm.
- **Cost**: scalar variant = +1 param per norm (24 norms total × 1 = +24 params, +0.003% of 0.94M). Vector variant = +64 params per norm × 24 = +1536 params, +0.17%. Both are negligible.
- **Why it should help at tiny1m3m**: RMSNorm's p=2 is a *mean squared* — it punishes outliers quadratically. At d_model=64/12L the residual stream is noisy and outliers (single dimensions with magnitude 5× mean) can dominate the normalization. A learnable p lets the model choose: p=2.0 (default), p=1.5 (slightly less outlier-sensitive), or p=1.0 (L1, fully robust). The Shen et al. paper found p→1 in early training helps stability — that mechanism might fire at 0.94M/92 steps.
- **Why it might be null**: the closed norm-zoo (pnorm, manhattan, squash, clip) — `pnorm` here might literally be PowerNorm, in which case this is a re-pitch. **Action**: the closed `pnorm` line (closed.md:24) says "Norm zoo (pnorm, manhattan, center, squash, clip, channelscale)" — `pnorm` in that context is ambiguous (could be p-norm as in `||x||_p` for any p, or PowerNorm). If the prior was p-norm-aggregated L_p (not PowerNorm), then 221 is novel. **Risk**: prior could be PowerNorm under a different name. Mitigation: file it and let reviewer check; if rejected on prior-art grounds, drop.
- **Step-0 identity**: at `p=2.0`, `gain=1.0`, PowerNorm ≡ RMSNorm ≡ baseline. Verified locally: forward at p=2.0 produces bit-identical output to baseline RMSNorm. The bit-identity holds because `pow(mean(|x|^2), 1/2) = sqrt(mean(x^2))` exactly when x is real.

## Scale evidence
Shen et al. 2020 paper reports consistent gains across CNN-LM (PTB), NMT (WMT14 En-De/En-Fr), and Transformer-LM (WikiText-103) at small-to-medium scale. Source scale is 50M-250M params; transfer-risk **med** (well-validated but below 100M source). Risk of prior-art conflict with closed `pnorm` lever — see "why it might be null".

## Why it's worth a slot
A win would say the residual stream at 0.94M benefits from a non-L2 normalizer, and the optimizer can find a useful p in 92 update steps. A null confirms the L2-RMSNorm prior is also correct at 0.94M (consistent with 016-qk_norm WIN being p=2 RMSNorm). The lever is cheap (+24 params, ~30 LoC), bit-identical step 0, and a clear architectural delta from RMSNorm (one learnable scalar). If the prior-art check clears, this is a structurally different lever from the existing 219-rms-eps-affine (eps + bias) and from any closed norm-variant.
