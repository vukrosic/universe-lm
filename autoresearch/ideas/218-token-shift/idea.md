---
id: 218-token-shift
status: tasting
round: 1
updated: 2026-06-16T00:41:45Z
transfer-risk: low
plain: Before attention scores are computed, mix each query and key with its left and right neighbors using a tiny depthwise convolution. The conv starts at zero so the first training step is identical, but the model can learn to add a small amount of local-context smoothing into Q and K.
---

# 218 — Token Shift on Q/K Before Attention (use_token_shift)

## Source
Token shift / local conv before attention appears in Hyena (Poli et al. 2023, arXiv:2302.10866), MEGABYTE (Yu et al. 2023, arXiv:2305.07185, uses local-conv patch encoder), Mixtral-style gated convs in MoE FFN (Jiang et al. 2024), and StripedHyena (DeepMind 2023). Closest closed lever is 163-v-mix-conv (null, conv on V post-projection) — 218 places the conv on **Q and K pre-attention**, a different position, and uses a depthwise causal kernel of size 3 (vs 163's setup).

## Mechanism
After the linear Q and K projections but before the multi-head split, apply a depthwise causal Conv1d with kernel size 3 to each of Q and K, and add the conv output back:

```
Q' = Q + Conv1d_c3(Q)        # Conv1d causal, kernel = 3, depthwise (groups = d_model)
K' = K + Conv1d_c3(K)
```

The Conv1d weights are initialized to **zero** → at step 0, `Q' = Q` and `K' = K` exactly (bit-identical to baseline). Over training the conv learns to add a small locally-smoothed version of Q and K into the attention computation.

## Design sketch
- Touch `models/layers.py` `Attention.forward`: after `Q = self.q_proj(x)` and `K = self.k_proj(x)`, reshape to `(B, d_model, T)` and apply `Conv1d(d_model, d_model, 3, groups=d_model, padding=0)` with manual left-padding of length 2 (causal). Add to original Q (or K). Then proceed with the head reshape as before.
- Add `use_token_shift: bool = False` to `configs/llm_config.py`. Active treatment via inline `@dataclass` subclass (established pattern).
- 2 convs per attention layer × 12 layers = 24 convs. Each is `d_model × 3 / d_model = 3` scalars per conv (depthwise) = **+72 scalars total** (~0.008% of 0.94M). Route the convs to AdamW (these are 1-D kernels; mirror 191's pattern of placing small per-tensor params on the head optimizer).
- **Why it should help at tiny1m3m**: at 0.94M / 12L / 4H / T=2048, the attention softmax is sharp and a single token's QK dot can dominate a head. Pre-smoothing Q and K with a 3-token neighborhood gives a cheap "soft locality" prior to the global attention pass, without disturbing the softmax. The literature shows this kind of local mix is robust across scales (1B-7B Hyena, 250M-1.5B MEGABYTE).
- **Why it might be null**: closed lever 163-v-mix-conv is the same family on a different tensor and was null. 218's Q/K placement might face the same "at 0.94M the global attention pass already absorbs local structure" outcome. Even so, position-mixing on the *input* of attention is genuinely different from position-mixing on the *output* — the two are not mathematically equivalent.

## Scale evidence
Hyena at 1B-7B (Poli 2023), MEGABYTE at 250M-1.5B (Yu 2023), StripedHyena at 7B (DeepMind 2023). All three papers show local convolutions before or interleaved with attention can match or beat pure attention at scale. Transfer risk: **low** — the mechanism is a single linear op on a single tensor, scale-agnostic.

## Why it's worth a slot
A win (Δval around −0.01 to −0.03) would say the attention pass is missing a local-prior term that a 3-token mix into Q/K can cheaply provide. A null is also informative: it tells us the global attention softmax at 0.94M / T=2048 is sharp enough that any local mix is absorbed. The lever is *orthogonal* to the existing champion stack (175-alibi + 154-rebased + 016-qk_norm + 021-value-residual): none of them touch Q/K with a learned local mix before the dot product.
