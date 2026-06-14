---
id: 163-v-mix-conv
status: needs-run
round: 2
updated: 2026-06-14T05:56:01Z
transfer-risk: low
plain: After the attention output is computed, smooth the value vectors across nearby tokens with a tiny sliding-window convolution — start with an identity filter so step-0 matches the baseline exactly.
---

# 163 — Post-Attention V-Mix Depthwise Convolution

## Source
- Poli et al. "Hyena: Toward Large Convolutional Language Models" (2023) — uses gated convolutions on V before the output projection. Validated at 1B+ scale (Striped Hyena 7B).
- Mehta et al. "Long Range Arena" benchmark work — depthwise convs on V are a known locality prior.
- FNet (Lee-Thorp et al. 2021) — FFT token mixing as alternative to attention, but the *conv-on-V residual* form is a distinct lever from FNet's full replacement.

Distinct from 143-shortconv (depthwise conv *pre-attention*, null at borderline) and 157-conv-ffn (depthwise conv *post-FFN-activation*, null). 163 is the third axis: depthwise conv *post-attention, on V before the O-projection*.

## Mechanism
After `attn_output = softmax(QK^T) V` is computed and reshaped to `[B, H, T, d_head]`, apply a depthwise `Conv1d(kernel_size=k, groups=d_model)` over the sequence axis on `attn_output`. The conv mixes each token's V-output with its k-1 neighbors *before* the W_O projection:
```
attn_output = attn_output.transpose(1, 2)          # [B, T, H, d_head]
attn_output = attn_output.reshape(B, T, d_model)
attn_output = depthwise_conv1d(attn_output)         # mix across T
attn_output = self.W_O(attn_output)
```
Init the conv weight to `[[0, 1, 0]]` per channel (center-tap identity) ⇒ conv is a strict identity at step 0. With `k=3`, each token mixes with its left and right neighbors; ~12 LoC; ~2.3k extra params at tiny1m3m (+0.25% of 0.94M).

## Design sketch
- **File**: `models/layers.py` — `MultiHeadAttention.__init__` adds `use_v_mix_conv: bool = False` and `v_mix_conv_kernel: int = 3` kwargs. When on, registers `self.v_mix_conv = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, padding=1, bias=False)`, then in `forward`, *after* `attn_output` is computed (post-SDPA, post-reshape, pre-W_O), apply `attn_output = self.v_mix_conv(attn_output.transpose(1, 2)).transpose(1, 2)`.
- **Config flag**: `use_v_mix_conv: bool = False`, `v_mix_conv_kernel: int = 3` (default).
- **Step-0 identity**: init `self.v_mix_conv.weight.data.zero_()` then set `self.v_mix_conv.weight.data[:, 0, 1] = 1.0` (center-tap identity). This makes the conv a strict identity at step 0, byte-identical to baseline. Same trick as 157.
- **Intuition**: tests whether the locality prior belongs in V (post-attention smoothing) vs Q/K (143 pre-attn) vs FFN (157 post-act). 143 closed at borderline-WIN-rule (mechanism suggestive, ctrl variance inflated the gap); 157 was null (FFN-internal locality not binding). V-mix is the third axis: does attention's *output* need local smoothing?
- **Why now**: the 3-axis locality test (pre-attn / post-act / post-attn) is incomplete without 163. A WIN here would localize the gain to V-axis and tell us the binding axis is the attention output, not the attention input or the FFN output. A null would close the post-attention locality axis at 0.94M.

## Scale evidence
Striped Hyena 7B (Poli et al. 2023) — gated conv on V at 7B scale. Hyena hierarchy validated at 1B+ (sub-quadratic long-context). Transfer risk is **low** (≥100M source scale, Hyena is a published / reproducible architecture).

## Why it's worth a slot
A win would tell us the *attention output* (not the attention input, as 143 tested, and not the FFN output, as 157 tested) is the binding locality axis at 0.94M, completing the 3-axis locality test. A null would close the post-attention locality axis alongside the closed pre-attention and post-FFN axes.

## Plan
- **Files**: `models/layers.py`, `configs/llm_config.py`, `_arq_163-v-mix-conv.py`.
- **Config flag**: `use_v_mix_conv: bool = False` (default off on `LLMConfig`); `v_mix_conv_kernel: int = 3` (odd int ≥ 3).
- **Config class**: `Tiny1M3MVMixConvConfig(Tiny1M3MConfig)` with `use_v_mix_conv=True, v_mix_conv_kernel=3`.
- **Step-0 identity**: conv weight init = `[0, 1, 0]` per channel ⇒ strict identity at step 0. Construction via direct `.data` assignment (no RNG advance) keeps RNG state aligned with the no-flag path.
- **Run command**: `/venv/main/bin/python /root/universe-lm/_arq_163-v-mix-conv.py`; compare against baseline cache.
- **Cost**: 12 layers × 3 × 64 = 2,304 extra params (+0.25% of 0.94M); forward: one extra depthwise conv per block per step.
