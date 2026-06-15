---
id: 180-qk-logit-conv
status: needs-run
round: 1
updated: 2026-06-15T06:00:12Z
transfer-risk: med
plain: Smooth the attention map with a tiny learnable depthwise convolution along the time axis (a soft "look locally" prior), starting with a delta-function filter so step-0 is byte-identical.
---

# 180 — Pre-Softmax 1D Causal Depthwise Conv on Attention Logits (QK^T)

## Source
- Conformer (Gulati et al. 2020, arXiv:2005.08100) sandwiches a 1D depthwise convolution between FFN and attention, but the convolution is applied to the residual stream `x`, not to attention logits. Different placement.
- ConvBERT (Jiang et al. 2020, arXiv:2008.02450) uses depthwise conv inside attention to mix Q/K vectors along the sequence axis — closer to our mechanism but applied at Q/K level, not at QK^T level.
- **Logit-Conv (the placement here)**: there is no published paper at this exact form. The lever is a smooth local-attention prior applied directly to the attention scores before softmax — the natural way to bias attention toward nearby positions without committing to a hard window.
- In-repo context: closed.md line "NSA / diff-attn / hybrid heads" covers learned hard attention patterns. 143-shortconv null/borderline is pre-attention conv on x. 151-rov-gated null is intra-layer rotary on V. 152/155/160/166 closed null are per-head scalar levers on softmax (NOT a structural smoothing). The placement "pre-softmax conv on QK^T" is novel.

## Mechanism
Standard attention:
1. `scores[b, h, t, s] = Q[b, h, t, :] · K[b, h, s, :] / √d_k` — shape `[B, H, T_q, T_k]`.
2. `attn[b, h, t, s] = softmax(scores[b, h, t, :], dim=-1)`.
3. `out[b, h, t, :] = attn[b, h, t, :] @ V[b, h, :, :]`.

With logit-conv (per head h, kernel size K=3, causal):
1. Same as step 1.
2. **For each (b, h, t)**, convolve scores along the s axis with a per-head kernel `w_h ∈ ℝ^3`:
   `scores_conv[b, h, t, s] = sum_{k=0..2} w_h[k] · scores[b, h, t, s-k]` (with s-k clipped to ≥ 0 for causal).
3. `attn[b, h, t, s] = softmax(scores_conv[b, h, t, :], dim=-1)`.
4. Same as step 3.

Bit-identity at step 0: `w_h = [0, 0, 1]` (delta function) ⇒ `scores_conv[b, h, t, s] = scores[b, h, t, s]` for all s ⇒ softmax unchanged ⇒ **byte-identical to baseline at step 0**.

The lever: `w_h[0] = -w_pos_raw_h`, `w_h[1] = 1 + w_diag_raw_h`, `w_h[2] = w_neg_raw_h` (or some parameterization that includes the identity init). Optimizer can grow any kernel shape — including smoothing (all-positive, sums to 1) or sharpening (negative weights).

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_logit_conv: bool = False` and `logit_conv_kernel_size: int = 3` to `MultiHeadAttention.__init__`. Allocate `self.logit_conv_w = nn.Parameter(torch.zeros(n_heads, 3))` (init zero). After computing scores but before softmax, apply per-head 1D causal conv along the s axis. Use `F.conv1d` with explicit padding: reshape scores to `[B*H, 1, T_k]`, conv with per-head weight (use grouped conv with H groups), reshape back. Or use a manual loop over H. Set the center weight of `self.logit_conv_w` to 1 at init via a fixed assignment: `self.logit_conv_w[:, kernel_size//2] = 1.0` after the zero init.
  - `configs/llm_config.py` — add `use_logit_conv: bool = False`. Add `Tiny1M3MLogitConvConfig` subclass with `use_logit_conv: bool = True`.
  - `models/llm.py` — thread into both `TransformerBlock` sites.
- **Config flag**: `use_logit_conv: bool = False`.
- **Step-0 byte-identical**: `w_h[:, 1] = 1` (center), `w_h[:, 0] = w_h[:, 2] = 0` ⇒ conv is delta ⇒ scores unchanged ⇒ softmax unchanged ⇒ **byte-identical to baseline at step 0 (max-abs-diff = 0.0)**.
- **Param count**: H=4, kernel_size=3, n_layers=12. Per block: 4·3 = 12 params. Total: 144 params (+0.015% of 0.94M).
- **Intuition (why it might lower val loss)**: at 0.94M/12L/4H, attention scores are noisy due to limited gradient signal. A soft local-attention prior (smoothing kernel) could help the model commit to nearby-token attention without hard-windowing (which the SWA sweep already showed helps but doesn't generalize). The delta-function init gives the optimizer a smooth path to grow into a smoothing kernel. Different from 143-shortconv (which convolves on x BEFORE QKV projection; 180 convolves on QK^T AFTER projection, directly smoothing the attention pattern).

## Scale evidence
- Conformer at 100M+ (ASR, audio) — conv-on-x placement is well-validated.
- ConvBERT at 100M (BERT-class) — conv-on-Q/K placement, validated.
- Logit-conv placement (this filing) has no direct published validation. The mechanism is a natural extension of the Conformer/ConvBERT family to the attention score space. Transfer-risk is **med** (placement is novel at 100M+ but the underlying locality-prior concept is well-validated).

## Why it's worth a slot
The bet, in one sharp sentence: **the closed shortconv null (143) tested conv on x with a gate=0 init that bit-locks the conv out at step 0 and only activates through a scalar gate — 180 tests conv directly on the attention logits with a delta-init (the conv is present and active but identity at step 0), giving the optimizer direct access to a per-head smoothing kernel without an intermediate gate absorbing the gradient.** A null at 0.94M would close the logit-smoothing axis and confirm that attention at our tier is robust to score-space smoothing; a win would unlock a fresh placement (logit-level conv) for Phase-2 ≥135M where the per-head smoothing kernel has more gradient signal to develop.
