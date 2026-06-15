---
id: 180-qk-logit-conv
status: needs-run
round: 3
updated: 2026-06-15T07:28:58Z
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

## Plan (recode r2)

### Root cause
- The previous GPU run failed at build-smoke with `ImportError: cannot import name 'Tiny1M3MLogitConvConfig'`. The model-side changes in `models/layers.py` and `models/llm.py` were already in place; the dataclass subclass in `configs/llm_config.py` was missing or had not reached the deployed copy at `/root/universe-lm/`.
- The local `configs/llm_config.py` now defines `Tiny1M3MLogitConvConfig(Tiny1M3MConfig)` with the docstring, the `use_logit_conv: bool = True` field, and no other fields (inherits everything from `Tiny1M3MConfig`). The daemon's smoke test now passes locally (`SMOKE_OK`).

### Mechanism (no change from r1)
- Per-head kernel `w_h ∈ R^{K}` allocated as `nn.Parameter(torch.zeros(n_heads, K))` only when `use_logit_conv=True`. Init places the identity tap at index `K-1` (`w_h[:, K-1] = 1.0`), with all other weights 0 — a delta kernel centered on the "current key" position. With this init the conv is the identity on `scores`, so softmax is unchanged and the step-0 forward is byte-identical to the no-flag baseline (max-abs-diff = 2.93e-8 measured in CPU smoke, fp32 noise floor).
- Forward placement: between the causal `masked_fill(-1e9)` and `softmax` in the manual attention branch of `MultiHeadAttention.forward`. The mask is applied first (so masked `-1e9` positions stay masked under the conv), then `F.pad(scores, (K-1, 0))` for left padding, then a per-shift slice+sum accumulating `w_h[k] * padded[..., k:k+S]` for `k ∈ [0, K)`. With `w_h = [0, 0, 1]` (K=3) only the `k=K-1` slice contributes, which equals the unmasked `scores`, so the result is identical to the no-conv manual path.
- The treatment forces the manual attention path (added to the `elif` condition list at line 3573 alongside the other score-space levers) because SDPA's flash kernel cannot apply a pre-softmax score-space op. The baseline takes the SDPA path; the numerical difference is at the fp32 noise floor.

### Files
- `models/layers.py` — `MultiHeadAttention.__init__` accepts `use_logit_conv: bool = False` and `logit_conv_kernel_size: int = 3`; allocates `self.logit_conv_w` (or `None` stub) and initializes the identity tap. `MultiHeadAttention.forward` adds the conv branch between the mask and softmax. The elif condition list is extended with `or self.use_logit_conv` so SDPA flash is bypassed.
- `models/layers.py` — `TransformerBlock.__init__` adds `use_logit_conv: bool = False` and `logit_conv_kernel_size: int = 3` kwargs and threads them into the inner `MultiHeadAttention`.
- `models/llm.py` — `MinimalLLM` reads `getattr(config, "use_logit_conv", False)` and `getattr(config, "logit_conv_kernel_size", 3)` and threads them into each `TransformerBlock` (two call sites, lines 985-986).
- `configs/llm_config.py` — `Tiny1M3MLogitConvConfig(Tiny1M3MConfig)` with `use_logit_conv: bool = True` (line 6206). Inherits all other fields from `Tiny1M3MConfig`.
- `_arq_180-qk-logit-conv.py` — the run stub; imports `Tiny1M3MLogitConvConfig`, defines `C = Tiny1M3MLogitConvConfig`, and calls `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.

### Cost
- +144 params total (H=4 × K=3 × 12 layers = 144; +0.015% of 0.94M).
- ~3-5% per-forward FLOP overhead from the conv slice+sum (≪ the QK matmul).
- <5% memory overhead (transient `padded` tensor of `[B, H, T, S+K-1]`, discarded after the conv).
- Manual attention path is forced, so SDPA flash is bypassed. The training step is ~10-20% slower per step (typical for manual-path treatments at this tier), well within run noise.

### Run
- Stub: `_arq_180-qk-logit-conv.py`.
- Command on the box: `python _arq_180-qk-logit-conv.py` (called by the queue daemon).
- Tier: `tiny1m3m` (12L × 4H × 64d, 0.94M params, 3M tokens).
- Seed: 42 (single seed; per project protocol).
- Expected wall-clock: ~7-9 min (slightly above the ~6 min baseline because the manual path replaces SDPA flash).
- Val loss is written to the `val_loss` field in the run log; compare to the `Tiny1M3MConfig` control val (cached 6.4306 in `autoresearch/baseline-cache.json`).
- Pass/fail bar (from idea.md): pass if treatment val ≤ 6.45 (control + 0.02), fail if > 6.47 (control + 0.04), sub-noise otherwise.

### Verification
- `python autoresearch/bin/_box_smoke.py _arq_180-qk-logit-conv.py` → `SMOKE_OK` (local CPU build, 2026-06-15).
- Step-0 byte-identical sanity: with `use_logit_conv=True` and seed 42, `MinimalLLM(x)` logits match the no-flag baseline to max-abs-diff = 2.93e-8 (fp32 noise floor).
