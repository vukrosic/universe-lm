---
id: 157-conv-ffn
status: done
round: 1
updated: 2026-06-14T01:19:02Z
transfer-risk: low
plain: Add a tiny sliding-window convolution inside the feed-forward block so each token can mix information with its immediate neighbors — different from the closed short-conv lever which sits *before* the attention block.
---

# 157 — Depthwise Conv inside FFN (Post-Activation)

## Source
- Jiang et al. "ConvBERT: Improving BERT with Span-based Dynamic Convolution" (2020, arXiv:2008.02496) — depthwise conv inside FFN.
- Mehta et al. "MobileViG / Mobile-FFN" family (2023) — depthwise conv in FFN for parameter efficiency.
- Woo et al. "ConvNeXt" FFN design (2020) — depthwise conv sandwich.

## Mechanism
Inside the FFN block (after the up-projection + activation), insert a depthwise `Conv1d(kernel_size=k, groups=d_model)` over the sequence axis. With `k=3`, each position mixes with its left and right neighbors after the activation. Init the conv weights to `[0, 1, 0]` (centered identity) so the conv is a no-op at step 0. ~20 LoC.

## Design sketch
- **File**: `models/layers.py` — add a `DepthwiseConv1d(k=3)` layer inside the FFN block, applied after the activation, before the down-projection.
- **Config flag**: `use_conv_ffn: bool`, `conv_ffn_kernel: int = 3` (default).
- **Step-0 identity**: init conv weight to `[[0, 1, 0]] * d_model` (broadcast over channels), making the depthwise conv a strict identity at step 0. `fp32 max-abs-diff < 1e-7` if the input dtype is fp32.
- **Intuition**: gives the FFN a free local-mixing step at zero parameter cost. Different from 143 shortconv (closed null), which was *pre-attention* depthwise conv; this is *inside FFN post-activation*. The 143 null closed the "attention-input locality prior" axis; this tests whether "FFN-output locality prior" is the binding lever.
- **Why now**: 143 closed at borderline-WIN-rule (mechanism suggestive, ctrl variance inflated the gap); an FFN-internal conv isolates whether the locality prior is the missing piece at 0.94M or whether 143's borderline was just ctrl noise.

## Scale evidence
ConvBERT (110M BERT), ConvNeXt (any scale). Transfer risk is **low** (≥100M source scale, multiple replications).

## Why it's worth a slot
A win would tell us the FFN *output* (not the attention input, as 143 tested) is the binding locality axis at 0.94M; a null would close the post-attention-locality axis alongside 143.

## Plan

### Files to change
- **NEW** `models/conv_ffn.py` — `ConvFFN` module: identity-init symmetric depthwise Conv1d (raw `Parameter` to avoid RNG advance on construction). ~85 LoC.
- `models/layers.py` — add `use_conv_ffn`, `conv_ffn_kernel` kwargs to `TransformerBlock.__init__`; lazy build of `self.conv_ffn`; apply `ff_out = self.conv_ffn(ff_out)` after every `ff_out = self.feed_forward(...)` call site (parallel / post-norm / pre-norm branches). +~50 LoC.
- `models/llm.py` — read `use_conv_ffn`/`conv_ffn_kernel` from config; pass through to both `TransformerBlock` and `YOCOLlamaBlock` constructions. +~15 LoC.
- `configs/llm_config.py` — add `use_conv_ffn: bool = False`, `conv_ffn_kernel: int = 3` to `LLMConfig`; add `Tiny1M3MConvFFNConfig` (flag on, kernel=3). +~30 LoC.

### Flag name & defaults
- `use_conv_ffn: bool` (default **False**) — turns the lever on.
- `conv_ffn_kernel: int = 3` — odd integer ≥ 3 (spec pin: 3).

### Step-0 byte-identity (the binding constraint)
- Conv weight tensor is `nn.Parameter(zeros(d_model, 1, k))` with center tap = 1.0 ⇒ the conv is a strict identity function `y = x` for any input at step 0.
- Construction is via raw `Parameter` (NOT `nn.Conv1d`) so the construction does **not** consume RNG — keeps the RNG state aligned with the baseline path for byte-identity. Same trick as 156-moa's `moa_extra_kv`.
- Verified: `MinimalLLM(use_conv_ffn=True)` and `MinimalLLM(use_conv_ffn=False)` with the same seed (42) produce `max_abs_diff = 0.0` on a 2×32 fp32 forward.

### Run command
The flag is read by `MinimalLLM.__init__` via `getattr(config, "use_conv_ffn", False)`. The new `Tiny1M3MConvFFNConfig` (at `configs/llm_config.py:~1825`) sets the flag to `True` and can be selected by the runner harness the same way as `Tiny1M3MShortConvConfig`. Read final val loss from the runner's `eval_milestones` array (the same one as `Tiny1M3MConfig`, see `eval_milestones` field).

### Cost summary
- 12 layers × 3 × 64 = 2,304 extra params at tiny1m3m (+0.25% of the 0.94M model).
- Forward FLOPs: 12 layers × 2 (left + right) × d_model = trivial.
- Training: one extra `F.conv1d` per block per step.

### NOT bit-identical if …
- A subsequent code path consumes RNG between the `use_conv_ffn=True` and `use_conv_ffn=False` builds (we verified this is fine — the raw-`Parameter` init skips the RNG advance).
- The user passes an even kernel (the constructor asserts `kernel_size % 2 == 1`).
