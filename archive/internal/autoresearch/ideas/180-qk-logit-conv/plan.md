# Plan — 180-qk-logit-conv

## Flag
- `MultiHeadAttention.use_logit_conv: bool = False` (default OFF) — added in `models/layers.py` `__init__`.
- `MultiHeadAttention.logit_conv_kernel_size: int = 3` — small constant, kept as a kwarg with default 3.
- `TransformerBlock.use_logit_conv` (pass-through) + `TransformerBlock.logit_conv_kernel_size` (pass-through).
- `MinimalLLM.use_logit_conv` (pass-through) + `MinimalLLM.logit_conv_kernel_size` (pass-through).
- `configs/llm_config.py` — add `Tiny1M3MLogitConvConfig(Tiny1M3MConfig)` with `use_logit_conv: bool = True`.

## Change
- `models/layers.py` — in `MultiHeadAttention.__init__`, allocate `self.logit_conv_w = nn.Parameter(torch.zeros(n_heads, kernel_size))` when `use_logit_conv=True`; init center weight = 1.0 with `with torch.no_grad(): self.logit_conv_w[:, kernel_size // 2] = 1.0`. Stubs (None) when flag is off so attribute lookups are always valid.
- `models/layers.py` — in `MultiHeadAttention.forward`, after the causal/SWA mask and BEFORE softmax in the manual branch (~line 3466), apply causal 1D depthwise conv along the key axis (last axis) of `scores`. Use `F.pad(scores, (K-1, 0))` for left padding, then a per-head weighted sum over K shifted slices:
  `scores[b,h,t,s] ← Σ_{k=0..K-1} w_h[k] · padded[b,h,t,s+k]`.
  Identity init `w[:, K-1] = 1`, all others 0 ⇒ output = scores ⇒ step-0 forward is byte-identical to no-flag baseline.
- `models/layers.py` — add `self.use_logit_conv` to the manual-attention-path condition list (alongside 152/155/166/173) so SDPA's flash kernel is bypassed when the lever is on (SDPA cannot apply a pre-softmax score-space op).
- `configs/llm_config.py` — add `Tiny1M3MLogitConvConfig` (mirrors 178's pattern). Default off → baseline path bit-identical.
- Step-0 ≈ baseline when flag off (no Parameter registered, no branch taken, no-op) and when flag on (conv with delta kernel = identity on scores ⇒ softmax unchanged ⇒ max-abs-diff = 0.0 vs baseline).

## Control
- **Control**: `Tiny1M3MConfig` — no flag, val 6.4306.
- **Treatment**: `Tiny1M3MLogitConvConfig` — `use_logit_conv=True`.
- **Seed**: 42 (one seed only — never multi-seed, per the protocol).
- **Tier**: tiny1m3m (12L × 4H × 64d, 0.94M params, 3M tokens).

## Cost
- **Params Δ**: +H·K per block = 4·3 = 12 params/block. At 12 layers: 144 params total (+0.015% of 0.94M).
- **FLOPs Δ**: per forward, the conv adds ~B·H·T·S·K ≈ 8·4·2048·2048·3 ≈ 400M FLOPs (one-shot, <5% of the QK matmul's ~8.6G FLOPs).
- **Memory Δ**: transient `padded` tensor of [B, H, T, S+K-1] floats ≈ 8·4·2048·2050·4B ≈ 540MB at fp32 / 270MB at fp16. Discarded after the conv. No persistent state beyond the 144 weights.
- **Wall-clock Δ**: ~3-5% slowdown from the conv slice+sum; well within run noise.

## Run
- **Command (on the box)**: `python _arq_180-qk-logit-conv.py` (the daemon's queue calls this with the args baked into the stub).
- **Tier**: tiny1m3m.
- **Seed**: 42.
- **Expected wall-clock**: ~6-8 min (slightly above the ~6 min baseline due to the conv slice+sum overhead).
- **Pass/fail bar** (from `idea.md`):
  - **Pass**: treatment val ≤ control val within 2 ctrl brackets (≤ 6.4306 + 0.02 ≈ 6.45).
  - **Fail**: treatment val > control + 0.04 (clearly worse).
  - **Sub-noise**: |treatment val − control val| < 0.02 ⇒ log and move on (variance = two-ctrl bracket at tiny1m3m).
- **Artifact**: `_arq_180-qk-logit-conv.py` (top-level `C` subclass of `Tiny1M3MLogitConvConfig`) + `autoresearch/ideas/180-qk-logit-conv/run.json`.
