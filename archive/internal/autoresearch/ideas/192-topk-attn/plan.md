# Plan — 192-topk-attn

## Flag
- `LLMConfig.use_topk_attn: bool = False` (default OFF) — added in `configs/llm_config.py` `LLMConfig` (sibling of `use_entmax: bool = False` at line 474).
- `LLMConfig.topk_k: int = 512` — small constant, default 512 (= T/4 at the tiny1m3m max_seq_len of 2048). Pinned as a config int (no learnable scalar).
- `MultiHeadAttention.use_topk_attn` (pass-through) + `MultiHeadAttention.topk_k` (pass-through) — added to `MultiHeadAttention.__init__` signature.
- `TransformerBlock.use_topk_attn` (pass-through) + `TransformerBlock.topk_k` (pass-through).
- `MinimalLLM.use_topk_attn` (pass-through) + `MinimalLLM.topk_k` (pass-through).
- `configs/llm_config.py` — add `Tiny1M3MTopKAttnConfig(Tiny1M3MConfig)` with `use_topk_attn: bool = True, topk_k: int = 512`.

## Change
- `configs/llm_config.py` — add `use_topk_attn: bool = False` and `topk_k: int = 512` to `LLMConfig` (right after the existing `use_entmax` block at line 474, with a comment that engages 173 / 022 / 154 — same family, different parameterization). Add `Tiny1M3MTopKAttnConfig(Tiny1M3MConfig)` with both flags on, default `k=512`.
- `models/layers.py` — in `MultiHeadAttention.__init__`, add `use_topk_attn: bool = False, topk_k: int = 512` to the signature (mirroring the 173 entmax/022 softpick plumbing). `self.use_topk_attn = use_topk_attn; self.topk_k = topk_k`. No `nn.Parameter` registered — `k` is a config int, not a learnable scalar.
- `models/layers.py` — in `MultiHeadAttention.forward` manual-attention branch, add a topk-soft branch alongside the existing `if self.use_entmax: ... else: torch.softmax(...)` site (around line 4422). Order: topk first (cheap, single sort+scatter per head, no projection), then softmax. The branch:
  ```python
  if self.use_topk_attn:
      # 192 — Hard top-k sparse attention (Touvron et al. 2021,
      # DeiT III, arXiv:2103.17239). Per-row pre-softmax hard
      # sparsification: keep only the k largest scores per row,
      # scatter -inf to the rest, then softmax-renormalize over k.
      # `k = min(topk_k, scores.size(-1))` is defensive against
      # shorter eval contexts. Applied AFTER the causal mask so
      # -inf future positions are below the topk budget and never
      # selected. 0 new params. Forces the manual attention path
      # (the scatter can't go through SDPA's flash kernel).
      k = min(self.topk_k, scores.size(-1))
      topk_vals, topk_idx = scores.topk(k, dim=-1)
      sparse_scores = torch.full_like(scores, float("-inf"))
      sparse_scores.scatter_(-1, topk_idx, topk_vals)
      attn_w = torch.softmax(sparse_scores, dim=-1)
  elif self.use_entmax:
      ...
  else:
      attn_w = torch.softmax(scores, dim=-1)
  ```
- `models/layers.py` — append `or self.use_topk_attn  # 192 — Top-K: scatter write can't go through SDPA's flash kernel.` to the manual-path-forcing `elif` chain at the `or self.use_softpick` site (around line 4139) so the SDPA path is bypassed when topk is on.
- `models/llm.py` — add `self.use_topk_attn = getattr(config, "use_topk_attn", False)` and `self.topk_k = getattr(config, "topk_k", 512)` (mirroring the 173 entmax pass-through at line 380), and pass both into both the standard-block and the `nn.Sequential`-block construction sites (around lines 837, 1179).
- Step-0 ≈ baseline when flag off: no branch taken, no `nn.Parameter`, the `else: torch.softmax(scores, dim=-1)` is reached unchanged ⇒ bit-identical to baseline.
- Step-0 when flag on: NOT bit-identical (the k=512 topk is non-trivially different from full softmax at step 0 — same category as 173 / 022 / 154). This is the lever; framing as a structural lever in the same cohort as 173 / 022 / 154 is the spec-correct way to handle it.

## Control
- **Control**: `Tiny1M3MConfig` — plain tiny1m3m baseline. Cache val reference per `autoresearch/baseline-cache.json`: ~6.40.
- **Treatment**: `Tiny1M3MTopKAttnConfig` — `use_topk_attn=True, topk_k=512` (k = T/4 = 75% sparsity at T=2048).
- **Seed**: 42 (one seed only — never multi-seed, per protocol).
- **Tier**: tiny1m3m (12L × 4H × 64d, 0.94M params, 3M tokens).

## Cost
- **Params Δ**: 0 new params. `topk_k` is a config int, not a learnable scalar.
- **FLOPs Δ**: per forward, topk adds one `torch.topk` (T·log(k) per row, but the kernel is dense) + one `scatter_` + one softmax over k=512 not T=2048. Net: topk + scatter is ~1·B·H·T·T ≈ 4·8·2048·2048 = 134M FLOPs (<2% of the ~8.6G QK matmul). Softmax over k=512 is ~4× cheaper than over T=2048. Net Δ < 5% per step, well within run noise.
- **Memory Δ**: transient `topk_vals`/`topk_idx` (each [B, H, T, k] = 8·4·2048·512·4B ≈ 134MB at fp32 / 67MB at fp16) + a one-shot `sparse_scores` of the same shape as `scores` (dropped after softmax). No persistent state.
- **Wall-clock Δ**: ~3-5% slowdown from the topk+scatter; well within the ±0.04 noise band.

## Run
- **Command (on the box)**: `python _arq_192-topk-attn.py` (the daemon's queue calls this with the args baked into the stub).
- **Tier**: tiny1m3m.
- **Seed**: 42.
- **Expected wall-clock**: ~6-7 min (slightly above the ~6 min baseline due to the topk+scatter overhead).
- **Pass/fail bar** (from `idea.md`):
  - **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
  - **NULL**: `|trt_val − ctrl_val| < 0.01`.
  - **DRIFT**: `trt_val > ctrl_val + 0.01`.
- **Predicted magnitude (per idea.md mechanism)**:
  - **Primary prediction: NULL (|Δval| < 0.01).** 173 already closed the *learned*-support axis at 0.94M with 3 recode rounds. 192's only advantage is dropping the support-size dimension, but that drops *expressivity* too.
  - **Long-shot WIN (Δval ∈ [-0.005, -0.015])**: 154-rebased WIN says locality prior helps; top-k as a *strict* structural prior + the *cleaner gradient* over 173 both fire.
  - **DRIFT risk (Δval ∈ [+0.01, +0.05])**: d_k=16 / H=4 with a forced 75% sparsity at 12L. Bounded by the same d_k=16 that 177 hit, but 192 doesn't cross heads, so DRIFT is bounded, not catastrophic.
- **Artifact**: `_arq_192-topk-attn.py` (top-level `C = Tiny1M3MTopKAttnConfig`) + `autoresearch/ideas/192-topk-attn/run.json`.
