# Plan — 200 rope-phase-offset-per-layer

## Flag
- `use_per_layer_k_rotation: bool = False`
  - `configs/llm_config.py:1258` (LLMConfig, default OFF)
  - `configs/llm_config.py:7394` (Tiny1M3MPerLayerKRotationConfig subclass, flag ON)
- Threaded through `models/layers.py:1442` (MHA kwarg) → `:5281` (TransformerBlock kwarg) → `models/llm.py:346` (model reads `getattr(config, ..., False)`) → pass-through at both MHA construction sites (`models/llm.py:842` and `:1209`).

## Change
1. **`models/layers.py` — `MultiHeadAttention.__init__`**: add `use_per_layer_k_rotation: bool = False` kwarg (line 1442). When True, allocate `self.per_layer_k_rotation_angles = nn.Parameter(torch.zeros(self.d_k // 2))` (line 2623) — **8 angles per block, shared across heads** (no head axis). Asserts even `d_k` for the per-pair 2D rotations (mirrors the 185 assertion pattern at line 2620).
2. **`models/layers.py` — `MultiHeadAttention.forward`**: add per-plane 2D rotation branch after the 185 static-K-rotation branch and before `use_k_gain` (line 3797). Q is **untouched**; K is reshaped `[B,T,H,d_k//2,2]` and rotated per-plane via `cos_a * K_a - sin_a * K_b`, `sin_a * K_a + cos_a * K_b` (cos/sin broadcast over the head axis). Site is post-RoPE / post-qk_norm_depth / post-GQA-repeat — matches the placement used by 185.
3. **`models/layers.py` — `TransformerBlock.__init__`**: add pass-through kwarg `:5281` → forwarded to inner MHA at `:5867`.
4. **`configs/llm_config.py`**: add `use_per_layer_k_rotation: bool = False` to `LLMConfig` (line 1258) and add `Tiny1M3MPerLayerKRotationConfig(Tiny1M3MConfig)` subclass with `use_per_layer_k_rotation: bool = True` (line 7394).
5. **`models/llm.py`**: read flag from config at `:346` (with `getattr(..., False)` fallback) and pass to both MHA construction sites (`:842`, `:1209`).
- Step-0 ≡ baseline when flag is OFF: no Parameter registered, no branch taken, baseline forward graph is bit-identical.
- Step-0 ≡ baseline when flag is ON: `φ_{l,i} = 0` ⇒ `cos(0) = 1.0`, `sin(0) = 0.0` in fp32 (bit-exact) ⇒ `R_l = I_{d_k}` ⇒ `K = R_l @ K = K` exactly ⇒ QK^T unchanged ⇒ loss unchanged. The build-smoke verifies `max_abs_diff(MinimalLLM(Tiny1M3MConfig())(ids), MinimalLLM(Tiny1M3MPerLayerKRotationConfig())(ids)) == 0.0` under seed 42.

## Control
- **Control**: `Tiny1M3MConfig` (flag OFF), seed 42, tier tiny1m3m (0.94M params · 3M tokens). Box `5b8a7fea8963`, val 6.3988 ± 0.04, n_measurements = 3.
- **Treatment**: `Tiny1M3MPerLayerKRotationConfig` (flag ON), seed 42, same tier.
- Single-seed rule: only seed 42. No multi-seed, no seed sweeps (per the protocol).

## Cost
- **Params**: 8 angles × 12 layers = **96 fp32 scalars = 384 bytes** total. +0.001% of the 0.94M model — negligible.
- **FLOPs**: ~8 plane rotations × 4 mul-add pairs × (B · T · H) = ~524K flops per block per forward. At tiny1m3m (B≈32, T=2048, H=4, 12 blocks), that's a few hundred thousand extra flops/step vs the 6.4-loss baseline's ~MFLOPs/step — negligible.
- **Memory**: 96 scalars total — negligible. No new buffers.

## Run
- Tier: `tiny1m3m` (always). Seed: 42 (always).
- Entry: `_arq_200-rope-phase-offset-per-layer.py` (repo root) — defines top-level `class C(Tiny1M3MPerLayerKRotationConfig): pass`, then `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- Descriptor: `autoresearch/ideas/200-rope-phase-offset-per-layer/run.json` (`name`, `arq_file`, `job_timeout="12m"`).
- Build-smoke: `MinimalLLM(C())` constructs on CPU without error (the same CPU smoke `_box_smoke.py` runs before GPU time). Expected wall-clock per tier: ~4–5 minutes for treatment + ctrls.
- **Pass/fail bar (from idea.md)**:
  - **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule. Δ ≤ −0.01 is a strong confirmation of the depth × pair axis.
  - **NULL**: `|trt_val − ctrl_val_mean| < 0.01`.
  - **DRIFT**: `trt_val > ctrl_val_mean + 0.01` (closes the lever family until ≥135M).
  - Sub-noise is **inconclusive** per the one-seed-only rule — do NOT propose multi-seed.