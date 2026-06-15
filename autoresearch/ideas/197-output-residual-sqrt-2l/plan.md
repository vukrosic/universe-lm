# Plan — 197 output-residual-sqrt-2l

## Flag
- `use_deepnet_alpha: bool = False`
  - `configs/llm_config.py:360` (`LLMConfig`, default OFF)
  - `configs/llm_config.py:2719` (`Tiny1M3MDeepNetAlphaConfig(Tiny1M3MConfig)` subclass, flag ON)
- Threaded through `models/llm.py` to both `TransformerBlock` construction sites (lines ~1017 and ~1299) — both gated behind `getattr(config, "use_deepnet_alpha", False)` so any older `LLMConfig` subclass without the attribute falls through to OFF.
- The block-level wiring lives in `models/layers.py:TransformerBlock.__init__` (the `use_deepnet_alpha` kwarg at the signature + `self.deepnet_alpha = float((2.0 * max(1, int(n_layers))) ** -0.5)` at the body).
- Used at 5 sites in `TransformerBlock.forward`:
  - **Parallel block** (line 7574): `x + α·(dropout(attn) + dropout(ffn))`
  - **Post-norm attn** (line 7611): `x = norm1(x + α·dropout(attn))`
  - **Post-norm ffn** (line 7632): `x = norm2(x + α·dropout(ffn))`
  - **Pre-norm attn** (line 7671): `elif self.use_deepnet_alpha: x = x + α·dropout(attn)` (sits between ReZero and `resid_mode` in the existing if/elif chain)
  - **Pre-norm ffn** (line 7781): same shape as attn

## Change
- **`configs/llm_config.py:360`** — already in place: `use_deepnet_alpha: bool = False` in `LLMConfig`.
- **`configs/llm_config.py:2672-2719`** — already in place: `Tiny1M3MDeepNetAlphaConfig(Tiny1M3MConfig)` with `use_deepnet_alpha: bool = True` (lever flag ON).
- **`models/layers.py`** — added the `elif self.use_deepnet_alpha: x = x + self.deepnet_alpha * self.dropout(...)` branch in 4 residual-add sites (pre-norm attn, pre-norm ffn, post-norm attn, post-norm ffn) and a top-of-return `if self.use_deepnet_alpha: return x + self.deepnet_alpha * (self.dropout(attn_out) + self.dropout(ff_out))` in the parallel block. The `self.deepnet_alpha` Python float is computed once at block construction (already present at the constructor body, line 7196). The flag-off baseline path is **bit-identical**: the `elif` chain is the same `ReZero → resid_mode → else` with deepnet_alpha injected as a parallel second branch — when the flag is off, control falls straight through to the existing `else: x = x + self.dropout(...)` add, no multiply, no extra op.
- **`models/llm.py`** — added `use_deepnet_alpha=getattr(config, "use_deepnet_alpha", False)` to both `TransformerBlock` constructor call sites (lines ~1017 and ~1299), with a 5-line block comment.
- Step-0 byte-identity when the flag is OFF: **preserved** (no extra op, no Parameter registered, no branch taken — verified by build-smoke below).
- Step-0 byte-identity when the flag is ON: **NOT preserved by construction** (the lever's purpose is the bounded regime from step 0; the spec explicitly documents this in `idea.md`).

## Control
- **Control**: `Tiny1M3MConfig` (flag OFF), seed 42, tier `tiny1m3m` (0.94M params · 3M tokens). Cache baseline 6.4216 at Vast V100 (per `token2science-papers-platform` memory; current champion ≈ 6.24, cache 6.40).
- **Treatment**: `Tiny1M3MDeepNetAlphaConfig` (flag ON), seed 42, same tier.
- **Two-ctrl rule applies** (the 0.005 bar is inside the cache noise band; the two-ctrl pass is the actual gate).
- Single-seed rule: only seed 42. No multi-seed, no seed sweeps (per the protocol).

## Cost
- **Params**: **0 new params** — `α` is a Python float computed once at block construction (one float per block, but it's a derived constant, not a learnable parameter). Total module size unchanged.
- **FLOPs**: 2 extra scalar multiplies per block per forward (one for the attention sublayer output, one for the FFN sublayer output, both `× α`). At tiny1m3m that's 24 scalar multiplies per token per forward — negligible compared to the matmul-heavy attention/FFN.
- **Memory**: 0 new buffers, 0 new Parameters. No shape changes anywhere.
- **Compilation**: the 5 added branches are pure-Python conditionals on `self.use_deepnet_alpha`; `torch.compile` (off at tiny1m3m by default anyway) is unaffected.

## Run
- Tier: `tiny1m3m` (always). Seed: 42 (always).
- Entry: `_arq_197-output-residual-sqrt-2l.py` (repo root) — defines top-level `class C(Tiny1M3MDeepNetAlphaConfig): pass`, then `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- Descriptor: `autoresearch/ideas/197-output-residual-sqrt-2l/run.json` (`name`, `arq_file`, `job_timeout="12m"`).
- Build-smoke: `MinimalLLM(C())` constructs on CPU without error (the same CPU smoke `_box_smoke.py` runs before GPU time).
- **Build-smoke verified** locally:
  - `MinimalLLM(Tiny1M3MConfig())(ids)` and `MinimalLLM(Tiny1M3MConfig())(ids)` with `eval()` + `seed reset` ⇒ `|Δy|max = 0.0` (off-path bit-identical).
  - `MinimalLLM(Tiny1M3MDeepNetAlphaConfig())(ids)` has `α = 0.20412...` and `use_deepnet_alpha = True` (the lever is wired).
  - The treatment is **not** bit-identical to the control (expected — lever's purpose is the bounded regime from step 0).
- Expected wall-clock per tier: ~4-5 min for treatment + the daemon-prepended ctrls.
- **Pass/fail bar (from `idea.md`)**:
  - **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule. A win gives a 0-param init lever with strong theoretical backing (DeepNet validated at 200-1000L).
  - **NULL**: `|trt_val − ctrl_val_mean| < 0.01`. A null closes the *fixed-depth-conditional-init* axis at this tier.
  - **DRIFT**: `trt_val > ctrl_val_mean + 0.01` (closes the lever family until ≥135M, since `0.204 ≈ 1/√24` may starve the residual stream at this tier).
  - Sub-noise is **inconclusive** per the one-seed-only rule — do NOT propose multi-seed.
