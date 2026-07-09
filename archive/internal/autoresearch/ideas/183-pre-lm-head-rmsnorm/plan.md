# Plan — 183 pre-lm-head-rmsnorm

## Flag
- `use_pre_lm_head_rmsnorm: bool = False` — default OFF
- File: `configs/llm_config.py:571` (declaration on `LLMConfig` base), `configs/llm_config.py:2449` (`Tiny1M3MPreLMHeadRMSNormConfig` subclass with flag ON, stacking on `Tiny1M3MAlibiConfig`)

## Change
- `configs/llm_config.py`:
  - L571: add `use_pre_lm_head_rmsnorm: bool = False` on `LLMConfig` base
  - L2449: `class Tiny1M3MPreLMHeadRMSNormConfig(Tiny1M3MAlibiConfig): use_pre_lm_head_rmsnorm: bool = True`
- `models/llm.py`:
  - L1337-1342: when `use_pre_lm_head_rmsnorm` is on, register `self.pre_head_norm = nn.RMSNorm(config.d_model, eps=1e-6)` (default `weight=1, bias=0` survives the global `_init_weights`) and `self.pre_head_scale = nn.Parameter(torch.zeros(()))` (scalar gate)
  - L1854-1856: in `_run_post_embed`, between `self.output_dropout(x)` and the `lm_head` (untied / factorized branches), apply the gated mix `x = (1 − scale) · x + scale · RMSNorm(x)`. With the flag off, the modules are not built and the forward path is byte-identical to the no-flag champion. With the flag on at step 0, `scale = 0` ⇒ the mix is exactly `x`, byte-identical to the champion.
- `_arq_183-pre-lm-head-rmsnorm.py` (repo root): `class C(Tiny1M3MPreLMHeadRMSNormConfig): pass`, calls `train_llm.main()` with the standard tiny1m3m/seed-42 args.

## Control
- Ctrl: `Tiny1M3MAlibiConfig` (current champion, val 6.2403, band 0.04, seed 42)
- Trt: `Tiny1M3MPreLMHeadRMSNormConfig` (Ctrl + `use_pre_lm_head_rmsnorm=True`)
- Tier: `tiny1m3m` (0.94M params · 3M tokens)
- Seed: 42 (one seed only)

## Cost
- Params Δ: +65 (64 RMSNorm gain weights + 1 scalar gate) = +0.007% of 0.94M
- FLOPs Δ: negligible (one RMSNorm over d_model=64 per token in fp32, applied once after the final dropout, before the LM head)
- Memory Δ: 1 fp32 scalar + 64 fp32 weights — negligible

## Run
- Command: `/venv/main/bin/python _arq_183-pre-lm-head-rmsnorm.py` on the RTX 3060 box
- Tier: `tiny1m3m` (seed 42, dataset `processed_data/pretrain_1B`, warmup `false`)
- Expected wall-clock: ≤ 12m (matches default `job_timeout`; tiny1m3m standard runs land in 6-9m on the 3060)
- Pass/fail bar (from `idea.md`):
  - **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule
  - **NULL**: `|trt_val − ctrl_val_mean| < 0.01`
  - **DRIFT**: `trt_val > ctrl_val_mean + 0.01`
  - Sub-noise (`|Δ| < 0.005`) is logged NULL with `cache_authoritative: true` per the one-seed-only rule
- Cache reference (re-pull on run day): `autoresearch/baseline-cache.json` (current cache has 175-alibi-slopes champion at val 6.2403, band 0.04)
