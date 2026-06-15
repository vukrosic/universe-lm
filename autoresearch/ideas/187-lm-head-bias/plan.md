# Plan — 187 lm-head-bias

## Flag
- `use_vocab_bias: bool = False` — default OFF
- File: `configs/llm_config.py:545` (declaration on `LLMConfig` base, OH5 docstring block at lines 541-544). **Already wired** — this plan does not add a new LLMConfig field.
- New subclass `Tiny1M3MAlibiLMHeadBiasConfig(Tiny1M3MAlibiConfig)` at `configs/llm_config.py:2497` (immediately after the 183 `Tiny1M3MPreLMHeadRMSNormConfig` subclass at line 2463, before the 147 DropConnect subclass at line 2497) with `use_vocab_bias: bool = True`. Stacks on the current ALiBi champion, matching the 184 / 183 subclass-on-champion pattern.

## Change
- `configs/llm_config.py`:
  - **No new LLMConfig field.** `use_vocab_bias` already exists at L545.
  - Add `Tiny1M3MAlibiLMHeadBiasConfig(Tiny1M3MAlibiConfig)` subclass with `use_vocab_bias: bool = True`, modeled after the 184 `Tiny1M3MLogitScaleConfig` and 183 `Tiny1M3MPreLMHeadRMSNormConfig` precedent subclasses.
- `models/llm.py`:
  - **No model changes.** Allocation at L1311-1313 (`if self.use_vocab_bias: self.vocab_bias = nn.Parameter(torch.zeros(config.vocab_size))`) and forward hook at L1883-1884 (`if self.use_vocab_bias: logits = logits + self.vocab_bias`) are already in place. OH5 VocabBias is fully wired.
- `_arq_187-lm-head-bias.py` (repo root): `class C(Tiny1M3MAlibiLMHeadBiasConfig): pass`, calls `train_llm.main()` with the standard tiny1m3m / seed-42 args.

## Control
- Ctrl: `Tiny1M3MAlibiConfig` (current champion, val 6.2403 ± 0.04, seed 42, box `5b8a7fea8963`)
- Trt: `Tiny1M3MAlibiLMHeadBiasConfig` (Ctrl + `use_vocab_bias=True`)
- Tier: `tiny1m3m` (0.94M params · 3M tokens)
- Seed: 42 (one seed only — never multi-seed)

## Cost
- Params Δ: +49,152 (1 fp32 scalar per vocab token; `vocab_size = 49152` at `configs/llm_config.py:26`). +5.23% of 0.94M. Routes to AdamW under the existing 1-D-parameter rule.
- FLOPs Δ: one fp32 add per (B, T, V) cell — negligible compute (plan row OH5 explicitly tags this "many params but trivial compute").
- Memory Δ: 49,152 fp32 scalars = ~192 KB of param storage + ~192 KB of optimizer state (AdamW: m, v in fp32) — negligible.

## Run
- Command: `/venv/main/bin/python _arq_187-lm-head-bias.py` on the RTX 3060 box
- Tier: `tiny1m3m` (seed 42, dataset `processed_data/pretrain_1B`, warmup `false`)
- Expected wall-clock: ≤ 12m (matches default `job_timeout`; tiny1m3m standard runs land in 6-9m on the 3060)
- Pass/fail bar (from `idea.md`):
  - **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` (= 6.2353) AND clears the two-ctrl rule
  - **NULL**: `|trt_val − ctrl_val_mean| < 0.01` (sub-noise inconclusive per one-seed-only rule)
  - **DRIFT**: `trt_val > ctrl_val_mean + 0.01` (= 6.2503) — could occur if the bias over-fits to the training unigram prior and the val distribution has a meaningfully different token prior
- Cache reference (re-pull on run day): `autoresearch/baseline-cache.json` (current pinned cache: champion `Tiny1M3MAlibiConfig` at val 6.2403, val_std 0.0088, noise_band 0.04, n_measurements 3, measured 2026-06-15T07:04:48Z)
- Step-0 byte-identity (verify on release): with `vocab_bias = zeros(V)`, `logits + 0 = logits` exactly in fp32 ⇒ `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0` AND `max_abs_diff(trt_step0_loss, ctrl_step0_loss) == 0.0` where ctrl is the ALiBi champion stack.