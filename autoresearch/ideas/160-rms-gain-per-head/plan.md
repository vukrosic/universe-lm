# Plan — 160 rms-gain-per-head

## Flag
`use_head_gain: bool` (default `False`) on `LLMConfig`.
Config class `Tiny1M3MHeadGainConfig(Tiny1M3MConfig)` with `use_head_gain=True`.
File: `configs/llm_config.py:140` (default) and `configs/llm_config.py:2012` (treatment).

## Change
- `models/layers.py` — `MultiHeadAttention.__init__` accepts `use_head_gain: bool = False`
  kwarg. When on, registers `self.head_gain = nn.Parameter(torch.ones(self.n_heads))`.
  In `forward()`, after `attn_output` is produced (post-SDPA or post-manual path,
  BEFORE the existing `use_attn_output_gate`/`use_attn_output_channel_gate`/
  `use_gated_attn` branches so it composes cleanly with all of them — they multiply
  through), apply `attn_output = attn_output * self.head_gain.view(1, n_heads, 1, 1)`.
  `TransformerBlock` threads the flag into the inner MHA.
- `models/llm.py` — `MinimalLLM` reads `use_head_gain=getattr(config, "use_head_gain",
  False)` and passes it through the standard `TransformerBlock(...)` constructor call.
- `configs/llm_config.py` — adds `use_head_gain: bool = False` on `LLMConfig` and
  `Tiny1M3MHeadGainConfig(Tiny1M3MConfig)` with `use_head_gain = True`.

Step-0 identity: `head_gain` init = `1.0` exactly ⇒ `o_h *= 1 = o_h` byte-identical
to baseline at step 0 when the flag is on. With flag off, no Parameter is registered,
no branch is taken — baseline path bit-identical.

## Control
- A: `Tiny1M3MConfig` (seed 42, flag OFF) — bare tier config.
- B: `Tiny1M3MHeadGainConfig` (seed 42, flag ON) — same tier, `use_head_gain=True`.
- Tier: `tiny1m3m` (0.94M params, 3M tokens). Seed 42 only.

## Cost
- Params: + H scalars/layer = +48 at tiny1m3m (4 heads × 12 layers, +0.005% of 0.94M).
- FLOPs: + 1 multiply per head per token per layer (~negligible).
- Memory: + 4×12 = 48 floats, ~negligible.

## Run
- Command (after code lands + sync to box):
  ```
  cd /root/universe-lm && /venv/main/bin/python _arq_160-rms-gain-per-head.py
  ```
  This invokes `train_llm.main()` with `--config_class __main__.C --seed 42
  --dataset_path processed_data/pretrain_1B --warmup false`. The daemon's
  `claimable()` picks the idea up from `needs-run` via `run.json`.
- Tier: `tiny1m3m`, seed 42.
- Expected wall-clock: ~2-6 min (same as baseline).
- **Pass/fail bar** (from `idea.md`):
  - PASS ≤ ctrl − 0.005 (a real win extends qk_norm's win on the pre-softmax
    magnitude axis to the post-AV axis).
  - NULL band |Δ| < 0.005 (the post-AV magnitude axis is plausibly redundant given
    the W_O projection that follows).
  - DRIFT > +0.005 (the lever is harmful — over-parameterized reinit risk).
