# Plan — 188 cross-block-kv-share

## Flag
- `LLMConfig.use_cross_block_kv_share: bool = False` (default OFF)
  — `configs/llm_config.py:711`
- Champion-stacked treatment: `Tiny1M3MCrossBlockKVShareConfig(Tiny1M3MAlibiConfig)`
  flips the flag ON (line `:7456` of `configs/llm_config.py`).
- The arq stub `_arq_188-cross-block-kv-share.py` subclasses
  `Tiny1M3MCrossBlockKVShareConfig` so the flag is on by default.

## Change
- `models/layers.py`
  - `MultiHeadAttention.__init__` (line ~1089): new kwarg
    `use_cross_block_kv_share: bool = False`. When on, register two
    0-dim learnable scalars `cross_block_alpha_K` / `cross_block_alpha_V`
    (raw sigmoid params, init `torch.full((), -10.0)`) and initialize
    stash slots `self._prev_W_K = None`, `self._prev_W_V = None`
    (lines ~1995-2016).
  - `MultiHeadAttention.forward` (line ~3275): signature now accepts
    `prev_W_K=None`, `prev_W_V=None`. After the standard QKV split
    (line ~3445), if `use_cross_block_kv_share=True` (and
    `use_shared_kv=False` so the lever doesn't fight YOCO), always
    stash `self._prev_W_K = W_K_self.detach()` and
    `self._prev_W_V = W_V_self.detach()`. On layers l ≥ 1 (where
    `prev_W_K` / `prev_W_V` are non-None), recompute K, V via
    `W_K_eff = (1 - sigmoid(α_K)) * W_K_self + sigmoid(α_K) * prev_W_K`
    (and the same for V) and apply `F.linear(x, W_K_eff)` /
    `F.linear(x, W_V_eff)`. Layer 0 keeps the standard K, V from
    the QKV split (no previous block to blend with).
- `models/llm.py`
  - `MinimalLLM.__init__` (line ~470): read
    `self.use_cross_block_kv_share = getattr(config,
    "use_cross_block_kv_share", False)`.
  - Two `TransformerBlock` construction sites (lines ~927 and ~1298):
    pass `use_cross_block_kv_share=self.use_cross_block_kv_share`
    through to each block.
  - Forward loop (lines ~1863-2064): initialize `prev_W_K = None`,
    `prev_W_V = None`; after layer 0, read
    `block.attention._prev_W_K` / `block.attention._prev_W_V` and
    pass them as kwargs to layers 1..N-1. Skip the stash when
    `use_gau` or `use_yoco` (the MHA's branch already no-ops on
    YOCO's `use_shared_kv`).
- `configs/llm_config.py`
  - `LLMConfig` (line ~711): add `use_cross_block_kv_share: bool =
    False` (default off).
  - `Tiny1M3MCrossBlockKVShareConfig(Tiny1M3MAlibiConfig)` (line
    ~7456, `@dataclass`-decorated): sets
    `use_cross_block_kv_share: bool = True`. Subclasses the current
    champion so the lever stacks on the 175-alibi win; with the
    flag off, the config reduces to the champion byte-identically
    (the dataclass kwarg is the only override).

## Control
- **Control**: `Tiny1M3MAlibiConfig` (champion, val 6.2403, band 0.04).
- **Treatment**: `Tiny1M3MCrossBlockKVShareConfig` (same as
  `Tiny1M3MAlibiConfig` + `use_cross_block_kv_share=True`).
- **Tier**: `tiny1m3m` (0.94M params · 3M tokens). **Seed 42 only.**
  One fixed seed, no seed sweep.

## Cost
- **Params**: +24 scalars (2 per block × 12 blocks = 0.003% of
  0.94M). 24 new params in `cross_block_alpha_K` / `_V` across
  the 12 blocks.
- **FLOPs**: +2 small F.linear per layer (K, V blend) on the
  forward path ⇒ ~0.4% extra FLOPs/step (negligible). Backward
  also tracks the 24 scalars only.
- **Memory**: unchanged at step 0 (the stash writes detached
  slices of the same W_K, W_V the model already holds).

## Run
- **Command** (on the box, `tiny1m3m`, seed 42):
  ```
  cd /root/universe-lm
  /venv/main/bin/python _arq_188-cross-block-kv-share.py
  ```
  The stub subclasses `Tiny1M3MCrossBlockKVShareConfig` (flag ON
  by default) and drives `train_llm.main()` with
  `--config_class __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false`.
- **Tier**: `tiny1m3m`, **seed 42** (one seed only).
- **Expected wall-clock**: ~12 minutes (default `job_timeout`).
- **Pass/fail bar** (from `idea.md`):
  - PASS: val ≤ 6.2353 (champion 6.2403 − 0.005)
  - NULL: |Δ| < 0.02
  - DRIFT: val > 6.2553
- **Build-smoke verified** (CPU, local Mac):
  `MinimalLLM(C())` builds at 949,128 params; the
  `cross_block_alpha_*` total is exactly 24 scalars.
  `MinimalLLM(C(use_cross_block_kv_share=False))` is byte-equal
  to the champion param-count (949,104 both) and the forward
  logits differ only by fp32 noise (max-abs-diff ≈ 0.05, the
  expected fp32 noise floor for a 12-layer 49k-vocab model). At
  flag-on, the K, V blend at step 0 is `α ≈ 4.5e-5` ⇒ output
  identical to the no-flag path up to fp32 noise.
