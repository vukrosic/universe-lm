# Plan — 191 token-attn-gain

## Flag
`use_token_attn_gain: bool = False` on `LLMConfig` (default OFF —
baseline path bit-identical). Treatment subclass
`Tiny1M3MTokenAttnGainConfig(Tiny1M3MAlibiConfig)` with
`use_token_attn_gain: bool = True`, `@dataclass`-decorated (per the
162/165/155/161/176/188 precedent that bare-class annotation breaks
dataclass field inheritance). Stacks on the 175-alibi champion
(per reviewer's recommendation, mirrors the 188 pattern that
subclasses `Tiny1M3MAlibiConfig`).

## Change

- **`configs/llm_config.py`** — add `use_token_attn_gain: bool = False`
  field on `LLMConfig` immediately after `use_cross_head_rmsnorm` (the
  181 sibling, since 191 is also a post-AV multiplicative lever). Add
  `@dataclass class Tiny1M3MTokenAttnGainConfig(Tiny1M3MAlibiConfig)`
  with `use_token_attn_gain: bool = True`, sibling of
  `Tiny1M3MCrossBlockKVShareConfig` (the 188 tier subclass is the
  closest precedent — also stacks on the 175-alibi champion).

- **`models/layers.py`** — `MultiHeadAttention.__init__`:
  - Add `use_token_attn_gain: bool = False` kwarg immediately after
    `use_cross_head_rmsnorm` (sibling of the 181 flag at line ~814).
  - Register one new parameter when the flag is on:
    - `self.token_attn_gain = nn.Parameter(torch.zeros(max_seq_len))`
      — per-position learnable scalar `γ_t ∈ R^T` (broadcast across
      batch and head/channel axes), init 0 ⇒ `(1 + 0) = 1.0` exactly
      ⇒ `attn * (1 + 0) = attn` byte-identical to baseline at step 0.
  - When the flag is off, set `self.token_attn_gain = None` (attribute
    lookup stays valid; the forward `if` guard short-circuits).

- **`models/layers.py`** — `TransformerBlock.__init__`: add
  `use_token_attn_gain: bool = False` kwarg, sibling of
  `use_cross_head_rmsnorm` (line ~4515). Pass through to the inner
  `MultiHeadAttention(...)` constructor.

- **`models/llm.py`** — `MinimalLLM.__init__`: add
  `self.use_token_attn_gain = getattr(config,
  "use_token_attn_gain", False)` immediately after the existing
  `self.use_cross_head_rmsnorm` read. Thread into both
  `TransformerBlock(...)` constructor sites (the standard block
  site at line ~1006 and the parallel site at line ~707), mirroring
  the 181 plumbing.

- **`models/layers.py`** — `MultiHeadAttention.forward` (apply site,
  AFTER the merge-reshape, on the post-merge `[B, T, d_model]`
  tensor, BEFORE the W_O projection — sibling of the `use_v_mix_conv`
  branch at line 4350 and the `use_av_carry` site at line 4361):
  ```python
  if self.use_token_attn_gain:
      # 191 — Per-token attention output gain. Multiply the
      # post-merge `[B, T, d_model]` attention output by a
      # learnable per-position scalar `(1 + γ_t)` where
      # `γ_t ∈ R^T` is shared across batch and the d_model
      # axis. Init γ=0 ⇒ (1 + 0) = 1 exactly ⇒
      # `attn * 1 = attn` byte-identical to baseline at step 0.
      # Per-token granularity (T scalars/block) is a different
      # axis from the closed per-head (160: H scalars),
      # per-channel (142: d_model scalars), and per-batch
      # diagonal (181: H + H·d_k) levers. Stacking on 175-alibi
      # champion matches the 188/186/021 stack-on pattern.
      attn_output = attn_output * (
          1.0 + self.token_attn_gain[:seq_len].view(1, seq_len, 1)
      )
  ```
  Slice to `[:seq_len]` so the [T_max]-length parameter is clipped
  to the actual forward `seq_len` (T_max = 2048 at tiny1m3m;
  inference at shorter T gets only the first `seq_len` γ values).
  Apply is on the post-merge tensor so it composes cleanly with
  every preceding pre-merge gate (`use_head_gain`,
  `use_attn_output_gate`, `use_attn_output_channel_gate`,
  `use_gated_attn`, `use_cross_head_rmsnorm`, `_apply_output_op`)
  by being a per-position multiplier in series.

- **`models/layers.py`** — `MultiHeadAttention.forward` (mutual
  exclusion asserts, sibling of the 181 mutex block at line ~2828):
  ```python
  assert not (self.use_token_attn_gain and self.use_head_gain), (
      "use_token_attn_gain=True is mutually exclusive with "
      "use_head_gain=True (both post-AV; the composition "
      "restructures the lever — turn 160 OFF to isolate 191)."
  )
  assert not (self.use_token_attn_gain and self.use_attn_output_gate), (
      "use_token_attn_gain=True is mutually exclusive with "
      "use_attn_output_gate=True (closed-045 per-head scalar "
      "ReZero gain; the composition restructures the lever)."
  )
  assert not (self.use_token_attn_gain and self.use_attn_output_channel_gate), (
      "use_token_attn_gain=True is mutually exclusive with "
      "use_attn_output_channel_gate=True (closed per-(h,k) "
      "ReZero gain; the composition restructures the lever)."
  )
  assert not (self.use_token_attn_gain and self.use_gated_attn), (
      "use_token_attn_gain=True is mutually exclusive with "
      "use_gated_attn=True (closed-024 input-conditional "
      "sigmoid gate; the composition restructures the lever)."
  )
  assert not (self.use_token_attn_gain and self.use_cross_head_rmsnorm), (
      "use_token_attn_gain=True is mutually exclusive with "
      "use_cross_head_rmsnorm=True (181 cross-head coupling; "
      "both post-AV and the composition restructures the lever)."
  )
  ```

**Step-0 identity**: at init `γ_t = 0` for all t ⇒ `(1 + 0) = 1`
exactly ⇒ `attn * 1 = attn` ⇒ **byte-identical to baseline at step 0
(max-abs-diff = 0.0)**. Algebraic identity, not fp32-tolerance — no
extra multiply-add noise.

## Control

- **Control**: unmodded `Tiny1M3MAlibiConfig` (the 175-alibi
  champion — current cache-authoritative baseline per
  `autoresearch/LEADERBOARD.md`). The daemon owns the ctrl (per
  `RUN-CONTRACT.md` §"Control is the daemon's, not the idea's"); we
  ship only the treatment stub.
- **Treatment**: `Tiny1M3MTokenAttnGainConfig` with
  `use_token_attn_gain: bool = True`. A/B at tiny1m3m, seed 42,
  single seed per the one-seed-only rule.
- **Cached baseline** (per `autoresearch/baseline-cache.json`):
  `val_mean = 6.4394 ± 0.04` (the 175-alibi cache; review.md cites
  6.4394 ± 0.04, distinct from the 181-cache 6.3988 because 181
  layers on `Tiny1M3MConfig`, not `Tiny1M3MAlibiConfig`).

## Cost

- **Params**: T=2048, n_layers=12. Per block: `T = 2048` scalars
  (shared across batch, heads, channels). Across 12 blocks: `12 ×
  2048 = 24,576` scalars. That's +2.6% of the 0.94M baseline
  (949,056 params) → 973,632 total. Well under the per-lever
  budget. Largest gain-family param count but mirrors the
  per-block granularity the lever requires.
- **FLOPs**: one FMA per (b, t, d_model) per block = 1× the
  attention's AV-product FLOPs divided by d_model (i.e., an
  elementwise scalar multiply of d_model channels per (b, t)).
  Negligible vs the dominant FFN/AV matmul cost.
- **Memory**: one new tensor per MHA of shape `[T_max = 2048]`,
  fp32, ~8 KB/block. Trivial.

## Run

- **Artifact**: `_arq_191-token-attn-gain.py` at repo root, imports
  `Tiny1M3MTokenAttnGainConfig` from `configs.llm_config`, defines
  `class C(Tiny1M3MTokenAttnGainConfig): pass` at top-level, and
  dispatches `train_llm.main()` with `--config_class __main__.C
  --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`
  (mirror the 188/181/176 `_arq_*.py` pattern). The stub's
  docstring should include a step-0 byte-identity check
  (`trt_step0_logits == ctrl_step0_logits` byte-exact, expected
  max-abs-diff = 0.0) — not run by default; documents the expected
  step-0 behavior.
- **Job**: `/venv/main/bin/python _arq_191-token-attn-gain.py` with
  `JOB_TIMEOUT=12m` (tiny1m3m runs in ~2-6 min; the cap keeps a
  hung treatment from burning the box).
- **Descriptor**: `autoresearch/ideas/191-token-attn-gain/run.json` —
  `{"name": "191-token-attn-gain", "arq_file":
  "_arq_191-token-attn-gain.py", "job_timeout": "12m"}`.
- **Val loss**: read from `~/arq/logs/191-token-attn-gain.log`
  (`grep "val_loss"`).
- **Pass/fail bar** (from `idea.md` §"Pass / fail bar"):
  - **WIN**: treatment val ≤ `val_mean − 0.02` (≤ **6.4194**) AND
    clears the ±0.04 noise band. Mirrors the 175-alibi bar.
  - **NULL**: |Δval| < 0.01 (inside [6.4294, 6.4494]) — the
    W_O-absorption hypothesis.
  - **DRIFT**: treatment val ≥ `val_mean + 0.04` (≥ **6.4794**).
  - **Crash / NaN / OOM** → `needs-recode` (round 1, inside budget).
  - **Sub-noise** (|Δval| < 0.01 but not DRIFT) is **INCONCLUSIVE**
    on one seed per the one-seed-only rule — do **not** re-run with
    extra seeds.

**LoC budget**: ~5 LoC flag decl + init + 5-line forward hook +
~12 LoC asserts + ~3 LoC config subclass + ~6 LoC plumbing. Total
~30 LoC, well under the 200 LoC cap. No new dependencies.

## Coordination note (per prompt §2)

`git status` shows 188's `use_cross_block_kv_share` flag is staged
in `models/layers.py` / `configs/llm_config.py` / `models/llm.py`.
191's flag name (`use_token_attn_gain`) does not collide with 188's
(`use_cross_block_kv_share`). 191 lands independently; no rebase
needed.

## Non-blocking note (mirrors 181 plan precedent)

The line-number references in this plan will drift against the
working tree as subsequent flags land. The semantic targets are
unambiguous: locate via `grep -n "use_cross_head_rmsnorm:
bool" models/layers.py` and `grep -n "if self.use_head_gain"
models/layers.py` and `grep -n "use_cross_head_rmsnorm=True is
mutually exclusive" models/layers.py`, and the implementation
places 191's flag decl, apply site, and asserts at those exact
semantic positions (sibling-of, not line-anchored).
