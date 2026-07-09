# Plan — 171 dropconnect-wo

## Flag

`use_dropconnect_wo: bool = False` (default OFF), `dropconnect_wo_rate:
float = 0.05` (ramp target — step-0 effective rate is 0.0 via the warmup
schedule), `dropconnect_wo_warmup_steps: int = 100` (ramp length).
Added next to `use_drop_key` / `drop_key_rate` on `LLMConfig`
(`configs/llm_config.py:735-768`); pass-through kwargs on
`MultiHeadAttention.__init__` (`models/layers.py:980`), storage
(`models/layers.py:1668-1680`); pass-through on `TransformerBlock`
(`models/layers.py:3377-3380`); pass-through on both block types from
`MinimalLLM.__init__` (`models/llm.py:270-274` reading the config and
`models/llm.py:629-631` / `models/llm.py:906-908` forwarding to the
blocks).

## Change

The parallel Claude had already wired a fixed-rate `dropconnect_wo` branch
on the W_O slice of the merged `qkvo_proj` (a Bernoulli mask on
`qkvo_proj[qkv_size:]`, rescale by `1/(1-p)`, eval-mode skip, no
parameter). I extended it with the **warmup-ramped effective rate** the
locked treatment calls for (`autoresearch/ideas/171-dropconnect-wo/idea.md`
mechanism §"Treatment (locked)"), so the lever matches the spec
exactly:

- **`configs/llm_config.py`** (parallel agent's diff already had
  `use_dropconnect_wo` / `dropconnect_wo_rate`):
  - **Aligned config default to ramp target.** `dropconnect_wo_rate`
    was 0.1 (Wan's CIFAR/ImageNet sweet spot from the parallel diff
    comment); I changed it to **0.05** to match the locked treatment
    (halved for ramp safety, per `idea.md:113-119`). The trt subclass
    `Tiny1M3MDropConnectWOConfig` overrides to 0.05 anyway, so the
    change is a no-op for the active run but keeps the off-by-default
    behavior aligned with the lock.
  - **Added `dropconnect_wo_warmup_steps: int = 100`** at
    `configs/llm_config.py:754-768`. The ramp length is the locked
    value (covers ~52% of the 192-step tiny1m3m schedule, leaving ~92
    steps at the steady-state rate).
  - **Added `Tiny1M3MDropConnectWOConfig` subclass** at
    `configs/llm_config.py:2075-2128` setting
    `use_dropconnect_wo=True, dropconnect_wo_rate=0.05,
    dropconnect_wo_warmup_steps=100`. Same pattern as
    `Tiny1M3MReLU2FFNConfig` / `Tiny1M3MSwigluFFNConfig` /
    `Tiny1M3MAttnLogitBiasConfig`.

- **`models/layers.py`** (parallel agent's diff already had the basic
  flag wiring and the W_O branch):
  - **`MultiHeadAttention.__init__`** (`models/layers.py:980`): added
    `dropconnect_wo_warmup_steps: int = 100` kwarg.
  - **`MultiHeadAttention.__init__` body**
    (`models/layers.py:1666-1680`): stored
    `self.dropconnect_wo_warmup_steps = int(dropconnect_wo_warmup_steps)`
    and `self._dc_step_count: int = 0` (per-MHA Python int counter,
    incremented at the END of forward so step 0 sees the count before
    it ticks, making the first forward call byte-identical).
  - **`MultiHeadAttention.forward` W_O application site**
    (`models/layers.py:3271-3341`): replaced the fixed-rate guard
    with the warmup-ramped effective rate:
    ```python
    effective_rate = dropconnect_wo_rate * min(step, warmup) / warmup
    ```
    The `self.training` guard is kept (eval short-circuits before any
    RNG is consumed); the `rate > 0.0` guard is moved inside the
    `effective_rate > 0.0` check (so the effective rate, not the raw
    rate, controls the mask application).
  - **`MultiHeadAttention.forward` end-of-function**
    (`models/layers.py:3362-3368`): added
    `self._dc_step_count += 1` immediately before `return output`.
    Done at the END of forward so step 0 (the first forward call) sees
    `step=0` ⇒ `effective_rate=0.0` ⇒ mask branch short-circuits ⇒
    trt forward is byte-identical to baseline at step 0.
  - **`TransformerBlock.__init__`** (`models/layers.py:3377-3380`):
    added `dropconnect_wo_warmup_steps: int = 100` pass-through kwarg.
  - **`TransformerBlock.__init__` MHA construction**
    (`models/layers.py:3882`): added
    `dropconnect_wo_warmup_steps=dropconnect_wo_warmup_steps` to the
    MHA call.

- **`models/llm.py`**:
  - **`MinimalLLM.__init__`** (`models/llm.py:270-274`): added
    `self.dropconnect_wo_warmup_steps = getattr(config,
    "dropconnect_wo_warmup_steps", 100)`.
  - **`MinimalLLM.__init__` YOCO upper-half block construction**
    (`models/llm.py:631`): added
    `dropconnect_wo_warmup_steps=self.dropconnect_wo_warmup_steps`.
  - **`MinimalLLM.__init__` standard TransformerBlock construction**
    (`models/llm.py:908`): same.

Step-0 byte-identical: with `use_dropconnect_wo=True` AND
`dropconnect_wo_rate=0.05` AND `dropconnect_wo_warmup_steps=100`, the
first forward call sees `step=0` ⇒ `effective_rate=0.05 * 0/100 = 0.0`
⇒ `effective_rate > 0.0` is False ⇒ mask branch never taken ⇒ no RNG
consumed ⇒ `w_o = qkvo_proj[qkv_size:]` (unchanged) ⇒ output =
`attn_output @ W_O` ⇒ identical to the no-DropConnect baseline
forward at step 0 (max-abs-diff = 0.0 across the full forward). Step
100 sees `effective_rate = 0.05` (ramp reaches the target); steps
100+ hold at 0.05. With `use_dropconnect_wo=False` (default) the
`self.use_dropconnect_wo` guard short-circuits the whole block ⇒
baseline path bit-identical regardless of rate / warmup.

## Control

- **Trt**: `Tiny1M3MDropConnectWOConfig` (config flags: rate=0.05,
  warmup=100, use_dropconnect_wo=True). Seed 42. Tier tiny1m3m.
- **Ctrl**: bare `Tiny1M3MConfig`, seed 42, tier tiny1m3m. (Daemon
  owns the ctrl — the cached 6.4216 / 6.4394±0.04 bracket from
  `autoresearch/baseline-cache.json` is reused; the daemon prepends
  ctrls itself when `baseline.sh check` returns MEASURE.)
- **Tier**: always `tiny1m3m` per the ONE-TIER-ONLY rule. No scale-up
  retry possible.

## Cost

- **Params Δ**: 0 (no new `nn.Parameter`). Mask is sampled fresh each
  forward from `torch.empty_like(w_o).bernoulli_(keep_prob)`; not a
  learnable parameter.
- **FLOPs Δ**: per forward, when `effective_rate > 0.0`: one
  `bernoulli_` (negligible) + one elementwise mul + one elementwise
  div on a `[d_model, d_model]` tensor = 3 × d_model² = 3 × 4096 =
  12288 ops per block per call × 12 blocks = 147456 ops/step. At
  effective_rate=0 this is 0 ops (branch short-circuited). At ~3M
  tokens / 192 steps, the ramp spends ~50 steps at near-zero rate
  and ~92 steps at full 0.05 rate. Total amortized cost ≈ 75% ×
  147456 ≈ 110592 ops/step × 192 ≈ 2.1 × 10⁷ ops total — sub-noise
  relative to the ~10⁹ base FLOPs.
- **Memory Δ**: 0 (mask is a temporary; `w_o` is a slice view of
  `qkvo_proj` that we rebind in scope, not a copy).
- **Wall-clock Δ**: < 1% of the 3-4 min baseline compute.

## Run

- **Tier**: `tiny1m3m` (mandatory; ONE-TIER-ONLY).
- **Seed**: `42` (mandatory; ONE-SEED-ONLY).
- **Command** (trt):
  `/venv/main/bin/python _arq_171-dropconnect-wo.py` (the daemon runs
  this via `python <arq_file>` per `RUN-CONTRACT.md`).
- **Expected wall-clock**: ~3-4 min (same tier as ctrl).
- **Pass/fail bar** (copied from `idea.md:160-170`):
  - **Δ ≤ −0.020**: signal — single-seed-detectable improvement.
  - **−0.020 < Δ < −0.005**: informative but inconclusive — treat as
    null-and-close per the two-ctrl WIN rule.
  - **Δ ≥ −0.005**: null.
  - **Null payoff** (per `idea.md:171-178`): "null closes the weight-
    level axis of the regularizer family (after token-level 147 and
    path-level 111, weight-level 171 completes the family exhaustion
    at 0.94M). The remaining regularizer axes at this tier are not
    in the per-mask family."
- **Baseline band**: cached 6.4216 from the Vast V100 box
  (`token2science-papers-platform` memory); 6.4394±0.04 / 6.4504±0.0558
  from the two-ctrl bracket in `autoresearch/baseline-cache.json`.
