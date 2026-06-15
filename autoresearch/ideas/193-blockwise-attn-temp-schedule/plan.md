# Plan — 193 blockwise-attn-temp-schedule

## Flag
- `use_block_temp_schedule: bool = False` on `LLMConfig` (next to
  `use_qk_clamp` / `qk_clamp_c` at `configs/llm_config.py:217-218`,
  in the same neighborhood as the depth-conditional-multiplicative
  siblings `use_qk_rms_scaling` (188) and `use_qk_clamp` (195)).
  Default OFF ⇒ baseline path bit-identical. New scalar kwarg
  `block_temp_alpha: float = 0.0` (default = flat schedule; trt = `−0.3`).
- The committed `Tiny1M3MBlockTempConfig(Tiny1M3MConfig)` subclass
  at `configs/llm_config.py` (just after `Tiny1M3MQKClampConfig`) sets
  `use_block_temp_schedule=True` and `block_temp_alpha=-0.3`. The
  subclass is `@dataclass`-decorated so the field overrides take effect
  (per the same caveat noted in `_arq_188-qk-rms-scaling.py`: a bare
  `class C(Tiny1M3MConfig): use_block_temp_schedule: bool = True`
  would NOT override the parent's field default because the subclass
  isn't re-`@dataclass`-decorated, so the annotation is ignored).
- The precomputed per-block scalar `τ_b = 1 + α · cos(π · b / (L − 1))`
  is computed inline in the `nn.ModuleList` comprehension in
  `models/llm.py` (`MinimalLLM.__init__`, the single
  `TransformerBlock(...)` call site) and passed as the `tau_b: float`
  kwarg. `tau_b` is registered as a non-Parameter `Buffer` of shape
  `[1]` on the inner `MultiHeadAttention` (per-block) when the flag
  is on.

## Change
- `configs/llm_config.py` — `LLMConfig` (added two fields at
  `:218`ish — same neighborhood as the depth-conditional siblings
  188 / 195). Added `Tiny1M3MBlockTempConfig(Tiny1M3MConfig)`
  subclass after `Tiny1M3MQKClampConfig` at the same level as the
  other attention-lever `Tiny1M3M<X>Config` subclasses.
- `models/layers.py` — three insertions:
  1. `MultiHeadAttention.__init__`: new kwargs `use_block_temp_schedule:
     bool = False`, `block_temp_alpha: float = 0.0`, `tau_b: float = 1.0`
     (the kwarg list insertion is right after `qk_clamp_c`, mirroring
     the 188 sibling's placement). In the body, when
     `use_block_temp_schedule=True`, register
     `self.tau_b = torch.tensor(float(tau_b), dtype=torch.float32)`
     as a non-Parameter `Buffer` (no gradient).
  2. `MultiHeadAttention.forward` (manual-path branch only): after
     the `scores = torch.matmul(Qn, Kn.transpose(-1, -2)) * scale`
     (or the 155 per-head-temp replacement) and BEFORE the
     `if self.use_qk_clamp:` branch, add `if self.use_block_temp_schedule:
     scores = scores / self.tau_b.view(1, 1, 1, 1)`. Inserted AFTER
     the scale but BEFORE the clamp + cross-block share so it composes
     uniformly with all downstream score-side levers (188 qk_rms_scaling,
     195 qk_clamp, 204 cross-block score share, etc.).
  3. Manual-path forcing list: added
     `or self.use_block_temp_schedule` to the existing forcing-list
     expression at `models/layers.py:4554ish`, right next to the
     `or self.use_qk_clamp` line. Forces the manual attention path
     so SDPA's flash kernel doesn't fuse QK^T+softmax+AV (the
     pre-softmax score must be exposed for the `τ_b` divide).
  4. `TransformerBlock.__init__`: pass-through kwargs
     `use_block_temp_schedule`, `block_temp_alpha`, `tau_b` (placed
     right after the `qk_clamp_c` pass-through, mirroring the MHA
     ordering). The TransformerBlock passes these through to the
     inner `MultiHeadAttention` constructor (single call site, just
     after `use_qk_clamp=use_qk_clamp, qk_clamp_c=qk_clamp_c`).
- `models/llm.py` — single insertion: at the only `TransformerBlock(...)`
  call site (inside the `nn.ModuleList` comprehension at `:1238`,
  inside `for i in range(n_unique)`), pass `use_block_temp_schedule`,
  `block_temp_alpha`, and the inline-computed `tau_b` (per-block
  formula using the comprehension variable `i`). At
  `block_temp_alpha=0.0` (the default) `tau_b = 1.0` for every block
  ⇒ `scores / 1.0 = scores` byte-identical to baseline. The three
  flags are gated by `getattr(config, ..., default)` so a config that
  predates 193 still constructs (the `use_block_temp_schedule` default
  is `False`, so no Buffer is registered on any MHA when the config
  doesn't set it).
- Step-0 byte-identity: `α = 0 ⇒ τ_b = 1` for all `b`. With the flag
  off (default), the entire `if self.use_block_temp_schedule:` branch
  is never taken in `forward()`, so the scores path is byte-identical
  to the standard `Q·K^T / √d_k`. With the flag on but `α = 0`,
  `scores / 1.0 = scores` in fp32 (IEEE 754 identity). The Buffer
  is only registered when the flag is on, so the ctrl ctrl
  (`Tiny1M3MConfig`) carries zero overhead.

## Control
- **Control (this run's ctrl)**: `Tiny1M3MConfig` (plain tiny1m3m
  baseline, cache-mean 6.3988 ± 0.04, n=3, measured 2026-06-15
  per `autoresearch/baseline-cache.json` val_mean). The daemon
  owns the baseline; we ship the treatment only.
- **Treatment**: `Tiny1M3MBlockTempConfig` (the new `@dataclass`
  subclass with `use_block_temp_schedule=True`, `block_temp_alpha=-0.3`).
- **Seed**: 42 (one seed only — `feedback-one-seed-only`).
- **Tier**: tiny1m3m (0.94M params, 3M tok, single seed 42).

## Cost
- **Params**: 0 new parameters. The schedule is hard-coded
  (`τ_b = 1 + α · cos(π · b / (L − 1))`); the per-block `tau_b`
  Buffer is a non-Parameter constant, not a learnable weight.
- **FLOPs**: +1 elementwise divide on `[B, H, T, T]` per block
  per forward (one `scores / self.tau_b.view(1, 1, 1, 1)`). At
  tiny1m3m (`T ≤ 2048`, `B = batch_size`, `H = 4` heads,
  12 blocks) this is ≈ `B·4·T²·12 = B · 200K` extra ops per forward
  — negligible vs the QK^T matmul (`B · 4 · T² · d_k = B · 4 · T² · 16`
  is the same order, but the divide is one fp32 op vs the matmul's
  `d_k = 16` multiply-adds). Net: < 1% extra FLOPs.
- **Memory**: 12 floats (`tau_b` per block) on `cpu/cuda` device,
  transferred with `state_dict()`. Negligible.
- **Compile path**: forces the manual attention path (see forcing
  list edit) — SDPA's flash kernel fuses QK^T+softmax+AV and can't
  expose the pre-softmax logit for the per-block temperature divide.
  The manual path is the canonical path used by all score-side
  levers (155, 188, 195, 204, etc.); no perf regression beyond
  what those levers already pay.

## Run
- Command (GPU box): `_arq_193-blockwise-attn-temp-schedule.py`
  with seed 42 and `--warmup false` (the daemon's `_box_smoke.py`
  runs the build on CPU first; the actual GPU run is launched by
  `queue-daemon.sh` reading `autoresearch/ideas/193-blockwise-attn-
  temp-schedule/run.json`).
- Tier: tiny1m3m, seed 42. Expected wall-clock: ≤12m
  (`job_timeout=12m` in `run.json`).
- Val read: from `autoresearch/remote-results/<run-dir>/trt_*.log`
  — search for the final `val` line emitted by `train.py`.
- Champion reference: `Tiny1M3MConfig` (baseline, val mean 6.3988 ±
  0.04, `autoresearch/baseline-cache.json`). The 175-alibi WIN at
  val 6.2631 is the **adjacent architectural reference** (175 is
  the depth-uniform *additive* WIN; 193 is the depth-varying
  *multiplicative* fixed-shape cousin).
- Pass/fail bar (from `idea.md`):
  - **WIN**: `trt_val ≤ ctrl_val − 0.01` AND clears the two-ctrl
    rule. A 0.01 bar is conservative given the 175 reference of
    −0.1585 (the multiplicative depth-varying analog has smaller
    lever amplitude, so a smaller Δ is realistic).
  - **NULL**: `|trt_val − ctrl_val| < 0.01` — closes the fixed-
    shape depth-conditional scale axis decisively. Conditional on
    188 nulling, also closes the entire per-block scale axis
    (learned + fixed both fail).
  - **DRIFT**: `trt_val > ctrl_val + 0.01` — sharpen-early past
    the locality-rewarding optimum; the multiplicative side may
    not reward locality as strongly as 175's additive side did.
  - **CONDITIONAL**: if 188-qk-rms-scaling reports a WIN ≥ −0.005
    before 193 is committed to the queue, redirect 193 to a
    different axis (188 already captured the depth-conditional
    scale on the *learned* axis; 193 is the *fixed-shape* control
    and is only informative as a baseline-vs-188 comparison). The
    runner's commit-handover should consult `autoresearch/closed.md`
    / `baseline-cache.json` for 188's status before queueing 193.
- Build-smoke verification (CPU, no torchtune dependency locally):
  - `MinimalLLM(Tiny1M3MConfig())` constructs: 949,056 params,
    no `tau_b` Buffer on any MHA (flag off path is bit-identical
    to no-flag baseline; 0 overhead).
  - `MinimalLLM(Tiny1M3MBlockTempConfig())` constructs: 949,056
    params + 12 Buffer floats, `tau_b = [0.7, 0.71, 0.75, 0.80,
    0.88, 0.96, 1.04, 1.12, 1.20, 1.25, 1.29, 1.30]` for blocks
    0..11 — matches the r2 sign-convention schedule in `idea.md`.