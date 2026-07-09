# Plan — 188 per-block QK-rms scaling

## Flag
- `use_qk_rms_scaling: bool = False` on `LLMConfig` (next to
  `use_per_layer_temp` at `configs/llm_config.py:175`). Default OFF
  ⇒ baseline path bit-identical. When ON, the MHA registers a
  per-block `qk_rms_param = nn.Parameter(torch.zeros(1))` and the
  forward multiplies the pre-softmax `Q·K^T/√d_k` scores by
  `qk_rms_param.exp()`.

## Change
- `configs/llm_config.py` — `LLMConfig` (already has the flag at
  `:175`, the dataclass field was added in the r2 cycle). Added
  `Tiny1M3MQKRMSConfig(Tiny1M3MQKNormConfig)` with
  `use_qk_rms_scaling: bool = True` (mirrors the `Tiny1M3MQKNorm
  DepthConfig` pattern at `:6602`).
- `models/layers.py` — `MultiHeadAttention.__init__`:
  - New kwarg `use_qk_rms_scaling: bool = False` (at
    `models/layers.py:1198`, right after `use_qk_norm_depth`).
  - Register `self.qk_rms_param = nn.Parameter(torch.zeros(1))`
    when ON, else `None` (mirrors the `qk_norm_scale` registration
    pattern at `:1761`).
  - In `forward()` (the manual path that the 016 WIN takes), after
    `scores = torch.matmul(Qn, Kn.transpose(-1, -2)) * scale` (at
    `:4217`), multiply `scores = scores * self.qk_rms_param.exp()`
    BEFORE the causal mask and softmax (and before any other
    score-side lever like 152/155/204 etc., so the lever composes
    uniformly with downstream tweaks).
  - Added `or self.use_qk_rms_scaling` to the manual-path forcing
    list (at `:4140`) — `scores * s_l` is a pre-softmax score
    multiply, SDPA's flash kernel fuses QK^T+softmax+AV and can't
    expose the pre-softmax logit for the per-block temperature.
- `models/llm.py` — `MinimalLLM.__init__` reads
  `self.use_qk_rms_scaling = getattr(config, "use_qk_rms_scaling",
  False)` and passes it to both MHA constructor sites
  (`:935`/`:1284`, alongside the `use_qk_norm_depth` pass-through).
- Step-0 byte-identity: `s_param_l = 0` init ⇒ `s_l = exp(0) = 1.0`
  exactly in IEEE 754 ⇒ `scores * 1.0 = scores` byte-identical to
  016-alone at step 0. No tolerance needed.

## Control
- **Control (this run's ctrl)**: `Tiny1M3MQKNormConfig` (the 016
  WIN, val 6.3906, `closed.md`). The daemon owns the baseline; we
  ship the treatment only.
- **Treatment**: `Tiny1M3MQKRMSConfig` (the new
  `use_qk_rms_scaling: bool = True` config that subclasses the 016
  WIN shape, keeping `use_qk_layernorm=True` and adding
  `use_qk_rms_scaling=True`).
- **Seed**: 42 (one seed only — `feedback-one-seed-only`).
- **Tier**: tiny1m3m (0.94M params, 3M tok, single seed 42).

## Cost
- **Params**: +12 scalars (1 per MHA × 12 blocks), +0.0013% of
  0.94M. Negligible.
- **FLOPs**: +1 elementwise multiply per block per forward (one
  `* self.qk_rms_param.exp()` on the `[B, H, T, T]` scores tensor).
  Negligible at tiny1m3m.
- **Memory**: 12 floats total. Negligible.
- **Compile path**: forces the manual attention path (see forcing
  list edit) — SDPA's flash kernel can't do the per-block
  temperature multiply. Manual path is already taken for the 016
  WIN (so this is the canonical path for the chosen control too,
  no perf regression beyond what 016 already pays).

## Run
- Command (GPU box): `_arq_188-qk-rms-scaling.py` with seed 42 and
  `--warmup false` (the daemon's `_box_smoke.py` runs the build
  on CPU first; the actual GPU run is launched by
  `queue-daemon.sh` reading `autoresearch/ideas/188-qk-rms-
  scaling/run.json`).
- Tier: tiny1m3m, seed 42. Expected wall-clock: ≤12m
  (job_timeout=12m in `run.json`).
- Val read: from `autoresearch/remote-results/<run-dir>/trt_*.log`
  — search for the final `val` line emitted by `train.py`.
- Champion reference: `Tiny1M3MQKNormConfig` (016 WIN, val 6.3906,
  `autoresearch/closed.md`).
- Pass/fail bar (from `idea.md`):
  - **WIN**: `trt_val ≤ 6.3906 − 0.005 = 6.3851` AND clears the
    two-ctrl rule.
  - **NULL**: `|trt_val − 6.3906| < 0.01` (i.e., 016 + per-block
    temp ≈ 016 alone).
  - **DRIFT**: `trt_val > 6.3906 + 0.01 = 6.4006` (per-block temp
    re-sharpens past the 016 optimum).
