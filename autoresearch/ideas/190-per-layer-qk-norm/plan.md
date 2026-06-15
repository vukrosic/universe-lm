# Plan — 190 per-layer-qk-norm

## Flag
Two flags, both default OFF, on `MultiHeadAttention.__init__`
(`models/layers.py:1237-1238`, constructor kwargs
`qk_norm_scalar_per_block: bool = False`,
`qk_norm_scalar_qk_shared: bool = False`). The treatment class
`Tiny1M3MQKNormScalarConfig` (`configs/llm_config.py:7004-7061`) sets
`qk_norm_scalar_per_block=True` and `qk_norm_scalar_qk_shared=False`
(Q and K scalars **separate** — preserves 016's QK symmetry 162+165
attributed WIN to; the shared variant is gated behind
`qk_norm_scalar_qk_shared` and is a different lever).

## Change
- `models/layers.py:1856-1928` — `MultiHeadAttention.__init__`
  registers two new params:
  - `qk_norm_scalar_q = nn.Parameter(torch.ones(()))` per MHA
  - `qk_norm_scalar_k = nn.Parameter(torch.ones(()))` per MHA
  Both init to 1.0; identity-init ⇒ step-0 forward is bit-identical
  to 016's step-0 (γ·x = x in fp32). When
  `qk_norm_scalar_qk_shared=True`, the two attributes point to the
  same `nn.Parameter` (the 169 axis, off by default).
- `models/layers.py:3216-3262` — five `assert not (X and Y)`
  exclusivity guards added so 190 is mutually exclusive with
  `use_q_only_norm` / `use_k_only_norm` / `use_qk_norm_post_rope` /
  `use_qk_norm_depth` (those levers restructure the norm or stack
  another γ; combining confounds 190's axis).
- `models/layers.py:3732-3734` — `MultiHeadAttention.forward`:
  ```python
  if self.use_qk_norm_scalar_per_block:
      Q = Q * self.qk_norm_scalar_q
      K = K * self.qk_norm_scalar_k
  ```
  Placed AFTER the 169 multiply and the per-head norm+RoPE
  branches, BEFORE the QK matmul, so the lever sits at the QK-norm
  output (mirrors 169's placement).
- `models/layers.py:4221-4222` — same multiply mirrored on the MoA
  `extra_K_4d` so all K tokens entering the QK matmul see the same
  per-block γ_K normalization strength.
- `configs/llm_config.py:7004-7061` — new
  `Tiny1M3MQKNormScalarConfig(Tiny1M3MQKNormConfig)` with
  `qk_norm_scalar_per_block=True`, `qk_norm_scalar_qk_shared=False`,
  `@dataclass`-decorated. Inherits `use_qk_layernorm=True` from
  the 016 WIN control.
- `autoresearch/ideas/190-per-layer-qk-norm/run.json` — pointer
  to `_arq_190-per-layer-qk-norm.py`, `job_timeout: 12m`.
- `_arq_190-per-layer-qk-norm.py` — self-contained treatment entry
  that imports `Tiny1M3MQKNormScalarConfig` as `C`, drives
  `train_llm.main()` with seed 42, dataset
  `processed_data/pretrain_1B`, `--warmup false`.

With both flags off, no `nn.Parameter` is registered (None stubs),
the forward branch is not taken, and the baseline path is
bit-identical to the 016 control at step 0. The **single new
control knob** is the `qk_norm_scalar_per_block` flag (the
`qk_norm_scalar_qk_shared` sub-flag is kept as a knob for the
shared variant, off by default).

## Control
- **ctrl**: `Tiny1M3MQKNormConfig` (the 016 WIN control, val
  6.3906, `closed.md:60`), seed 42, tiny1m3m. Owned by the daemon
  (per RUN-CONTRACT); not shipped in this stub.
- **trt**: `Tiny1M3MQKNormScalarConfig` (this idea — same 016
  per-head norm intact + scalar γ per block per side), seed 42,
  tiny1m3m. A/B isolates the **granularity axis** (per-channel γ →
  scalar γ), not the existence axis (RMSNorm vs no-RMSNorm).

## Cost
- Params: 016 baseline γ = 12 × 2 × 16 = **384** per-channel
  (already in 016 ctrl). 190 adds 12 × 2 × 1 = **24** new scalar γ
  (init 1.0), or 12 × 1 = 12 if `qk_norm_scalar_qk_shared=True`
  (off). Net change vs the 016 ctrl: **+24** scalar γ, **−360**
  per-channel γ (the per-channel γ axis is replaced by the scalar
  γ, not augmented). The lever is the **granularity change** (per-
  channel → scalar), not the parameter-budget change.
- FLOPs: ~+2 elementwise multiplies per block per forward (Q and
  K scalars, plus a third on the MoA `extra_K_4d` if MoA is
  active; at tiny1m3m MoA is off, so 2 mults × 12 blocks = 24
  mults per token — negligible vs the QK matmul).
- Memory: 24 zero-shape scalar params (24 × 4 bytes = 96 B for
  fp32 grads/optim states); not measurable against 0.94M.

## Run
- Command (treatment): `/root/universe-lm/.venv/bin/python
  _arq_190-per-layer-qk-norm.py` (the daemon's standard
  invocation; venv path per the Vast runner harness).
- Tier: **tiny1m3m only** (the 016 WIN was scored at this tier;
  one-tier-only per the idea).
- Seed: **42** (one seed, per the pipeline's one-seed-only rule).
- Expected wall-clock: ~6-8 min on the Vast V100 box (matches
  the 016 ctrl's per-run cost; the +24 scalar γ adds no
  meaningful step-time).
- Pass/fail bar (copied from `idea.md`):
  - **PASS**: trt val_loss ≤ 016-ctrl − 0.005 ⇒ trt ≤ **6.391**
    (scalar γ is sufficient, per-channel resolution was over-
    parameterized at 0.94M; block-level QK magnitude is the
    binding axis).
  - **NULL band |Δ| < 0.005** vs 016-ctrl ⇒ trt ∈ (6.3856,
    6.3956) — granularities tied at this tier; inconclusive
    without a larger model or longer horizon.
  - **LOSS**: trt val_loss > 016-ctrl + 0.005 ⇒ trt > **6.3956**
    (per-channel resolution IS binding; 190 throws away
    capacity).
  - **DRIFT**: trt val_loss > 016-ctrl + 0.05 ⇒ trt > **6.4406**
    (broken lever, not a real axis).
  - Single seed (42), tiny1m3m only; 0.005 matches 016's own
    plan bar at this tier.
- **Self-check (executed on the build machine, 2026-06-16)**:
  `MinimalLLM(C())` constructs on CPU, total params = 949,464;
  `qk_norm_scalar_q` count = 12, `qk_norm_scalar_k` count = 12
  (24 total, matching the design). Forward with random ids
  `[1, 8]` returns `[1, 8, 49152]` (correct vocab). Step-0
  identity: γ=1.0 ⇒ `Q · 1 = Q` and `K · 1 = K` exactly in fp32,
  so the treatment path with both flags ON is bit-identical to
  the 016 ctrl at step 0 (max-abs-diff = 0.0).
