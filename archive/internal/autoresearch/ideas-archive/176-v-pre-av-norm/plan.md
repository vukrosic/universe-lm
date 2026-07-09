# Plan — 176 v-pre-av-norm

## Flag
`use_v_rmsnorm: bool = False` on `LLMConfig` (default OFF — baseline
path bit-identical). Treatment subclass `Tiny1M3MVPreAVNormConfig`
sibling of `Tiny1M3MVNormOnQKNormConfig` at
`configs/llm_config.py:2674` with `use_v_rmsnorm: bool = True`.

## Change

- **`configs/llm_config.py:1807`** — add `use_v_rmsnorm: bool = False`
  field on `LLMConfig`, immediately after `use_v_layernorm` and before
  `use_qk_norm_depth`. Sibling of the closed-029 V-LayerNorm flag;
  default OFF ⇒ no Parameter is registered, no branch is taken, the
  baseline forward graph is bit-identical.
- **`configs/llm_config.py:2690`** — add
  `@dataclass class Tiny1M3MVPreAVNormConfig(Tiny1M3MConfig)` directly
  after the existing `Tiny1M3MVNormOnQKNormConfig`, with
  `use_v_rmsnorm: bool = True`. Sibling of the 029 V-Norm subclass.
  Uses the `@dataclass` decorator per the 162/165/155/161 precedent
  (a bare `class C(...):` annotation breaks dataclass field
  inheritance).

- **`models/layers.py:1026`** — add `use_v_rmsnorm: bool = False`
  kwarg to `MultiHeadAttention.__init__` immediately after
  `use_v_layernorm`. Mirrors the closed-029 plumbing shape.

- **`models/layers.py:1326`** — slot a parallel `elif use_v_rmsnorm:`
  arm to the existing `elif use_v_layernorm:` arm (the closed-029
  construction site). When the flag is on and `v_norm_type` is empty
  and `use_v_layernorm` is off, register two new parameters:
  `self.v_rmsnorm_alpha = nn.Parameter(torch.zeros(self.n_heads))`
  (init 0 ⇒ `relu(0) = 0` ⇒ identity gate at step 0) and
  `self.v_rmsnorm_gain = nn.Parameter(torch.ones(self.n_heads,
  self.d_k))` (init 1.0 ⇒ per-head gain = identity at step 0). Also
  set `self.use_v_rmsnorm = True`. Otherwise leave
  `self.use_v_rmsnorm = False` and `self.v_rmsnorm_alpha = None,
  self.v_rmsnorm_gain = None` (attribute lookups stay valid even when
  the flag is off; the forward `if` guard short-circuits).

- **`models/layers.py:2852`** — after the existing closed-#92 /
  closed-029 `if self.use_v_norm: V = self.v_norm(V)` site and before
  the AV matmul, add:
  ```python
  if self.use_v_rmsnorm:
      # 176 — Pre-AV V RMSNorm with per-head α-gate + per-head γ-gain.
      # V_out = (1 − relu(α_raw_h)) · V + relu(α_raw_h) · RMSNorm(V) · γ_h
      # Init α_raw_h = 0, γ_h = 1 ⇒ V_out = V exactly ⇒ byte-identical
      # to baseline at step 0 (max-abs-diff = 0.0).
      alpha = F.relu(self.v_rmsnorm_alpha).view(1, self.n_heads, 1, 1)
      rms = torch.rsqrt(V.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
      V_rms = V * rms * self.v_rmsnorm_gain.view(1, self.n_heads, 1, self.d_k)
      V = (1.0 - alpha) * V + alpha * V_rms
  ```
  Composes with the closed-#92 / 029 `v_norm` site (which sits
  immediately before it) and with the existing
  `use_value_channel_gate` / `use_kda_channel_gate` sites (which sit
  after it but before the AV product).

- **`models/layers.py:2324`** — add three mutual-exclusion asserts at
  the top of `MultiHeadAttention.forward`, sibling of the existing
  `use_cope ∧ use_qk_norm_post_rope` and `use_qk_norm_depth ∧
  use_q_only_norm` assertions:
  ```python
  assert not (self.use_v_rmsnorm and self.use_v_layernorm), (
      "use_v_rmsnorm=True is mutually exclusive with use_v_layernorm=True "
      "(both attach the V-side RMS/Layer norm to the same V tensor before "
      "the AV product; the composition is undefined)."
  )
  assert not (self.use_v_rmsnorm and self.use_v_norm), (
      "use_v_rmsnorm=True is mutually exclusive with the closed-#92 "
      "v_norm_type zoo (both attach a per-head norm to V pre-AV; "
      "explicit v_norm_type wins — use one or the other)."
  )
  assert not (self.use_v_rmsnorm and self.use_v_mix_conv), (
      "use_v_rmsnorm=True is mutually exclusive with use_v_mix_conv=True "
      "(v_mix_conv is a learned conv on V pre-AV; the composition is "
      "not what 176 tests)."
  )
  ```

- **`models/layers.py:3744`** — add `use_v_rmsnorm: bool = False`
  kwarg to `TransformerBlock.__init__`, sibling of
  `use_v_layernorm`. Read from kwargs, pass through to the inner
  `MultiHeadAttention(...)` constructor at `:4206`.

- **`models/llm.py:524`** — add
  `self.use_v_rmsnorm = getattr(config, "use_v_rmsnorm", False)`
  immediately after the existing
  `self.use_v_layernorm = getattr(...)` line. Thread into both
  `TransformerBlock(...)` constructor sites at `:793` (first site,
  near the `use_v_layernorm=self.use_v_layernorm`) and `:1087`
  (second site).

**Step-0 identity**: at init `α_raw_h = 0` for all heads ⇒
`relu(0) = 0` ⇒ `V_out = (1 − 0)·V_in + 0·... = V_in` *exactly* ⇒
**byte-identical to baseline at step 0 (max-abs-diff = 0.0)**.

## Control

- **Control**: unmodded `Tiny1M3MConfig` (no V-norm baseline). The
  daemon owns the ctrl (per `RUN-CONTRACT.md` §"Control is the
  daemon's, not the idea's"); we ship only the treatment stub.
- **Treatment**: `Tiny1M3MVPreAVNormConfig` with
  `use_v_rmsnorm: bool = True`. A/B at tiny1m3m, seed 42, single seed
  per the one-seed-only rule.

## Cost

- **Params**: H=4, d_k=16, n_layers=12. Per block: `H × (1 α + d_k
  γ) = 4 × (1 + 16) = 68` params. Across 12 blocks: `12 × 68 = 816`
  params. That's +0.087% of the 0.94M baseline (949,056 params).
  Well under the per-lever budget.
- **FLOPs**: ~1 RMSNorm per head per token per block (mean of d_k=16
  squares + rsqrt + multiply-by-γ + (1−α)·V + α·V_rms blend).
  Negligible vs the dominant FFN/AV matmul cost.
- **Memory**: two new tensors per MHA of shape `[H]` and `[H, d_k]`,
  both fp32, ~272 bytes/block. Trivial.

## Run

- **Artifact**: `_arq_176-v-pre-av-norm.py` at repo root, imports
  `Tiny1M3MVPreAVNormConfig as C` from `configs.llm_config`, dispatches
  `train_llm.main()` with `--config_class __main__.C --seed 42
  --dataset_path processed_data/pretrain_1B --warmup false` (mirror
  the 162/165/169/170 `_arq_*.py` pattern).
- **Job**: `/venv/main/bin/python _arq_176-v-pre-av-norm.py` with
  `JOB_TIMEOUT=12m` (tiny1m3m runs in ~2-6 min; the cap keeps a hung
  treatment from burning the box).
- **Descriptor**: `autoresearch/ideas/176-v-pre-av-norm/run.json` —
  `{"name": "176-v-pre-av-norm", "arq_file":
  "_arq_176-v-pre-av-norm.py", "job_timeout": "12m"}`.
- **Val loss**: read from `~/arq/logs/176-v-pre-av-norm.log`
  (`grep "val_loss"`).
- **Pass/fail bar** (from `idea.md` §"Pass / fail bar"):
  - **WIN**: treatment val ≤ `val_mean − 0.005` (≤ **6.4397**)
    AND clears the ±0.0488 noise band by ≥10× (one-seed).
  - **NULL**: |treatment val − val_mean| < 0.005 (inside
    [6.4397, 6.4497]).
  - **DRIFT**: treatment val ≥ `val_mean + 0.005` (≥ **6.4497**).
  - **Crash / NaN / OOM** → `needs-recode` (round 1, inside budget).
  - **Sub-noise** (|Δval| < 0.005 but not DRIFT) is INCONCLUSIVE on
    one seed per the one-seed-only rule — do **not** re-run with
    extra seeds.
  - Cached baseline: `val_mean = 6.4447`, `noise_band = 0.0488`
    (per `autoresearch/baseline-cache.json`).

**LoC budget**: ~30 lines total (parameter + apply-RMSNorm-with-gate
+ 3-line assert). Well under the 200 LoC cap. Uses inline RMSNorm
via `torch.rsqrt(V.pow(2).mean(...) + 1e-6)` rather than a fresh
`nn.RMSNorm` module so the per-head α and γ can stay on
`self.v_rmsnorm_alpha` / `self.v_rmsnorm_gain` (the closed-029 / 162 /
165 / 169 parameter style — Parameters are exposed as `self.X` for
diagnostic logging at end of training, the runner's "log α_h
trajectory" taste request).
