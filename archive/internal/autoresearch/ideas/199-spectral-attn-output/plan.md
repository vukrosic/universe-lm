# Plan — 199 Spectral-Norm-Bounded W_O Projection (Per-Block Learnable Lipschitz Cap)

## Flag
`use_wo_spectral_cap: bool = False` (default OFF) on `LLMConfig`. Auxiliary
`wo_spectral_cap_pi_iters: int = 1` (power-iteration steps per forward).

Treatment subclass: `Tiny1M3MWOSpectralCapConfig(Tiny1M3MConfig)` at the
bottom of `configs/llm_config.py` (added after the existing 197/207
treatment subclasses, matching the established placement convention). The
subclass overrides `use_wo_spectral_cap: bool = True` and
`wo_spectral_cap_pi_iters: int = 1` (dataclass-decorated so the parent
default is properly overridden — same pitfall as `_arq_161-dyt-temp.py`
and the 197/207 sibling treatment configs).

## Change

### `configs/llm_config.py`
- Add `use_wo_spectral_cap: bool = False` and
  `wo_spectral_cap_pi_iters: int = 1` to `LLMConfig` (with a long
  comment block matching the style of `use_lowrank_wo` at line 1255 and
  `use_lowrank_wv` at line 1281). The comment explains: (a) the
  per-block `γ_l` scalar (init 0 ⇒ `exp(γ_l) = 1`), (b) the
  per-block power-iteration Buffer `u_l ∈ R^{d_model}` and the
  captured `σ_max_init` Buffer (both populated on the first
  forward), (c) the byte-identity guarantee (γ=0 + σ_max_now =
  σ_max_init ⇒ cap factor = 1 ⇒ `W_O_eff == W_O`), (d) the
  asymmetric (clip-only) cap, (e) default off ⇒ baseline path
  bit-identical, (f) the distinction from 128-spectral-decoupling
  (gradient-space) and 160-rms-gain-per-head (post-AV/post-W_O
  magnitude).
- Add the `@dataclass` subclass `Tiny1M3MWOSpectralCapConfig(Tiny1M3MConfig)`
  with `use_wo_spectral_cap: bool = True` and
  `wo_spectral_cap_pi_iters: int = 1`.

### `models/layers.py` — `MultiHeadAttention.__init__`
- Add `use_wo_spectral_cap: bool = False` and
  `wo_spectral_cap_pi_iters: int = 1` kwargs to the MHA constructor
  signature (after the `use_lowrank_wv`/`wv_lowrank_alpha_init` pair at
  line 1430-1431, before the `use_tied_wo_across_blocks` pair).
- When the flag is on:
  - Allocate one 0-dim learnable scalar `wo_spectral_cap_gamma =
    nn.Parameter(torch.zeros(()))` per MHA.
  - Register two Buffers (`persistent=False` so they don't bloat
    `state_dict`): `_wo_pi_u` of shape `[d_model]` (the
    power-iteration vector, lazily seeded) and
    `_wo_pi_sigma_max_init` of shape `[]` (the captured σ_max from
    the first forward).
  - Set `_wo_pi_initialized = False` (a plain Python bool flag
    that flips True after the first forward).
- When off: stub `wo_spectral_cap_gamma = None`, `_wo_pi_u = None`,
  `_wo_pi_sigma_max_init = None`, `_wo_pi_initialized = False` so
  attribute lookups are always safe.

### `models/layers.py` — `MultiHeadAttention.forward` (W_O application site)
- After extracting `w_o = self.qkvo_proj[self.qkv_size:]` and BEFORE
  the 197-Tied-W_O blend (and therefore BEFORE the 171-DropConnect
  mask and 207-W_O-LowRank addition), add the spectral-cap block:
  ```python
  if self.use_wo_spectral_cap:
      with torch.no_grad():
          # Deterministic seed on the first forward only.
          if not self._wo_pi_initialized:
              seed_u = torch.zeros(d_model, ...); seed_u[0] = 1.0
              self._wo_pi_u.copy_(seed_u)
          # Run `wo_spectral_cap_pi_iters` PI steps.
          u = self._wo_pi_u
          for _ in range(self.wo_spectral_cap_pi_iters):
              wu = w_o @ u
              u = wu / (wu.norm() + 1e-12)
          self._wo_pi_u.copy_(u)
          wu_final = w_o @ u
          sigma_max_now = (wu_final.norm() / (u.norm() + 1e-12)).detach()
          # Snapshot σ_max_init on the FIRST forward only — same
          # value as σ_max_now ⇒ cap factor = 1 exactly at step 0.
          if not self._wo_pi_initialized:
              self._wo_pi_sigma_max_init.copy_(sigma_max_now)
              self._wo_pi_initialized = True
      sigma_max_init = self._wo_pi_sigma_max_init
      cap_factor = torch.minimum(
          torch.ones_like(sigma_max_init),
          (sigma_max_init * torch.exp(self.wo_spectral_cap_gamma))
          / (sigma_max_now + 1e-12),
      )
      w_o = w_o * cap_factor
  ```
- PI state is updated under `no_grad` (does NOT consume backward
  graph) but the cap factor on the CURRENT `w_o` IS in the autograd
  graph (because `w_o` is a leaf Parameter slice — gradient flows
  through `w_o * cap_factor` to `w_o`).

### `models/layers.py` — `TransformerBlock.__init__`
- Add `use_wo_spectral_cap: bool = False` and
  `wo_spectral_cap_pi_iters: int = 1` kwargs after the
  `tied_wo_shared=None` parameter.
- Pass-through to the inner MHA at the construction site (after the
  `tied_wo_shared=tied_wo_shared` pass-through).

### `models/llm.py` — `MinimalLLM.__init__`
- Read the flag: `self.use_wo_spectral_cap = getattr(config,
  "use_wo_spectral_cap", False)` and
  `self.wo_spectral_cap_pi_iters = int(getattr(config,
  "wo_spectral_cap_pi_iters", 1))` (alongside the existing
  `self.use_lowrank_wo` reads at line 293).
- Pass-through to every `TransformerBlock` construction in BOTH the
  YOCO upper-half `yoco_upper_blocks` ModuleList AND the standard
  `transformer_blocks` ModuleList (matching the existing pass-through
  convention used by 207-LowRank-WO / 194-LowRank-WV / 197-Tied-WO).

## Control
- **Control**: `Tiny1M3MConfig` (val 6.40 ± 0.04 cached for this box,
  daemon owns the baseline).
- **Treatment**: `Tiny1M3MWOSpectralCapConfig` (this plan, the new
  subclass with `use_wo_spectral_cap=True`).
- **Tier**: tiny1m3m (0.94M params, 3M tokens, 12 layers, d_model=64).
- **Seed**: 42, always. No multi-seed (per the §1 one-seed-only rule).

## Cost
- **Params**: +12 `γ_l` scalars (one per block, init 0) = +12
  params (+0.001% of 0.94M). Treatment is param-superset of control
  (no per-block params removed), so the A/B is parameter-shape,
  not model-size.
- **FLOPs**: per forward, per block, 1 matmul + 1 norm (PI step) +
  1 matmul + 1 norm (σ_max_now estimate) + 1 scalar min + 1
  scalar multiply + 1 element-wise multiply on `[d_model, d_model]`
  for `w_o * cap_factor`. At d_model=64, n_blocks=12: ~12 ×
  (8K + 64 + 8K + 64 + 1 + 1 + 4K) = ~240K FLOPs/step, well within
  the per-step noise band (the SDPA QK^T/AV matmuls dominate).
  <0.5% wall-clock overhead.
- **Memory**: 0 extra activations. +12 persistent Parameters of 1
  float each (48 bytes). +12 `u` Buffers of d_model=64 floats (3 KB
  total) + 12 `sigma_init` Buffers of 1 float each (48 bytes
  total). All Buffers `persistent=False` so they don't bloat
  `state_dict`.
- **Optimizer state**: +12 entries for `γ_l` (AdamW: 24 floats =
  96 bytes total). Sub-noise.

## Run
- **Command**: standard daemon dispatch via
  `_arq_199-spectral-attn-output.py` (see §4b artifact below).
- **Tier / seed**: tiny1m3m / seed 42.
- **Expected wall-clock**: ~6 min (12m `job_timeout` gives plenty of
  headroom; the 196/192/207/197 siblings all completed in this
  window at the same tier).
- **Pass/fail bar** (copied from `idea.md`):
  - **WIN**: `trt_val ≤ ctrl_val − 0.01` AND clears the two-ctrl
    bracket.
  - **NULL**: `|trt_val − ctrl_val| < 0.01` (sub-noise band).
  - **DRIFT**: `trt_val > ctrl_val + 0.01`.
  - Reference: cached ctrl mean 6.40 ± 0.04 (this box); box noise
    floor ±0.01 at 0.94M/3M.
  - **Family null (axis closure)**: 199 null + 160 null + 142
    null + 181 null + 176 null ⇒ post-attention-shape family
    closed at 5 deep at 0.94M.
