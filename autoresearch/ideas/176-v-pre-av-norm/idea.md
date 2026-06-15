---
id: 176-v-pre-av-norm
status: running
round: 1
updated: 2026-06-15T04:59:49Z
transfer-risk: med
plain: Apply RMSNorm to the value vectors (V) before they get multiplied by attention weights, with learnable gain starting at 1.0 so step-0 is identical to the baseline.
---

# 176 — Pre-AV V RMSNorm (RMSNorm on Value Vectors Before Attention-Output Matmul)

## Source
- **Primary cite (V-pre-AV normalization primitive):** Wortsman et al.
  2023, arXiv:2309.14322 — Stability AI report on V-side normalization
  as a Scale-and-Shift primitive. The already-existing 029-v-norm
  LayerNorm override cites this paper at `models/layers.py:1017`.
  176 is the per-head α-gated RMSNorm+γ-gain parameterization of
  Wortsman 2023's V-norm primitive: closed-029 is "V-pre-AV LayerNorm,
  no gate, no per-head gain"; 176 layers a per-head α-gate (init 0)
  and a per-head γ-gain (init 1) on top of the same tensor location,
  producing a smooth learnable blend.
- Mechanism is the natural V-side analog of 016-qk-norm (RMSNorm on Q
  and K before QK^T, **WIN** at tiny1m3m with Δ=-0.0138/-0.0185). The
  lever family is "normalize the attention-input tensors before they
  interact": QK-norm normalizes Q and K pre-softmax; 176 normalizes V
  pre-AV.
- In-repo context:
  - **016-qk-norm WIN** at tiny1m3m (Δ -0.0138/-0.0185, ≫ gap 0.0047).
    The win tells us pre-interaction normalization is a productive
    lever family at our tier.
  - **162-q-only-norm NULL** (Δ -0.0043 inside band) and
    **165-k-only-norm NULL** (Δ -0.0293, plan PASS bar missed by 0.031).
    The two single-side nulls together with the joint WIN tell us the
    QK-norm WIN was carried by the *joint* QK symmetry, not by either
    side alone. V-side is a separate axis: V has no symmetry partner
    in the pre-AV stage (only V → AV), so V-norm can act independently.
  - **160-rms-gain-per-head NULL** (Δ -0.0023 inside |Δ|<0.005 plan
    band): post-AV per-head gain is closed. 176 is *pre-AV* per-head
    V norm — distinct tensor location.
- V normalization is not in `closed.md`. The lever is novel at our
  tier and a natural extension of the 016 family.

## Mechanism
Standard attention:
1. Q = x @ W_Q, K = x @ W_K, V = x @ W_V.
2. scores = softmax(QK^T / √d_k).
3. out = scores @ V (per head) → concat → x @ W_O.

With V pre-AV RMSNorm:
1. Q, K, V same as above.
2. **V ← RMSNorm(V) · γ_h** where `γ_h ∈ ℝ^{d_k}` is a per-head learnable
   gain (init 1.0). RMSNorm: `V ← V / √(mean(V²) + ε) ⊙ γ_h`. No
   mean subtraction (RMS, not LayerNorm).
3. scores, out same as above.

**Step-0 byte-identical (max-abs-diff = 0.0)**: `γ_h = 1` for all heads
(init) ⇒ `V ← V / √(mean(V²) + ε) ⊙ 1 = V / √(mean(V²) + ε)`. This
is NOT byte-identical — V is rescaled. So `γ_h = 1` alone doesn't
give us identity.

**Identity init via α-gate**: make the entire RMSNorm *optional* and
gate by a learnable scalar `α_h` init 0:
```
V_out = (1 − α_h) · V_in + α_h · RMSNorm(V_in) · γ_h
```
With `α_h = 0` for all heads (init), `V_out = V_in` exactly —
**byte-identical (max-abs-diff = 0.0)**. The lever's knob is α_h:
pushing it positive introduces V normalization; α=1 gives full
`RMSNorm(V)·γ_h`. The transition is smooth and the lever can be
partial. (Other parameterizations — sigmoid init −10, straight-through
clamp — also yield bit-identical forward at step 0 but add complexity
without buying anything at this tier; relu is the simplest.)

**Locked parameterization**: `α_h = relu(α_raw_h)` with
`α_raw_h ∈ ℝ^{H}` init 0, `γ_h ∈ ℝ^{H × d_k}` init 1. The relu
constrains the gate to grow only (lever can introduce V-norm, never
reverse it). The forward is `V_out = (1 − relu(α_raw_h)) · V_in +
relu(α_raw_h) · RMSNorm(V_in) · γ_h`, computed per head along the
`d_k` axis.

## Design sketch
- **Files**:
  - `models/layers.py` — `MultiHeadAttention.__init__`: add
    `use_v_rmsnorm: bool = False`. Add
    `self.v_rmsnorm_alpha = nn.Parameter(torch.zeros(n_heads))` (init 0)
    and `self.v_rmsnorm_gain = nn.Parameter(torch.ones(n_heads, d_k))`
    (init 1.0). Wire into the existing parameter list.
  - `MultiHeadAttention.forward`: after the V projection, apply
    `V = (1 − relu(α_h)) · V_in + relu(α_h) · RMSNorm(V_in) · γ_h`
    per head, then continue with scores @ V as usual.
  - `configs/llm_config.py` — add `use_v_rmsnorm: bool = False`.
  - `models/llm.py` — pass through the kwargs at the existing MHA
    construction sites.
- **Config flag**: `use_v_rmsnorm: bool = False` (off by default,
  baseline path bit-identical).
- **Step-0 byte-identical**: at init, `α_h = 0` for all heads
  (relu(0) = 0) ⇒ `V = (1 − 0) · V_in + 0 · ... = V_in` exactly ⇒
  **byte-identical to baseline at step 0 (max-abs-diff = 0.0)**.
- **Intuition (why it might lower val loss)**: 016-qk-norm WIN
  (pre-softmax normalization) is the strongest evidence that
  pre-interaction normalization helps at this tier. V is the
  third tensor in the attention pipeline (Q, K are inputs to
  softmax; V is the values that get weighted). Pre-AV V
  normalization controls the *magnitude* of the attention output
  before the W_O projection, which is a fresh axis the closed
  per-head-attention-shape levers (152, 155, 160, 162, 165, 166)
  don't cover. The QK-norm WIN tells us normalization-on-attention-
  tensors is a productive family; V is the missing tensor.
- **LoC**: ~30 (parameter + apply-RMSNorm-with-gate). Uses the
  existing `nn.RMSNorm` from `torch.nn` (PyTorch ≥2.4 already used
  in this repo per `models/layers.py:340, 879`).

## Scale evidence
- 016-qk-norm WIN at tiny1m3m is the *closest* scale evidence:
  Δ -0.0138/-0.0185 tells us pre-interaction normalization helps
  at this tier. 176 extends the family to V.
- V-side normalization is less commonly cited in the literature
  than Q/K normalization. The closest analog is "QKNorm" applied
  to all three of Q, K, V in some recent models (Chameleon, etc.),
  but V normalization is rarely isolated.
- 160-rms-gain-per-head NULL (post-AV gain) suggests the post-AV
  axis is closed. 176 is pre-AV, a structurally different axis.
- 162-q-only-norm NULL and 165-k-only-norm NULL both tell us
  single-side Q/K normalization doesn't fire. V has no symmetry
  partner so this argument doesn't apply.
- **Transfer risk: med** (extends 016 to V; mechanism is novel
  at our tier and not directly validated at ≥100M. The bet is
  that 016's win extends to V because V has the same role as
  Q/K in the attention pipeline: it's an attention-input tensor
  that gets combined linearly with the others.)

## Why it's worth a slot
The bet: 016's WIN says pre-softmax normalization helps; 176 tests
the V-side extension. V is the only attention-input tensor not yet
tested for normalization. The gate-init-α=0 trick gives clean
bit-identity at step 0. We expect Δval ∈ [-0.005, -0.020] (modest,
similar to the 016 family). A null tells us V is structurally
different from Q/K (Q and K interact via dot product, V is just
weighted-averaged — there's no "norm of V affects the geometry"
argument) and the pre-interaction normalization family is exhausted
at 0.94M. A win unlocks the V-norm axis at Phase-2 where each head
has more gradient signal to develop a useful gate-α value.

## Plan

**Files changed**

- `configs/llm_config.py` — add `use_v_rmsnorm: bool = False` field on
  `LLMConfig` (sibling of `use_v_layernorm` at line 1807; default off,
  baseline path bit-identical). Add `@dataclass
  Tiny1M3MVPreAVNormConfig(Tiny1M3MConfig)` with
  `use_v_rmsnorm: bool = True` (sibling of `Tiny1M3MLayerNorm*` configs
  near line 2690 — must use the `@dataclass` decorator, NOT a bare
  `class C(...):` annotation, per the 162/165/155/161 precedent that
  bare-class annotation breaks dataclass field inheritance).
- `models/layers.py` — add `use_v_rmsnorm: bool = False` kwarg to
  `MultiHeadAttention.__init__` at line 1026 (sibling of
  `use_v_layernorm` at the same line). When on, register
  `self.v_rmsnorm_alpha = nn.Parameter(torch.zeros(H))` (init 0) and
  `self.v_rmsnorm_gain = nn.Parameter(torch.ones(H, d_k))` (init 1)
  next to the existing `v_norm` construction at line 1324
  (slot a parallel `elif use_v_rmsnorm:` arm). Apply the gate-α+γ
  after `v = self.W_V(x)` and before `scores @ v` (i.e. before any
  RoPE, since V is *not* rotary in the current MHA). The
  `self.use_v_norm` site at line 2852 is the construction template —
  slot the gate as `if self.use_v_rmsnorm: V = self._apply_v_rmsnorm(V)`
  *after* the closed-#92 `v_norm` site and *before* the AV matmul.
  Verify vs. the existing v-residual site at line ~2848 if
  `021-value-residual` is also on, and the `self.alpha_av` site at
  line 1418/3521 if AV-output carry is on (the gate must run on V
  *before* the AV product; AV-output carry is post-product and
  doesn't interact).
- `models/layers.py` — `TransformerBlock.__init__` adds
  `use_v_rmsnorm: bool = False` kwarg (sibling of `use_v_layernorm`
  at line 3740), reads it from kwargs, passes it into the MHA
  constructor at lines 4204-4206.
- `models/llm.py` — read `self.use_v_rmsnorm = getattr(config,
  "use_v_rmsnorm", False)` at line ~440 (sibling of
  `self.use_v_layernorm`); thread it into both `TransformerBlock(...)`
  constructor sites at lines ~685 and ~941.

**Mutual exclusion asserts** (top of `MultiHeadAttention.forward`,
mirror the `use_cope ∧ use_qk_norm_post_rope` assertion at line 1948):
```
assert not (self.use_v_rmsnorm and self.use_v_layernorm), \
    "use_v_rmsnorm and use_v_layernorm are mutually exclusive (V-pre-AV)"
assert not (self.use_v_rmsnorm and self.use_v_norm), \
    "use_v_rmsnorm and closed-#92 v_norm_type are mutually exclusive"
```
The `use_v_mix_conv` check is optional (clean independence insurance,
not a hard requirement — only add if the impl is trivially free).

**Flag name**: `use_v_rmsnorm: bool = False` (off by default).

**Step-0 identity**: at init, `α_raw_h = 0` for all heads ⇒
`relu(0) = 0` ⇒ `V_out = (1 − 0) · V_in + 0 · ... = V_in` *exactly* ⇒
**byte-identical to baseline at step 0 (max-abs-diff = 0.0)**. Verify
in the runner with a fp32 max-abs-diff test (assert
`trt_step0_logits == ctrl_step0_logits` byte-exact).

**Param count**: H=4, d_k=16, n_layers=12. Per block:
`H × (1 α + d_k γ) = 4 × (1 + 16) = 68` params. Across 12 blocks:
`12 × 68 = 816` params. That's +0.087% of the 0.94M baseline
(949,056 params). Well under the per-lever budget.
(Not 204 as a hypothetical taste-summary might say — re-derive from
H=4, d_k=16, n_layers=12, not H=12.)

**CPU build-smoke** (the daemon's `MinimalLLM(C())` check):
- `MinimalLLM(Tiny1M3MConfig())` → 949,056 params (baseline).
- `MinimalLLM(Tiny1M3MVPreAVNormConfig())` → 949,872 params
  (+816 = 12 × 68).
- The 176 build-smoke must call `MinimalLLM(C())` cleanly before
  any GPU time is spent (the daemon's `_box_smoke.py` wraps this).

**Run command**:
- Artifact: `_arq_176-v-pre-av-norm.py` at repo root, imports
  `Tiny1M3MVPreAVNormConfig as C` from `configs.llm_config`, dispatches
  `train_llm.main()` with `--config_class __main__.C --seed 42
  --dataset_path processed_data/pretrain_1B --warmup false`
  (mirror the 162/165/169/170 `_arq_*.py` pattern).
- Job: `python _arq_176-v-pre-av-norm.py` with `JOB_TIMEOUT=12m`
  (tiny1m3m runs in ~2-6 min; the cap keeps a hung treatment from
  burning the box for 40 min).
- Descriptor: `autoresearch/ideas/176-v-pre-av-norm/run.json` —
  `{"name": "176-v-pre-av-norm", "arq_file": "_arq_176-v-pre-av-norm.py",
   "job_timeout": "12m"}`.
- Val loss is read from the run's log via
  `grep "val_loss" ~/arq/logs/176-v-pre-av-norm.log`.

**LoC budget**: ~30 lines total (parameter + apply-RMSNorm-with-gate +
3-line assert). Well under the 200 LoC cap. Uses the existing
`nn.RMSNorm` from `torch.nn` (PyTorch ≥2.4 already used in this repo
per `models/layers.py:340, 879`).

## Pass / fail bar

**Control**: unmodded `Tiny1M3MConfig` (no V-norm baseline).
**Cached baseline val_mean** ≈ 6.4447, noise_band ≈ 0.0488
(see `autoresearch/baseline-cache.json`).

- **WIN (V-norm helps at 0.94M):** treatment val ≤ val_mean − 0.005
  (i.e. ≤ 6.4397) **AND** clears the plan bar of Δval ≤ −0.005 vs the
  bare no-norm ctrl. Mirrors the 016-qk-norm bar: clears the
  ±0.049 noise band by ≥10× (one-seed; sub-noise is inconclusive,
  not real). Win message: "per-head α-gated V-norm lowers val at
  0.94M ⇒ pre-AV normalization extends to V."
- **NULL (V-norm axis closed at this tier):** |treatment val −
  val_mean| < 0.005 (i.e. inside [6.4397, 6.4497]). Null message:
  "per-head α-gated V-norm is indistinguishable from the no-norm
  baseline at 0.94M ⇒ pre-AV V is structurally inert at our tier,
  or the per-head gate can't accumulate enough gradient signal
  in 3k steps."
- **DRIFT (lever harmful):** treatment val ≥ val_mean + 0.005
  (i.e. ≥ 6.4497). Drift message: "the V-norm rescaling disturbs
  a useful prior (e.g. learned V magnitudes encode positional
  information)."
- Crash / NaN / OOM → `needs-recode` (round 1, inside budget).
- Sub-noise (|Δval| < 0.005 but not DRIFT) is INCONCLUSIVE on one
  seed per the one-seed-only rule — do **not** re-run with extra
  seeds; the next reviewer adjudicates and may call for a
  follow-up lever that targets the gate differently (e.g. init
  α_raw_h to a small positive constant rather than 0, so the gate
  is open from step 1).
