# Plan — 165 k-only-norm

## Flag
- `use_k_only_norm: bool` (default `False`) on `LLMConfig`.
  - Field: `configs/llm_config.py:587` (sits next to 162's
    `use_q_only_norm` at :577 — explicit-asymmetric sibling).
- A/B subclass: `Tiny1M3MKOnlyNormConfig(Tiny1M3MConfig)` (newly
  added at end of `configs/llm_config.py` near :5322),
  `@dataclass`-decorated with `use_k_only_norm: bool = True`.
  Imported directly from `configs.llm_config` by
  `_arq_165-k-only-norm.py` — the dataclass-inheritance pitfall
  (159/155/161) precludes the bare
  `class C(Tiny1M3MConfig): use_k_only_norm: bool = True` annotation
  pattern.

## Change
- `configs/llm_config.py` — `use_k_only_norm: bool = False` already
  present at line 587, with a comment block documenting the
  RMSNorm-weight=1-bias=0 step-0 rescaling trade-off (spec-allowed
  fp32 max-abs-diff < 1e-3 tolerance, same as 159/162).
- `configs/llm_config.py` — `Tiny1M3MKOnlyNormConfig` already present
  at line 5322, `@dataclass`-decorated, `use_k_only_norm: bool = True`.
  This is the class the daemon imports as `C` for the build-smoke
  and run.
- `models/layers.py` — `MultiHeadAttention.__init__` already accepts
  `use_k_only_norm: bool = False` (declared at line 877 alongside
  the 162 knob) and, when on, registers
  `self.k_only_norm = nn.RMSNorm(self.d_k, eps=1e-6)` at lines
  1075-1077. A mutual-exclusion assert with 162 is already in place
  at lines 1084-1087 ("162 + 165 re-derives the symmetric 016 path;
  pick one.").
- `models/layers.py` — `forward()` extended the three QK-norm
  branches to handle `use_k_only_norm` as the K-mirror of 162:
  - `use_nope or use_cope` branch (no rotary): `K = self.k_only_norm(K)`;
    Q untouched.
  - `use_qk_norm_post_rope`: `Q = self.rotary(Q)`,
    `K = self.k_only_norm(self.rotary(K))`; Q untouched.
  - default (pre-RoPE norm): `Q = self.rotary(Q)`,
    `K = self.rotary(self.k_only_norm(K))`; Q untouched.
  Same 3-branch structure threaded through the MoA `extra_K`
  branch.
- `models/layers.py` — `TransformerBlock.__init__` accepts
  `use_k_only_norm: bool = False` (declared next to 162's kwarg at
  line 3144) and pass-throughs it to the MHA constructor at line 3554.
- `models/llm.py` — `MinimalLLM.__init__` reads
  `self.use_k_only_norm = getattr(config, "use_k_only_norm", False)`
  at line 441 (next to the 162 read at :440) and pass-throughs it
  into both `TransformerBlock(...)` constructors at lines 686 and
  942 (siblings of the 162 pass-throughs).

Step-0 identity (flag OFF): no `k_only_norm` module is registered,
no branch is taken, baseline path bit-identical. The 162 `q_only_norm`
module and branches are also untouched, so flag OFF + 162 OFF is the
existing 016 baseline with both unused knobs set `False`.

CPU build-smoke (the daemon's `MinimalLLM(C())` check):
- `MinimalLLM(Tiny1M3MConfig())` → 949,056 params ✓
- `MinimalLLM(Tiny1M3MKOnlyNormConfig())` → 949,248 params ✓
- Delta = +192 (one `nn.RMSNorm(d_k=16)` weight per block × 12
  blocks; bias is `None` by default — `nn.RMSNorm` has no bias).
- Module-registration check: `k_only_norm` is registered at
  `transformer_blocks.{0..11}.attention.k_only_norm`;
  `q_only_norm` is NOT registered (mutual exclusion held).
- Forward check: `MinimalLLM(C())` constructs and runs a 16-token
  forward on CPU without error.

Step-0 identity (flag ON): the new `k_only_norm` `nn.RMSNorm`
parameter consumes RNG state during model construction, AND at the
first forward pass rescales K by `1 / sqrt(mean(K²))` per token per
head. The former is the same caveat as every other flag-on path; the
latter is the **accepted rescaling trade-off** documented in
`idea.md` and `review.md` (spec allows fp32 max-abs-diff < 1e-3,
159/162 precedent).

## Control
- A: `configs.llm_config.Tiny1M3MConfig` (seed 42, flag OFF) — bare
  tier config. The daemon owns this control via
  `autoresearch/bin/baseline.sh`.
- B: `_arq_165-k-only-norm.py` (seed 42, flag ON) — same tier,
  `use_k_only_norm=True`. The `C` class is the build-smoke target.
- Tier: `tiny1m3m` (0.94M params, 3M tokens). Seed 42 only
  (one-seed-only rule).

## Cost
- Params: + `d_k` = +16 per block (weight only — `nn.RMSNorm` has no
  bias), × 12 blocks = +192 = +0.02% of 0.94M (verified by counting
  `MinimalLLM(C()).parameters()`).
- FLOPs: 1·d_head per token per forward (a single RMSNorm pass on K)
  × 12 layers × ~250 tokens/step = ~negligible.
- Memory: + 192 floats; activation memory unchanged.

## Run
- Artifact: `_arq_165-k-only-norm.py` (repo root) imports
  `Tiny1M3MKOnlyNormConfig as C` from `configs.llm_config` and
  dispatches `train_llm.main()` with `--config_class __main__.C
  --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- Descriptor: `autoresearch/ideas/165-k-only-norm/run.json` —
  `{"name": "165-k-only-norm", "arq_file": "_arq_165-k-only-norm.py",
  "job_timeout": "12m"}`.
- Daemon (`autoresearch/bin/queue-daemon.sh`): scp's the stub, runs
  the CPU build-smoke (`MinimalLLM(C())` constructs without error),
  then launches the run in the `arq` tmux.
- **Pass/fail bar** (mirror of 162 — the two experiments are the
  clean 3-way orthogonal axis test, so the bars must be shape-matched
  to keep the attribution call fair):
  - **PASS (K-side carries 016):** treatment val ≤ 016-qk-norm's
    recorded val by ≥ 0.005 (the same shape as 016's own bar). Win
    message: "K-only matches or beats the symmetric QK-norm win ⇒
    K-side normalization is the binding axis."
  - **NULL (Q-side / symmetry carries 016):** |treatment val − ctrl
    val| < 0.005 against the **bare no-norm ctrl** (not 016). Null
    message: "K-only ≡ no-norm at 0.94M ⇒ 016's WIN came from the
    Q-side normalization or the symmetry, not from K."
  - **DRIFT (lever harmful):** treatment val ≥ ctrl + 0.005. Drift
    message: "the K-side rescaling disturbs a useful prior."
  - Crash / NaN / OOM → `needs-recode` (round 1, inside budget).
- Reference: 016-qk-norm (the closed symmetric WIN at tiny1m3m,
  Δ ≈ -0.014 vs both ctrls, pass-bar -0.005 cleared ~3×).
- Read-out: the daemon greps `Final Val Loss:` from the per-idea log
  (`/root/arq/logs/165-k-only-norm.log` on the box; the local
  equivalent is whatever path the queue-daemon assigns). Pass/fail
  is then arithmetically applied via `autoresearch/bin/baseline.sh
  verdict` against the recorded ctrl mean ± band.
