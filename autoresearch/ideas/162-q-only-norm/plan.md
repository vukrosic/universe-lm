# Plan — 162 q-only-norm

## Flag
- `use_q_only_norm: bool` (default `False`) on `LLMConfig`.
  - Field: `configs/llm_config.py:558` (sits next to the closed 016
    `qk_norm_type` lever — same family, explicit-asymmetric sibling).
- A/B subclass: `Tiny1M3MQOnlyNormConfig(Tiny1M3MConfig)` (newly
  added at end of `configs/llm_config.py`), `@dataclass`-decorated
  with `use_q_only_norm: bool = True`. Imported directly from
  `configs.llm_config` by `_arq_162-q-only-norm.py` — the dataclass
  inheritance pitfall (159/155/161) precludes the bare
  `class C(Tiny1M3MConfig): use_q_only_norm: bool = True` annotation
  pattern.

## Change
- `configs/llm_config.py` — adds `use_q_only_norm: bool = False` at
  line 558, with a comment block documenting the trade-off (RMSNorm
  weight=1, bias=0 ⇒ step-0 rescaling, spec-allowed fp32 max-abs-diff
  < 1e-3 tolerance, same as 159-emb-layernorm).
- `models/layers.py` — `MultiHeadAttention.__init__` accepts
  `use_q_only_norm: bool = False` (declared at line 825 alongside the
  closed 016 knob) and, when on, registers `self.q_only_norm =
  nn.RMSNorm(self.d_k, eps=1e-6)` at lines 1011-1013. In `forward`,
  the lever overrides the symmetric QK-norm path in all three
  branches (no separate `k_norm` call):
  - `use_nope or use_cope` branch (no rotary): `:2021-2022`
    `Q = self.q_only_norm(Q)`; K untouched.
  - `use_qk_norm_post_rope`: `:2027-2029`
    `Q = self.q_only_norm(self.rotary(Q))`; `K = self.rotary(K)`,
    K untouched.
  - default (pre-RoPE norm): `:2034-2036`
    `Q = self.rotary(self.q_only_norm(Q))`; `K = self.rotary(K)`,
    K untouched.
  The flag is read **before** the symmetric `q_norm/k_norm` branch so
  `use_q_only_norm=True` short-circuits 016's path entirely — they are
  mutually exclusive at this site.
- `configs/llm_config.py` — adds `Tiny1M3MQOnlyNormConfig
  (Tiny1M3MConfig)` `@dataclass` subclass with
  `use_q_only_norm: bool = True`. This is the class the daemon
  imports as `C` for the build-smoke and run.
- `models/llm.py` — `MinimalLLM.__init__` reads `self.use_q_only_norm
  = getattr(config, "use_q_only_norm", False)` at line 445 and
  pass-throughs it into both `TransformerBlock(...)` constructors at
  lines 680 and 935 (no new parameter, no new module on the model).

Step-0 identity (flag OFF): no `q_only_norm` module is registered,
no branch is taken, baseline path bit-identical. Verified locally:
`MinimalLLM(Tiny1M3MConfig())` ≡ `MinimalLLM(Tiny1M3MConfig())` to 0.0
max-abs-diff on a 16-token forward at seed 42 (the flag-off path is
literally the existing 016 baseline with the unused-knob still set
False — no extra state).

Step-0 identity (flag ON): the new `q_only_norm` `nn.RMSNorm`
parameter consumes RNG state during model construction, AND at the
first forward pass rescale Q by `1 / sqrt(mean(Q²))` per token per
head. The former is the same caveat as every other flag-on path; the
latter is the **accepted rescaling trade-off** documented in `idea.md`
and `review.md` (spec allows fp32 max-abs-diff < 1e-3, 159-emb-
layernorm precedent).

## Control
- A: `configs.llm_config.Tiny1M3MConfig` (seed 42, flag OFF) — bare
  tier config. The daemon owns this control via
  `autoresearch/bin/baseline.sh`.
- B: `_arq_162-q-only-norm.py` (seed 42, flag ON) — same tier,
  `use_q_only_norm=True`. The `C` class is the build-smoke target.
- Tier: `tiny1m3m` (0.94M params, 3M tokens). Seed 42 only
  (one-seed-only rule).

## Cost
- Params: + `d_k` = +64 at tiny1m3m (one `nn.RMSNorm(d_head)` per
  block, 12 blocks ⇒ +768 = +0.08% of 0.94M).
- FLOPs: 1·d_head per token per forward (a single RMSNorm pass on Q)
  × 12 layers × ~250 tokens/step = ~negligible.
- Memory: + `d_k` floats per block × 12 blocks = +768 floats;
  activation memory unchanged.

## Run
- Artifact: `_arq_162-q-only-norm.py` (repo root) defines top-level
  `class C(Tiny1M3MConfig): use_q_only_norm: bool = True` and
  dispatches `train_llm.main()` with `--config_class __main__.C
  --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- Descriptor: `autoresearch/ideas/162-q-only-norm/run.json` —
  `{"name": "162-q-only-norm", "arq_file": "_arq_162-q-only-norm.py",
  "job_timeout": "12m"}`.
- Daemon (`autoresearch/bin/queue-daemon.sh`): scp's the stub, runs
  the CPU build-smoke (`MinimalLLM(C())` constructs without error),
  then launches the run in the `arq` tmux.
- **Pass/fail bar** (tightened per `review.md` r1 — the taste headline
  of "~half of 016's gain ~-0.007" sits inside the ±0.04 noise band
  and is not detectable on a single ctrl, so we frame the bar as a
  match-or-beat comparison against the 016 baseline **and** an
  explicit null hypothesis):
  - **PASS (Q-side carries 016):** treatment val ≤ 016-qk-norm's
    recorded val by ≥ 0.005 (the same shape as 016's own bar). Win
    message: "Q-only matches or beats the symmetric QK-norm win ⇒
    Q-side normalization is the binding axis."
  - **NULL (K-side / symmetry carries 016):** |treatment val − ctrl
    val| < 0.005 against the **bare no-norm ctrl** (not 016). Null
    message: "Q-only ≡ no-norm at 0.94M ⇒ 016's WIN came from the
    K-side normalization or the symmetry, not from Q."
  - **DRIFT (lever harmful):** treatment val ≥ ctrl + 0.005. Drift
    message: "the rescaling disturbs a useful prior."
  - Crash / NaN / OOM → `needs-recode` (round 1, inside budget).
- Reference: 016-qk-norm (the closed symmetric WIN at tiny1m3m,
  Δ ≈ -0.014 vs both ctrls, pass-bar -0.005 cleared ~3×).
