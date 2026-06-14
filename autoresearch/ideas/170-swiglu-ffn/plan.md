# Plan — 170 SwiGLU FFN

## Flag
- `use_swiglu_ffn: bool = False` (default off, bit-identical baseline when off)
  - declared in `LLMConfig` at `configs/llm_config.py:276` (the new
    block inserted after `use_relu2_ffn` at line 263)
  - `Tiny1M3MSwigluFFNConfig(Tiny1M3MConfig)` subclass at
    `configs/llm_config.py:2029` (flips `use_swiglu_ffn=True`)
  - plumbed through `MinimalLLM.__init__` (`models/llm.py:411`) and
    passed to every `TransformerBlock(...)` constructor in both the
    YOCO upper-half path (`models/llm.py:711`) and the standard path
    (`models/llm.py:973`)
  - read by `TransformerBlock.__init__` (`models/layers.py:3410`,
    `elif use_swiglu_ffn:` branch at line 3945)

## Change
- `models/components.py` — add `SwiGLUZeroInitFeedForward` class
  (lines 80–130, after the existing `SwiGLUFeedForward`). Three
  projections: `gate_proj`, `up_proj`, `down_proj` (all `bias=False`,
  matching the existing SwiGLU shape). At construction, zeros
  `gate_proj.weight` so `silu(W_gate·x) = silu(0) = 0` ⇒ FFN output is
  exactly 0 at step 0.
- `models/layers.py` — `TransformerBlock.__init__` gains
  `use_swiglu_ffn: bool = False` kwarg; the new `elif use_swiglu_ffn:`
  branch sits AHEAD of `use_soft_moe` / `use_switch_ffn` /
  `use_ttt_ffn` and AHEAD of the existing `ffn_variant == "swiglu"`
  cascade. Mutual-exclusion asserts added (review finding 2) to fail
  loud on conflicting FFN-replacement flags.
- `configs/llm_config.py` — `LLMConfig.use_swiglu_ffn: bool = False`
  field (default off) and `Tiny1M3MSwigluFFNConfig` subclass.
- `models/llm.py` — reads `use_swiglu_ffn` off the config and passes
  through to every `TransformerBlock(...)` call.
- `_arq_170-swiglu-ffn.py` (repo root) — runner stub (top-level `C`
  class subclassing `Tiny1M3MSwigluFFNConfig`, mirrors `_arq_153`).
- `autoresearch/ideas/170-swiglu-ffn/run.json` — daemon descriptor.

Step-0 ≈ baseline when flag is OFF: the new branch is never taken,
no `SwiGLUZeroInitFeedForward` is constructed, `gate_proj` weights
are never allocated. Forward graph is bit-identical to the
no-flag tiny1m3m baseline.

## Control
- A: `Tiny1M3MConfig` (baseline, `use_swiglu_ffn=False`),
     `ffn_variant="squared_relu"` (default), seed 42, tiny1m3m tier
- B: `Tiny1M3MSwigluFFNConfig` (`use_swiglu_ffn=True`),
     everything else inherited, seed 42, tiny1m3m tier
- ctrl cached at `autoresearch/baseline-cache.json` — val ≈ 6.4394 ± 0.04
  on Vast V100 (`baseline.sh verdict` returns `CACHED` for the ctrl)

## Cost
- **Params Δ**: at tiny1m3m (d_model=192, d_ff_baseline=256,
  n_blocks=4): 3 × 192 × 170 = 97,920 per FFN vs baseline
  2 × 192 × 256 = 98,304 ⇒ −384 per FFN, −1,536 total (−0.16% of
  ~0.95M). Shazeer 2/3 trick keeps param count within ~0.4% of
  baseline per FFN.
- **FLOPs Δ**: per token, two extra `d_model × d_ff_swiglu` matmuls
  per block for `gate_proj` and the same number for `up_proj` (these
  are the two inputs to the gate ⊙ value product). At step 0 the
  gate is zero so the `silu(gate) ⊙ up` product is exactly zero ⇒
  the `down_proj` matmul is over a zero activation (waste, but
  mathematically a no-op). The optimizer ramps the gate in over the
  first few hundred steps. At convergence: ~1.0× baseline FFN FLOPs
  (the 2/3 trick compensates for the extra projection).
- **Memory Δ**: ~equal to baseline (fewer params, same activation
  cache shape at the FFN boundary).

## Self-check note: gate zero-init re-applied after `apply(_init_weights)`

`MinimalLLM.__init__` calls `self.apply(self._init_weights)` which
re-initializes every `nn.Linear` weight with `normal_(std=0.02)` —
this *overwrites* the in-constructor zero-init on `gate_proj.weight`
inside `SwiGLUZeroInitFeedForward.__init__`. Without a fix the gate
would NOT be zero at step 0, breaking the ReZero identity (`silu(0)=0
⇒ FFN output = 0`). The fix mirrors the existing pattern for
`use_gated_attn` (`models/llm.py:1278`): after the global
`apply(_init_weights)`, walk every block's FFN and re-zero
`gate_proj.weight` for the `SwiGLUZeroInitFeedForward` class. With
the fix, `ffn.gate_proj.weight.abs().sum() == 0.0` is verified at
CPU construct time.

## Run
- **Command**: `cd /Users/vukrosic/my-life/llm-research-kit-scaling && /venv/main/bin/python _arq_170-swiglu-ffn.py`
- **Tier**: tiny1m3m (only). One seed — **seed 42**, always.
- **Expected wall-clock**: ~3–4 minutes per run (matches the 153
  ReLU² baseline tier cost — the 2/3 trick keeps the FFN FLOPs
  flat; the extra `gate_proj` projection is the same shape as
  `up_proj`).
- **Pass/fail bar** (from idea.md, tightened per review finding 1):
  - **PASS** = Δval ≤ −0.005 vs cached baseline (≈ 6.4394) — i.e.
    final val ≤ 6.4344 — gated gating-structure win
  - **NULL** = |Δ| < 0.01 (within run-to-run noise)
  - **DRIFT** = Δval > +0.01 — gating fails to learn at 0.94M
  - The expected Δval ≈ −0.01 to −0.04 sits within the
    PASS band; the −0.01 end is right at the noise floor (so
    the verifier may classify it as NULL if the noise is
    conservative), the −0.04 end is comfortably PASS.
- **Self-check before release**:
  - Flag OFF path bit-identical to baseline (no new module built
    when `use_swiglu_ffn=False`, asserts not taken) ✓
  - Treatment path actually exercises `SwiGLUZeroInitFeedForward`
    (verified in CPU build-smoke, see `flip.sh` log line for the
    `planning -> needs-run` transition)
  - `plan.md` pass/fail bar matches `idea.md` (PASS ≤ −0.005,
    NULL |Δ| < 0.01, DRIFT > +0.01) ✓
  - `run.json` + `_arq_170-swiglu-ffn.py` emitted; stub defines
    top-level `C`; `MinimalLLM(C())` constructs on CPU without
    error (verified by daemon's build-smoke before GPU allocation) ✓
