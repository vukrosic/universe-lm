# Plan — 196-ffn-glu-mish (MishGLU FFN)

## Flag
`use_mish_glu: bool = False` (default OFF).
- Defined on `LLMConfig` in `configs/llm_config.py:373`.
- Threaded through `models/llm.py` as `self.use_mish_glu = getattr(config, "use_mish_glu", False)` (mirrors the `use_swiglu_ffn` pattern).
- Passed to `TransformerBlock.__init__` at both call sites in `models/llm.py` (YOCO upper-half + standard stack) — the YOCO upper-half is unaffected for `tiny1m3m` (no YOCO there), but the pass-through is in place.
- Dispatched in `TransformerBlock.__init__` (the FFN-replacement cascade) — sits AHEAD of `use_soft_moe` / `use_switch_ffn` / `use_ttt_ffn` / `ffn_variant` branches so it isn't silently shadowed, and asserts mutual exclusion with `use_swiglu_ffn` and `ffn_variant="swiglu"`.

## Change
- **`models/components.py`**: added a top-level `mish(x) = x * torch.tanh(F.softplus(x))` helper (Misra 2019) and a `MishGLUFeedForward` module structurally identical to `SwiGLUFeedForward` but with the inner activation swapped from `F.silu` to `mish`. No zero-init on `gate_proj.weight` (per review finding 3 — `mish(0)=0` gives the silence automatically, and an explicit zero-init would mask the gradient signal the lever depends on). Kaiming-uniform init (the standard `nn.Linear` default) is correct.
- **`models/layers.py`**:
  - Added `use_mish_glu: bool = False` kwarg to `TransformerBlock.__init__`.
  - Added the dispatch branch: `elif use_mish_glu: ... self.feed_forward = MishGLUFeedForward(d_model, (2*d_ff)//3, dropout)` (Shazeer 2/3 trick to match SwiGLU's 32,640-param FFN). Asserts `not (use_soft_moe or use_switch_ffn or use_ttt_ffn)`, `not use_swiglu_ffn`, and `ffn_variant != "swiglu"`.
- **`models/llm.py`**: added `self.use_mish_glu = getattr(config, "use_mish_glu", False)` next to the existing `self.use_swiglu_ffn` line, and added `use_mish_glu=self.use_mish_glu` to both `TransformerBlock` call sites (YOCO upper-half + standard block). No `_init_weights` re-zero step (none needed for MishGLU — finding 3).
- **`configs/llm_config.py`**:
  - Added `use_mish_glu: bool = False` to `LLMConfig` (alongside `use_swiglu_ffn: bool = False`).
  - Added `@dataclass class Tiny1M3MMishGLUConfig(Tiny1M3MConfig): use_mish_glu: bool = True` (alongside `Tiny1M3MSwigluFFNConfig`).

**Step-0 behavior.** Function-level agreement at `z=0`: `mish(0) = silu(0) = 0`. The two activations disagree at `z ≠ 0` (the gate input under Kaiming init is `N(0, σ)`, not identically zero), so the *forward* is not byte-identical to SwiGLU step 0 — only the *activation evaluated at the origin* agrees. The *gradient* is the lever: `dMish/dx|_{x=0} ≈ 0.6` vs `dSiLU/dx|_{x=0} = 0.5` (a 20% boost at the origin, where the gate-input distribution `N(0, 1)` spends ~38% of its mass). Finding 2 in the review explicitly documents this distinction so the smoke test is not mistaken for a fail. (This is the reviewer-approved design — no change to the 2/3-trick or zero-init pattern.)

## Control
- **Control**: `Tiny1M3MConfig` (val ≈ 6.3988 ± 0.04 per `autoresearch/baseline-cache.json`; 619cf8059d37 cached at 6.2403 pinned). Plain baseline path; `use_mish_glu=False` ⇒ the new branch is never taken.
- **Treatment**: `Tiny1M3MMishGLUConfig` (val to be measured; `use_mish_glu=True` ⇒ 3-projection MishGLU with d_ff scaled 2/3 = 170).
- **Seed**: always 42 (one seed only — see `feedback-one-seed-only`).
- **Tier**: tiny1m3m (0.94M / 12L / 92 steps). No screen20m / full ladder / multi-tier reference per review finding.

## Cost
- **Params**: 947,520 (treatment) vs 949,056 (baseline) ⇒ **−1,536 params (−0.16%)** across the 12 blocks. Per-block FFN: 3 × 64 × 170 = 32,640 (treatment, MishGLU) vs 2 × 64 × 256 = 32,768 (baseline, squared_relu). Within the ~0.4% parity claim of the 2/3 trick.
- **FLOPs**: 3 matmuls per FFN (gate, up, down) vs baseline 2 matmuls (up, down) — gate/up are independent so they can fuse; per-token FLOPs are ≈1.5× the baseline FFN FLOPs (the 2/3 trick recovers most of the param overhead, but matmul count is still 3 vs 2). The base `MishGLUFeedForward` itself adds a `tanh(softplus(z))` per gate element — fp32-stable everywhere, no extra matmul.
- **Memory**: identical to SwiGLU (170) — same param count, same activation memory. No new params, no new buffers.

## Run
- **Command**: `python _arq_196-ffn-glu-mish.py` (the daemon's `bin/queue-daemon.sh` reads `autoresearch/ideas/196-ffn-glu-mish/run.json` and invokes this).
- **Tier**: tiny1m3m (seed 42, dataset `processed_data/pretrain_1B`, `--warmup false`).
- **Wall-clock**: matches SwiGLU (170) to within ~5% (same param count, same matmul count); expect ≈6 minutes on the Vast V100 box.
- **Pass/fail bar (from idea.md)**:
  - **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
  - **NULL**: `|trt_val − ctrl_val| < 0.01` (closes the *inner-activation axis* at 0.94M, orthogonal to 170's closed *outer* axis).
  - **DRIFT**: `trt_val > ctrl_val + 0.01`.
- **Cache reference**: champion val ≈ 6.24, baseline cache 6.40.
