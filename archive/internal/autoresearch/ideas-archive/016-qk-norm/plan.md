# Plan — 016 QK-Norm

## Flag
- `LLMConfig.use_qk_layernorm: bool = False` — default OFF, declared
  in `configs/llm_config.py` (right after the `use_moonlight_muon`
  flag block).
- When OFF: the Q/K norms in `MultiHeadAttention` are
  `nn.RMSNorm(d_head)` (the existing baseline path), so the model is
  **bit-identical** to the current control.
- When ON: the Q/K norms are `nn.LayerNorm(d_head)` (γ=1, β=0 init →
  identity at step 0), bounding the per-head logit
  `Q·K/√d_head` to `|·| ≤ √d_head`. The residual-stream norms
  (`norm1`, `norm2`, final `norm`) stay on RMSNorm — the lever is
  strictly the per-head logit bounding, not a residual-stream
  re-centering.

## Change
- `configs/llm_config.py`:
  - Add `use_qk_layernorm: bool = False` to `LLMConfig` (single
    boolean, no extra params, default off).
  - Add `Tiny1M3MQKNormConfig` recipe (inherits `Tiny1M3MConfig`,
    flips the flag). ~9 LoC including the docstring.
- `models/llm.py`:
  - Read `use_qk_layernorm` from `config` via `getattr` (with
    `False` default) so the base `LLMConfig` doesn't need to be
    edited at the call site.
  - Pass it through to `TransformerBlock` in the block
    construction call. ~3 LoC of plumbing.
- `models/layers.py`:
  - Add `use_qk_layernorm: bool = False` to both
    `MultiHeadAttention.__init__` and `TransformerBlock.__init__`.
  - In `MHA.__init__`, compute a single `_qk_use_ln` boolean that
    is `True` when EITHER the global `use_layernorm` OR the new
    QK-specific `use_qk_layernorm` is on, and use it to build
    BOTH `q_norm` and `k_norm` via `make_norm` with
    `use_layernorm=_qk_use_ln`. The override is applied to the
    two `q_norm = make_norm(...)` sites (one for the joint QK
    norm, one for the Q-side tweaks override) so both Q and K
    flip to LayerNorm when the flag fires.
  - Pass `use_qk_layernorm` from `TransformerBlock` to its
    `MultiHeadAttention`. ~10 LoC total.
- Step-0 invariance: `nn.LayerNorm` is identity at step 0 when
  `weight=1, bias=0` (PyTorch default), and the same flag-gated
  path means the baseline run is bit-identical when the flag is
  off (verified with a seeded two-model forward diff = 0).

## Control
- **Control** (`ctrl`): `Tiny1M3MConfig` — Q/K on RMSNorm (the
  current baseline), seed 42.
- **Treatment** (`trt`): `Tiny1M3MQKNormConfig` — Q/K on
  LayerNorm, residual stream still on RMSNorm, seed 42.
- **Tier**: tiny1m3m (3M tokens, ~92 steps).
- **Seed**: 42 (one seed only — per project rule).

## Cost
- **Params Δ**: +2 · n_layers · 2 · d_head LayerNorm gains/biases.
  At `n_layers=12, d_head=16` (tiny1m3m) that's +768 params
  (~0.08% over the ~0.94M baseline). At `n_layers=24, d_head=24`
  (Screen10M20M) it's +2,304 (~0.03% over ~7.7M). Negligible.
- **FLOPs Δ**: +2 per (token, head) LayerNorm in the attention
  pre-softmax path. A few hundred extra FLOPs per token per
  layer — well under 0.01% of the model's total.
- **Memory Δ**: +2 · d_head per head in each layer (LayerNorm
  gain + bias). Negligible.
- **Wall-clock Δ**: ~0% — LayerNorm and RMSNorm have the same
  asymptotic cost (one mean+var reduction + one mul/add per
  element).

## Run
- Tier: tiny1m3m.
- Command: existing runner with `--config Tiny1M3MQKNormConfig`
  (and `Tiny1M3MConfig` for ctrl).
- Seed: 42 (single seed, hard-pinned per project rule).
- Expected wall-clock: ~10–15 min per side on a single H100, in
  line with the other tiny1m3m A/Bs.
- **Pass/fail bar** (from `idea.md` / `review.md`):
  - **PASS** (real win): `trt ≤ ctrl − 0.005` on val_loss at the
    final eval milestone. The taste review puts leverage at the
    low end of the hypothesis range for 6 layers
    (~-0.005 to -0.01).
  - **NULL** (inconclusive, on-noise): `|Δ| < 0.005` — log and
    move on. A clean NULL at 6 layers is itself informative:
    partitions "QK-Norm's benefit is concentrated in deeper
    stacks" from the residual-stream norm-zoo nulls.
  - **DRIFT** (regression): `trt > ctrl + 0.005` — flag and
    reconsider.
- Composition: pairs cleanly with 017 (Sub-LN) in the same
  batch — orthogonal sites (per-head logit vs per-sublayer
  output), so testing both partitions the depth-stability axis.
