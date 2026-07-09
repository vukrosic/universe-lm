# Plan — 189 cosformer-linear-attn

## Flag
- `LLMConfig.use_cosformer: bool = False` (default OFF) +
  `cosformer_gamma_init: float = 0.0` — `configs/llm_config.py`
  (new fields on `LLMConfig`, default off).
- Treatment: a one-line config subclass
  `Tiny1M3MCosFormerConfig(Tiny1M3MConfig)` with
  `use_cosformer: bool = True`. The arq stub
  `_arq_189-cosformer-linear-attn.py` subclasses this so the
  flag is on by default.

## Change
- `models/layers.py`
  - `MultiHeadAttention.__init__`: new kwargs
    `use_cosformer: bool = False`,
    `cosformer_gamma_init: float = 0.0`. When the flag is on,
    `self.use_cosformer = True` is stored but no new Parameter
    is registered on the MHA — γ lives on the model (see below).
    The branch is gated on `self.use_cosformer` and reads
    `cosformer_gamma` from the forward kwargs.
  - `MultiHeadAttention.forward` (signature at line 3349): add
    `cosformer_gamma=None` kwarg. New `elif self.use_cosformer:`
    branch placed **right after** the existing
    `elif self.use_linear_attn:` block (currently at
    `models/layers.py:4448`). The branch computes
    `Q' = cos(Q)`, `K' = exp(cosformer_gamma·K) ⊙ cos(K)`,
    `out = (Q' · (cumsum(K'^T · V))) / (Q' · cumsum(K'))`
    with prefix-sum causal via `[end_idx, start_idx]` — the same
    pattern the `use_linear_attn` branch already uses at
    `models/layers.py:4448-4477`. The denominator is **mandatory**
    and bound in the spec: no skip-flag, no opt-out
    (`out = out / denom.clamp_min(1e-6)`). Float promotion to
    `Q.dtype` (fp32) for the matmul, cast back to `V.dtype` at
    the end — same convention the `use_linear_attn` branch uses.
  - Mutually exclusive with `use_linear_attn`, `use_diff_attn`,
    `use_nsa_global`, `use_hybrid_heads`, `use_multiscale_heads`
    asserted in `forward()` (the cosFormer branch IS the
    attention path; combining with another is double-attention
    and a structural lever change).
- `models/llm.py`
  - `MinimalLLM.__init__` (around line 470): read
    `self.use_cosformer = getattr(config, "use_cosformer",
    False)`. When the flag is on, register
    `self.cosformer_gammas = nn.Parameter(torch.zeros(config.n_layers))`
    on the model — follows the 161-`layer_temperature` pattern
    (line 795 of layers.py: "the parameter lives on the model,
    not the MHA, so the optimizer sees ONE `nn.Parameter`").
  - Two `TransformerBlock` construction sites (around lines 952
    and 1329): pass `use_cosformer=self.use_cosformer` through to
    each block.
  - Forward loop: in each block call, pass
    `cosformer_gamma=self.cosformer_gammas[block_idx]` (or `None`
    when the flag is off). Mirrors the
    `layer_temperature[layer_index]` plumbing already used by 161.
- `configs/llm_config.py`
  - `LLMConfig`: add `use_cosformer: bool = False`,
    `cosformer_gamma_init: float = 0.0` (default off / 0).
  - New `Tiny1M3MCosFormerConfig(Tiny1M3MConfig)` setting
    `use_cosformer: bool = True`.

## Control
- **Control**: `Tiny1M3MConfig` (baseline, val ≈ 6.4216).
- **Treatment**: `Tiny1M3MCosFormerConfig` (same as baseline +
  `use_cosformer=True`).
- **Tier**: `tiny1m3m` (0.94M params · 3M tokens).
  **Seed 42 only** (one seed, per the protocol).
- Direct prior is 004-retnet-retention (closed null, Δ=+0.04
  wrong-sign at 0.94M with φ(x)=elu(x)+1). 189 swaps φ to
  exp(γx)·cos(x); a 189 null closes the linear-time
  kernel-replacement family at 0.94M, a 189 win isolates the
  failure mode to feature-map shape.

## Cost
- **Params**: +12 scalars (one γ per block, 12 blocks at
  tiny1m3m). γ is registered as a single
  `nn.Parameter(torch.zeros(n_layers))` on `MinimalLLM`, so
  exactly one optimizer entry, not 12. Cost is
  +0.0013% of 0.94M.
- **FLOPs**: at T=2048 with H=4, d_k=16, the linear-time branch
  is `O(T·d_k²·H) = 2,097,152` matmul ops per layer per step
  (K'V is `[H,d_k,d_k]` × `[H,T,d_k]`, then Q'·KV is
  `[H,T,d_k]` × `[H,d_k,d_k]`). Roughly equivalent to the
  softmax baseline at this T — the linear-vs-quadratic bet is
  invisible at T=2048; the lever reduces to a *kernel-shape*
  bet (cosine vs softmax). Cost is well under 5% of the
  baseline forward.
- **Memory**: +12 scalars on the model, no extra activations
  beyond the cumsum buffers the `use_linear_attn` branch
  already allocates (the pattern is identical).

## Run
- **Command** (on the box, `tiny1m3m`, seed 42):
  ```
  cd /root/universe-lm
  /venv/main/bin/python _arq_189-cosformer-linear-attn.py
  ```
  The stub subclasses `Tiny1M3MCosFormerConfig` (flag ON by
  default) and drives `train_llm.main()` with
  `--config_class __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false`.
- **Tier**: `tiny1m3m`, **seed 42** (one seed only).
- **Expected wall-clock**: ~12 minutes (default `job_timeout`).
- **Pass/fail bar** (from `idea.md` §"Pass-bar (r1 fix — tighter
  than default)"):
  - **WIN**: val_loss Δ ≤ −0.005 (default bar, tightened for
    softmax-replacement transition risk)
  - **NULL**: val_loss Δ ≥ +0.003 (tighter than default +0.01)
  - **NOISE BAND**: −0.003 < Δ < −0.005 → inconclusive, treated
    as null. The lever must EITHER win cleanly OR fail cleanly.
  - **Aux diagnostic**: attention entropy at step 10 ≥ softmax's
    × 1.10. Catches degenerate-φ collapse.
- **Bit-identity check at step 0** (real-model, fp32):
  `trt = build(use_cosformer=True); ctrl = build(use_cosformer=False)`;
  `trt_out, ctrl_out = trt(x), ctrl(x)` for one real batch.
  At γ=0 the lever reduces to the cumulative mean of V over the
  causal prefix (since cos(Q)·cos(K)^T ≈ 1 and cumsum(K') ≈
  (t+1)). Test: `cummean = ctrl_out.cumsum(dim=1) /
  torch.arange(1, T+1, device=ctrl_out.device).view(1,T,1,1,1)`;
  `assert (trt_out - cummean).abs().max() < 1e-6`. Reduces to
  `assert (trt_out[0] - ctrl_out[0]).abs().max() < 1e-6` at
  t=0 and global-mean equivalence at t=T-1. The toy
  `d=64, T=512` test in the original spec is NOT the project
  standard — the real-model test above is. ✓
