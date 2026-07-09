# Plan — 161 dyt-temp

## Flag
`use_per_layer_temp: bool` (default `False`) on `LLMConfig`.
Config class `Tiny1M3MPerLayerTempConfig(Tiny1M3MConfig)` with `use_per_layer_temp=True`.
File: `configs/llm_config.py` (default, near `use_per_head_temp`) and the
treatment subclass added right after `Tiny1M3MPerHeadTempConfig`.

## Change
- `configs/llm_config.py` — adds `use_per_layer_temp: bool = False` on
  `LLMConfig` (next to the closed `use_per_head_temp` flag for the
  per-head sibling lever). Adds `Tiny1M3MPerLayerTempConfig(Tiny1M3MConfig)`
  `@dataclass` subclass with `use_per_layer_temp = True`. This is the
  class the daemon imports as `C` for the build-smoke and run.
- `models/llm.py` — `MinimalLLM.__init__` reads
  `self.use_per_layer_temp = getattr(config, "use_per_layer_temp", False)`.
  When on, registers ONE shared `nn.Parameter` on the model
  (`self.layer_temperature = nn.Parameter(torch.full((n_layers,),
  1/sqrt(d_k)))`) — ONE parameter (not n_layers of them) so the
  optimizer sees a flat layout. Then each `TransformerBlock` is given
  a back-reference: `blk._layer_temperature = self.layer_temperature`.
  The flag is also threaded into the standard `TransformerBlock(...)`
  constructor call (`use_per_layer_temp=...`) so each block passes it
  on to its inner MHA.
- `models/layers.py` — `MultiHeadAttention.__init__` accepts
  `use_per_layer_temp: bool = False` (the parameter itself lives on
  the MODEL, not here — see above; this flag just controls whether
  MHA applies the scaling at forward). `forward()` signature gains two
  kwargs: `layer_index=None` and `layer_temperature=None`. The
  branch is `if self.use_per_layer_temp and layer_temperature is not
  None and layer_index is not None: scores = scores *
  layer_temperature[layer_index].view(1, 1, 1, 1)`. Applied AFTER the
  standard `1/sqrt(d_k)` (or per-head `τ_h`) scale so it composes with
  every existing score-side lever (alibi, cosine, qk_bilinear,
  talking_heads_q, antisym_qk, q_feature_map, per_head_temp,
  attn_logit_bias). Added in BOTH the FIRE-branch manual path and the
  non-FIRE manual path. The lever is also added to the manual-path
  forcing condition (alongside `use_per_head_temp`) so SDPA's flash/
  efficient backends don't perturb step-0 numerics.
- `models/layers.py` — `TransformerBlock.__init__` accepts
  `use_per_layer_temp: bool = False` (pass-through to inner MHA).
  `TransformerBlock.forward` signature gains no new required args —
  it reads `getattr(self, "_layer_temperature", None)` and passes
  `layer_index=layer_index, layer_temperature=...` into
  `self.attention(...)` at all three call sites (pre-norm, post-norm,
  parallel block). `layer_index` is already plumbed by the model loop.

Step-0 identity (flag OFF): no Parameter is registered, no branch is
taken, baseline path bit-identical. Verified locally: `MinimalLLM
(Tiny1M3MConfig())` ≡ `MinimalLLM(Tiny1M3MConfig())` to 0.0 max-abs-diff
on a 16-token forward at seed 42.

Step-0 identity (flag ON, not the goal but documented): the
`layer_temperature` parameter consumes RNG state during model
construction, so the flag-on model is NOT byte-identical to the bare
baseline at step 0. The init `τ_l = 1/sqrt(d_k)` exactly means each
layer's score scale `Q_h K_h^T * τ_l` matches baseline at the
multiplication site, but the model's RNG state diverges from the
flag-off path (the new Parameter is created). This is the same caveat
as `use_per_head_temp` (155) and most other flag-on lever paths —
flag-on is the treatment, not the identity check; flag-OFF is the
identity check.

## Control
- A: `Tiny1M3MConfig` (seed 42, flag OFF) — bare tier config. The
  daemon owns this control.
- B: `Tiny1M3MPerLayerTempConfig` (seed 42, flag ON) — same tier,
  `use_per_layer_temp=True`.
- Tier: `tiny1m3m` (0.94M params, 3M tokens). Seed 42 only (one-seed-
  only rule).

## Cost
- Params: + n_layers scalars = +12 at tiny1m3m (12 layers, +0.001% of
  0.94M). Stored as ONE `nn.Parameter` of shape `[n_layers]` on the
  model, broadcast per-block.
- FLOPs: + 1 multiply per (head, query, key) per layer per token per
  forward (~negligible — one scalar broadcast).
- Memory: + 12 floats, ~negligible.

## Run
- Command (after code lands + sync to box):
  ```
  cd /root/universe-lm && /venv/main/bin/python _arq_161-dyt-temp.py
  ```
  This invokes `train_llm.main()` with `--config_class __main__.C --seed 42
  --dataset_path processed_data/pretrain_1B --warmup false`. The daemon's
  `claimable()` picks the idea up from `needs-run` via `run.json`.
- Tier: `tiny1m3m`, seed 42.
- Expected wall-clock: ~2-6 min (same as baseline — the extra
  multiply is a single scalar broadcast, negligible vs the manual
  attention path overhead which is already forced by the closed
  score-side levers at the same site).
- **Pass/fail bar** (from `idea.md`):
  - PASS ≤ ctrl − 0.005 (layers want different attention
    temperatures at this scale — early broad, late sharp).
  - NULL band |Δ| < 0.005 (the `1/sqrt(d_k)` constant is the
    canonical default; Q/K gradients can absorb any per-layer scale
    change; per-block normalization absorbs the variance).
  - DRIFT > +0.005 (the lever is harmful at this scale — the
    gradient from `τ_l` disturbs a useful prior).
