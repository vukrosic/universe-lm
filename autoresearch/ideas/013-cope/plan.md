# Plan — 013 CoPE (Content-aware Positional Encoding)

## Flag
`use_cope: bool = False` (default OFF). Added at `configs/llm_config.py:164`.
A preset `Tiny1M3MCoPEOnFireConfig` (fire + cope) is also at line 681, so the
runner can launch the treatment via `--config_class` directly (no `_arq_013.py`
needed — see §Run).

## Change

### 1. New file `models/cope.py` (~88 LoC)
- `CoPE` class with `__init__(d_model, n_heads, max_seq_len, threshold=0.0)`.
- Learnable per-head probe `self.probe ∈ R^{H×D}` initialized with
  `nn.init.normal_(self.probe, mean=0.0, std=0.02)` (mirrors FIRE's
  per-head content projection init at `models/fire_pe.py:60`).
- `forward(x) → [B, H, T, T]` bias:
  - `score[b,t,h] = einsum("btd,hd->bth", x, probe)`
  - `g = sigmoid(score - 0.0)`  (τ=0 pinned)
  - `cum_g = cumsum(g, dim=1)`
  - `offset[b,h,i,j] = cum_g[b,i,h] - cum_g[b,j-1,h]` (count in [j, i];
    uses zero-prepend trick for the j=0 case)

### 2. `models/layers.py` (wiring)
- Import: `from .cope import CoPE` (line 7).
- `MultiHeadAttention.__init__`:
  - New kwarg `use_cope: bool = False`.
  - When `True`: `self.rotary = None`; build `self.cope = CoPE(d_model,
    n_heads, max_seq_len)`. When `False`: build `self.rotary` as
    before; `self.cope = None`. Gates the Rotary construction per the
    RoPE call-site audit (review.md:42-46).
  - `forward`: top-of-function assert that `use_cope` and
    `use_qk_norm_post_rope` are not both `True` (mutually exclusive —
    post-RoPE norm has nothing to act on).
  - `forward`: extend the `use_nope`-only RoPE skip to also cover
    `use_cope` (CoPE replaces RoPE; Q/K RMSNorm still runs).
  - `forward`: extend the manual-attention-routing condition to
    include `or self.use_cope` (CoPE bias is added to the [B,H,T,T]
    score tensor, so the manual path is required).
  - `forward`: in the FIRE branch, after adding `fire_bias` to scores,
    add `self.cope(x)` if `use_cope`. In the tweaks branch, after all
    score-side tweaks (Q3/Q4/Q1/Q10/Q9/Q27) and before the mask, add
    `self.cope(x)` if `use_cope`.
- `TransformerBlock.__init__`: new kwarg `use_cope: bool = False`,
    pass-through to the MHA constructor.

### 3. `models/llm.py` (config plumbing)
- L218: `self.use_cope = getattr(config, "use_cope", False)`.
- L331: pass `use_cope=self.use_cope` into `TransformerBlock(...)`.

### 4. `configs/llm_config.py`
- New field `use_cope: bool = False` with docstring (around lines 152-164).
- New preset `Tiny1M3MCoPEOnFireConfig` (line 681) sets
  `use_fire_pe=True; use_cope=True` — the stacked treatment.

## Control
- **Treatment**: `Tiny1M3MCoPEOnFireConfig` (fire ON + cope ON) — the
  stacked content-conditional lever. PASS bar is vs the FIRE-equipped
  baseline (6.3234 per closed.md, 009 WIN).
- **Stacking rationale**: per idea.md, the A/B is "FIRE + CoPE" vs
  "FIRE alone" — CoPE is a *stacked* lever on top of the current best
  baseline, not a replacement. A standalone CoPE (no FIRE) is also
  useful as a secondary probe but is not the primary A/B.
- **Tier**: `tiny1m3m` (0.94M params, 3M tokens), seed 42.
- **Ctrl for box-drift**: the runner's standard `Tiny1M3MConfig` ctrl
  is fine for box-drift detection (it's the no-flags baseline). The
  real A/B compares treatment to the historical 6.3234 (the FIRE-only
  ctrl from 009's WIN run).

## Cost
- **Params**: probe [n_heads × d_model] = 4 × 64 = 256 params/block
  × 12 blocks = **3,072 extra params** at tiny1m3m (+0.3% over the
  949k baseline). At screen10m: 6 × 144 × 24 = 20,736 (+0.27%).
- **FLOPs per layer**: forward adds one matmul (B·T·D·H) and a few
  elementwise ops (sigmoid, cumsum, diff). Negligible next to Q@K^T
  (B·H·T²·D) and softmax.
- **Memory**: peak activation for the [B, H, T, T] bias is
  2 × 4 × 2048² × 4 bytes ≈ **132 MB per layer** at tiny1m3m;
  ~200 MB at screen10m. Comparable to the attention scores
  themselves. No parameter memory overhead beyond the probe.

## Run
- **Command** (per `vast-runner-harness` memory):
  ```bash
  /venv/main/bin/python train_llm.py \
    --config_class configs.llm_config.Tiny1M3MCoPEOnFireConfig \
    --seed 42 --dataset_path processed_data/pretrain_1B --warmup false
  ```
  No `_arq_013.py` needed — the treatment is a registered config class.
  If a FIRE-only ctrl is desired for the stacked A/B, write
  `_arq_013_fire.py` with `use_fire_pe=True` and run that too.
- **Wall-clock**: tiny1m3m ≈ 5-10 min. CoPE adds ~5% FLOPs to
  attention (manual path is slightly slower than the SDPA fast path
  for the FIRE branch) — wall-clock bump is small.
- **Pre-flight**: smoke `MinimalLLM(cfg)` for ctrl + treatment +
  CoPE+FIRE stacked (CPU, no training) — verified; all build;
  forward finite; max diff OFF vs ON-standalone = 0.072 (CoPE takes
  effect from step 0 — the probe is non-zero, so bias is non-zero).

## Pass / fail bar (from idea.md / review.md)
- **PASS**: Δ ≤ −0.01 vs the **FIRE-equipped baseline (6.3234)**.
- **NULL / INCONCLUSIVE**: |Δ| < 0.01.
- **DRIFT**: Δ > +0.01.
- **Hypothesis range** (from idea.md): Δ ∈ [−0.01, −0.02] (tightened
  from [−0.005, −0.02] per the review).
- **Identity at step 0**: NOT bit-identical — `g ≈ 0.5 ± 0.04` at init
  (probe N(0, 0.02), so `score ~ N(0, 0.02·√D)`), giving
  `offset ≈ (i − j + 1) · 0.5` (linear in distance, RoPE-like). This
  matches the DeepNet-style "lever takes effect at step 0" expectation.
  The A/B is still clean: control has no CoPE bias, treatment has the
  small RoPE-like bias at init that grows as the probe learns.
