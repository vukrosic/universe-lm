# Plan — 197 Tied W_O Across Blocks (Soft Blend, α_b_raw Init −10)

## Flag
`use_tied_wo_across_blocks: bool = False` (default OFF) on `LLMConfig`.
Treatment subclass: `Tiny1M3MTiedWOConfig` at `configs/llm_config.py:____` (added
at the bottom of the file alongside the other Tiny1M3MConfig subclasses, matching
the 188 / 192 / 196 / 207 placement convention).

## Change

### `configs/llm_config.py`
- Add `use_tied_wo_across_blocks: bool = False` to `LLMConfig` (with a long
  comment block matching the style of `use_lowrank_wo` at line 1308 / 171 at
  line 1288). The comment explains: (a) one shared `W_O_shared` Parameter
  allocated on the model of shape `[d_model, d_model]`, (b) one
  `tied_wo_alpha_raw` Parameter per MHA, init `−10.0` (sigmoid → ≈ 4.54e-5),
  (c) the per-block blend `W_O_eff = (1 − σ(α))·W_O_b + σ(α)·W_O_shared`,
  (d) step-0 byte-identical within fp32 noise of one extra multiply-add (same
  tolerance as 188 / 206), (e) default off ⇒ baseline path bit-identical.
- Add the `@dataclass` subclass `Tiny1M3MTiedWOConfig(Tiny1M3MConfig)` with
  `use_tied_wo_across_blocks: bool = True` (mirrors `Tiny1M3MTopKAttnConfig`
  at line ~6726 — the dataclass-inheritance pitfall means a bare non-dataclass
  subclass won't override the parent default).

### `models/llm.py` (MinimalLLM.__init__)
- Read the flag: `self.use_tied_wo_across_blocks = getattr(config,
  "use_tied_wo_across_blocks", False)`.
- When the flag is on, allocate the single shared matrix on the model
  itself: `self.tied_wo_shared = nn.Parameter(torch.empty(d_model,
  d_model))` then `torch.nn.init.normal_(self.tied_wo_shared, mean=0.0,
  std=0.02)` (matches the standard `qkvo_proj` init from the baseline —
  per the `plain` field of `idea.md` which says "init at the baseline's W_O";
  the baseline uses Kaiming/normal std=0.02 for the qkvo slice, see
  `models/components.py:__init_weights` and the `torch.empty` + manual
  `normal_` in the 207-lowrank-wo path at layers.py:2004-2008).
- Plumb `use_tied_wo_across_blocks=self.use_tied_wo_across_blocks` AND
  `tied_wo_shared=self.tied_wo_shared` through to every `TransformerBlock`
  construction (in BOTH the `yoco_upper_blocks` ModuleList AND the
  standard `transformer_blocks` ModuleList, matching the existing
  pass-through convention used by 188 / 171 / 207).
- Mutually-exclusive with `use_yoco` and `use_gau` at construction (the
  shared W_O shared parameter lives on the model and the YOCO upper-half /
  GAU blocks don't take a `tied_wo_shared` kwarg). Add an assert mirroring
  the `use_hyper_connections + use_yoco` guard at line 1933-1944. (If YOCO
  is on, the W_O is shared anyway via YOCO's KV sharing — tying W_O on top
  is incoherent.)

### `models/layers.py` (TransformerBlock.__init__ + MHA.__init__ + MHA.forward)
- Add `use_tied_wo_across_blocks: bool = False` and `tied_wo_shared=None`
  kwargs to `TransformerBlock.__init__` (alongside the 188 / 204 / 207
  pass-through kwargs at the end of the signature, around line 5713-5722).
- Add the same two kwargs to `MultiHeadAttention.__init__` (alongside the
  `use_lowrank_wo` / `use_dropconnect_wo` W_O-side kwargs at line 1286-
  1335). When `use_tied_wo_across_blocks=True`:
  - Store `self.tied_wo_shared = tied_wo_shared` (this is the SAME parameter
    reference for every block — the model's `tied_wo_shared`, not a copy).
  - Allocate `self.tied_wo_alpha_raw = nn.Parameter(torch.full((), -10.0))`.
  - In the `else` branch (flag off), set both to `None` stubs for safe
    attribute lookups.
- In `MHA.forward` at the W_O application site (line ~5066, right after
  `w_o = self.qkvo_proj[self.qkv_size:]` and BEFORE the 171-DropConnect
  branch), add the blend block:
  ```python
  # 197 — Tied W_O across blocks. Soft-blend the per-block
  # W_O with a single global W_O_shared via a learnable
  # per-block sigmoid-bounded α. At step 0 α ≈ 4.54e-5, so
  # the contribution from W_O_shared is on the order of 1e-7
  # in std, well within fp32 noise of one extra multiply-add
  # ⇒ forward is bit-identical to baseline up to fp32 noise.
  # Composes with 171-DropConnect (next) and 207-LowRank
  # (after that) — both still see a valid O-slice tensor.
  if self.use_tied_wo_across_blocks:
      alpha = torch.sigmoid(self.tied_wo_alpha_raw)
      w_o = (1.0 - alpha) * w_o + alpha * self.tied_wo_shared
  ```
  The blend site is BEFORE 171-DropConnect and 207-LowRank so the order
  composes: `W_O_eff = ((1-α)·W_O_b + α·W_O_shared) → [optional 171 mask]
  → [optional 207 lowrank add] → F.linear`. This is the cheapest possible
  insertion point (one extra multiply-add on a `[d_model, d_model]` tensor
  per block per forward).

## Control
- **Control**: `Tiny1M3MConfig` (val 6.40 ± 0.04 cached; daemon owns it).
- **Treatment**: `Tiny1M3MTiedWOConfig` (this plan, the new subclass).
- **Tier**: tiny1m3m (0.94M params, 3M tokens, 12 layers, d_model=64).
- **Seed**: 42, always. No multi-seed (per the §1 one-seed-only rule).

## Cost
- **Params**: +1 shared `W_O_shared` (d_model² = 4,096) + 12
  `tied_wo_alpha_raw` scalars (12) = +4,108 params (+0.44% of 0.94M).
  Treatment is param-*superset* of control (no per-block params removed),
  so the A/B is parameter-shape, not model-size.
- **FLOPs**: +1 element-wise add + 1 element-wise multiply on a
  `[d_model, d_model]` tensor per block per forward (the blend
  `w_o = (1-α)·w_o + α·W_O_shared`). At tiny1m3m that's ~12 blocks ×
  4,096 = ~49K extra FLOPs/forward, well within the per-step noise band
  (the SDPA QK^T/AV matmuls dominate). <0.1% wall-clock overhead.
- **Memory**: 0 extra activations (the blend rewrites `w_o` in-place
  style — it produces a new tensor, but `F.linear` reads it once and the
  old `w_o` slice is GC'd). +1 persistent Parameter of 4,096 floats
  (16 KB at fp32, 8 KB at bf16).
- **Optimizer state**: +1 entry for `tied_wo_shared` (AdamW: 2 × 4,096
  = 8,192 floats = 32 KB) + 12 entries for `tied_wo_alpha_raw`
  (negligible). Total +40 KB. Sub-noise.

## Run
- **Command**: standard daemon dispatch via `_arq_197-tied-wo-across-blocks.py`
  (see §4b artifact below).
- **Tier / seed**: tiny1m3m / seed 42.
- **Expected wall-clock**: ~6 min (12m job_timeout gives plenty of headroom;
  the 196/192/207 siblings all completed in this window at the same tier).
- **Pass/fail bar** (copied from `idea.md`):
  - **WIN**: `trt_val ≤ ctrl_val − 0.01` AND clears the two-ctrl bracket.
  - **NULL**: `|trt_val − ctrl_val| < 0.01` (sub-noise band).
  - **DRIFT**: `trt_val > ctrl_val + 0.01`.
  - Reference: champion val 6.2403 (175-alibi), current baseline mean
    6.40 ± 0.04 (cache); box noise floor ±0.01 at 0.94M/3M.
