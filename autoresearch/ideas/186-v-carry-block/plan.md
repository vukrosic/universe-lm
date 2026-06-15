# Plan — 186 within-block V-carry

## Flag
- `use_v_carry_block: bool = False` — default OFF. New field on
  - `configs/llm_config.py` — `LLMConfig` (default off) and the treatment subclass
    `Tiny1M3MVCarryBlockConfig(Tiny1M3MAlibiConfig)` with `use_v_carry_block: bool = True`.
    Stacks on the current 175-alibi champion per `autoresearch/champion.json` (val 6.2403);
    the 180-qk-logit-conv config was reverted as a causal-mask LEAK, so it is NOT the
    champion — subclassing it would inherit the leak. With `use_v_carry_block=False`,
    the subclass reduces to the champion (max-abs-diff = 0.0 verified in build smoke).
  - `models/layers.py` — `MultiHeadAttention.__init__(use_v_carry_block: bool = False)`.
  - `models/llm.py` — `MinimalLLM` reads `config.use_v_carry_block` and threads
    `use_v_carry_block=...` into every `TransformerBlock(...)` construction.

## Change
- `configs/llm_config.py`
  - Add `use_v_carry_block: bool = False` to `LLMConfig` (single source of truth
    for the dataclass, mirrored into `MultiHeadAttention` and `TransformerBlock`).
  - Append a new `Tiny1M3MVCarryBlockConfig(Tiny1M3MAlibiConfig)` subclass that flips
    `use_v_carry_block = True` (matches the established pattern of
    `Tiny1M3MValueResidualConfig`, `Tiny1M3MQCarryConfig`, `Tiny1M3MAVOutputCarryConfig`).
    Stacking on `Tiny1M3MAlibiConfig` (the 175-alibi champion, val 6.2403) means the
    lever composes on top of learnable per-head ALiBi slopes, not on the bare
    `Tiny1M3MConfig` baseline.
- `models/layers.py`
  - `MultiHeadAttention.__init__`: accept `use_v_carry_block: bool = False`. When
    on, allocate `self.v_carry_alphas = nn.Parameter(torch.zeros(n_heads))`
    (init 0 ⇒ `α_h = tanh(0) = 0` exactly ⇒ recurrence collapses to identity at
    step 0). Otherwise `self.v_carry_alphas = None` (stub for attribute safety;
    forward branch is gated on `self.use_v_carry_block`).
  - `MultiHeadAttention.forward`: after the `Q, K, V = …transpose(1, 2)` line
    (V in `[B, H, T, d_k]`), and immediately after the existing
    `use_value_residual` stash/blend block (line 3254), insert the recurrence
    branch:

    ```python
    if self.use_v_carry_block:
        # α_h = tanh(v_carry_alphas_h) keeps |α_h| ≤ 1, so the closed-form
        # sum Σ_k α^k V[k] stays bounded even at T=2048.
        alpha = torch.tanh(self.v_carry_alphas)  # [H], |alpha| < 1
        T = V.size(2)
        # kernel[h, k] = alpha_h^k, length T. F.conv1d is cross-correlation,
        # so we FLIP the kernel to recover the causal Σ-k form, and left-pad
        # the input by T-1 zeros. The closed form is exactly the recurrence
        # V_new[t] = α·V_new[t-1] + V_orig[t] for t ≥ 1, V_new[0] = V_orig[0]
        # (verifiable by induction on t; see plan.md §Change derivation).
        arange = torch.arange(T, device=V.device, dtype=V.dtype)
        alpha_pow = alpha.unsqueeze(1).pow(arange.unsqueeze(0))  # [H, T]
        kernel = alpha_pow.flip(1)                              # [H, T]
        # depthwise conv1d: each (B, h, j) channel processed independently
        # along T with its head's kernel — matches 134-mega's depthwise
        # depthwise-EMA conv1d pattern (lines 2823-2834).
        V_flat = V.permute(0, 1, 3, 2).contiguous().reshape(
            V.size(0), self.n_heads * V.size(3), T
        )                                                       # [B, H*d_k, T]
        V_padded = F.pad(V_flat, (T - 1, 0))                   # [B, H*d_k, 2T-1]
        weight = kernel.unsqueeze(2).expand(
            self.n_heads, V.size(3), T
        ).reshape(self.n_heads * V.size(3), 1, T)              # [H*d_k, 1, T]
        V_out = F.conv1d(V_padded, weight, groups=self.n_heads * V.size(3))
        V = V_out.reshape(
            V.size(0), self.n_heads, V.size(3), T
        ).permute(0, 1, 3, 2).contiguous()                      # [B, H, T, d_k]
    ```
  - `TransformerBlock.__init__`: accept `use_v_carry_block: bool = False`,
    forward to inner MHA. (No model-level state — purely local to each block,
    unlike 021/164/168 which stash carry across blocks. The recurrence runs
    causally within a single block; no inter-block plumbing needed.)
- `models/llm.py`
  - `MinimalLLM.__init__`: add `self.use_v_carry_block = getattr(config,
    "use_v_carry_block", False)`.
  - Both `TransformerBlock(...)` construction sites (line 779 and 1097, the
    pre-norm and post-norm paths) pass `use_v_carry_block=self.use_v_carry_block`.
- `autoresearch/ideas/186-v-carry-block/run.json` — the daemon descriptor.
- `_arq_186-v-carry-block.py` — repo-root bootstrap with top-level `C`
  subclass.

## Control
- **Control**: `configs/llm_config.Tiny1M3MConfig`, seed 42, `--warmup false`,
  dataset `processed_data/pretrain_1B`, val_mean from the cache reference
  (`autoresearch/baseline-cache.json` box `5b8a7fea8963`, RTX 3060,
  val_mean=6.3988 / noise_band=0.04 / n_measurements=3 — re-pull on run day).
  The daemon owns the ctrl.
- **Treatment**: `Tiny1M3MVCarryBlockConfig` (one flag flip: `use_v_carry_block=True`),
  otherwise identical. Seed 42, same warmup/dataset. The lever is *only* the
  per-head recurrent mix on V; nothing else changes.
- **Tier**: tiny1m3m only. Single seed 42 per the one-seed-only rule.

## Cost
- **Params**: H=4 heads × n_layers=12 = 48 new scalars (`v_carry_alphas`) —
  +0.005% of the 0.94M base. Trivial.
- **FLOPs (per layer, per forward)**: the depthwise conv1d is
  `B · (H·d_k) · T · K = 2 · 64 · 2048 · 2048 ≈ 0.54 GFLOPs`. Same shape as
  134-Mega-EMA's depthwise conv (lines 2823-2834 use the same
  `F.conv1d(padded, kernel, groups=…)` with kernel length T). 12 layers ×
  ~0.5 GFLOPs ≈ 6 GFLOPs/forward added vs ~1.5 GFLOPs/forward FFN cost
  (the conv is ~4× the FFN cost). The Python for-loop alternative is ~2k
  sequential ops per head per block — much slower on GPU. Document the
  choice; tag the implementation with a comment.
- **Memory**: kernel weight is `[H*d_k, 1, T]` = `[64, 1, 2048]` = ~512 KB per
  layer; alpha_pow `[H, T]` = ~32 KB per layer. Negligible.
- **Wall-clock**: tiny1m3m training is ~12 min on the box; +6 GFLOPs/forward
  is ~+5-10% wall-clock. Within the 12m `job_timeout` budget.

## Run
- **Command** (treatment):
  `python _arq_186-v-carry-block.py`
  → `train_llm.main()` with `--config_class __main__.C --seed 42
   --dataset_path processed_data/pretrain_1B --warmup false`.
- **Tier**: tiny1m3m, seed 42 (one seed only per the one-seed-only rule).
- **Expected wall-clock**: ~12-14m on the box (treatment), ctrls ~3× from
  daemon's MEASURE path.
- **Pass/fail bar** (copied from `idea.md`):
  - WIN: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule.
  - NULL: `|trt_val − ctrl_val_mean| < 0.01` (most likely outcome per the
    168 null pattern).
  - DRIFT: `trt_val > ctrl_val_mean + 0.01`.
  - Sub-noise is inconclusive per one-seed-only rule.
- **Self-check** (§5): flag OFF reproduces baseline (asserted via the
  `if self.use_v_carry_block:` gate — branch never taken ⇒ forward bit-
  identical). Treatment path runs the conv1d and the `MinimalLLM(C())`
  build-smoke constructs on CPU. `max_abs_diff(trt_step0_logits,
  ctrl_step0_logits) == 0.0` is the secondary smoke (α_h = 0 ⇒ kernel =
  `[1, 0, …, 0]` after tanh, V_new = V exactly).
- **Per-head α_h readout**: append `trt.attention.v_carry_alphas.detach()
  .cpu().abs()` to the run artifact (per reviewer note) so a null is
  interpretable (did the optimizer not move α? did it move to 0? did it
  diverge per-head?).
