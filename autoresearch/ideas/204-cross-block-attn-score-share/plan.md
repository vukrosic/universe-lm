# Plan — 204 cross-block-attn-score-share

## Flag
- `use_cross_block_score_share: bool = False` (default OFF) and
  `score_share_alpha_init: float = -10.0` on `LLMConfig` in
  `configs/llm_config.py` (single source of truth, mirrored into
  `MultiHeadAttention` and `TransformerBlock`). New fields on
  `LLMConfig` immediately after `use_cross_block_kv_share: bool = False`
  (line 545 area). Treatment subclass
  `Tiny1M3MCrossBlockScoreShareConfig(Tiny1M3MAlibiConfig)` (subclass
  pattern, `@dataclass`-decorated per the 188 / 161 precedence) flips
  `use_cross_block_score_share: bool = True` and stacks on the
  current 175-alibi champion per `autoresearch/champion.json` (val
  6.2403, band 0.04). With `use_cross_block_score_share=False` the
  subclass reduces to the champion (max-abs-diff = 0.0; verified in
  build smoke by toggling the flag and comparing
  `MinimalLLM(C(use_cross_block_score_share=False)).forward(...)`
  logits against the champion).
- `models/layers.py` — `MultiHeadAttention.__init__(use_cross_block_score_share: bool = False, score_share_alpha_init: float = -10.0)`.
- `models/llm.py` — `MinimalLLM.__init__` reads
  `config.use_cross_block_score_share` and threads
  `use_cross_block_score_share=...` into every `TransformerBlock(...)`
  construction (the two sites at lines ~821 and ~1166).

## Change
- `configs/llm_config.py`
  - Add `use_cross_block_score_share: bool = False` to `LLMConfig`.
  - Add `score_share_alpha_init: float = -10.0` to `LLMConfig` (so
    tests can vary the init; pinned to -10 in the treatment config).
  - Append a new
    `Tiny1M3MCrossBlockScoreShareConfig(Tiny1M3MAlibiConfig)` subclass
    that flips `use_cross_block_score_share: bool = True`
    (dataclass-decorated; mirrors the 188 / 161 pattern).
- `models/layers.py`
  - `MultiHeadAttention.__init__`: accept
    `use_cross_block_score_share: bool = False` and
    `score_share_alpha_init: float = -10.0`. When on, allocate
    `self.score_share_alpha_raw = nn.Parameter(torch.full((),
    score_share_alpha_init))` (init -10 ⇒ `σ(-10) ≈ 4.5398e-5` at
    step 0 ⇒ `scores_eff ≈ scores_self` within fp32 noise of one
    extra multiply-add). Plus a forward-pass-local stash slot
    `self._prev_block_scores = None` (initialized to `None` even
    when the flag is off so the model loop's
    `block.attention._prev_block_scores` readout is always safe).
    When off, stub `self.score_share_alpha_raw = None` and
    `self._prev_block_scores = None` (the forward branch is gated
    on `self.use_cross_block_score_share`).
  - `MultiHeadAttention.forward`:
    1. Add `prev_block_scores=None` kwarg.
    2. Add `or self.use_cross_block_score_share` to the manual-path
       trigger list at `models/layers.py:3865` (next to the existing
       `use_attn_logit_bias`, `use_per_head_temp`, etc. —
       `use_cross_block_score_share` is a score-space op, so it has
       to land on the manual attention path; SDPA's flash kernel
       cannot apply a per-head score blend).
    3. Inside the manual path, IMMEDIATELY AFTER the base scores
       computation (line 3921, `scores = torch.matmul(Qn,
       Kn.transpose(-1, -2)) * scale`), inject the blend:

       ```python
       if self.use_cross_block_score_share:
           # Always stash the current layer's pre-softmax scores
           # (detached) so the model loop can read them after the
           # layer-0 call and pipe them as `prev_block_scores=` to
           # layers 1..N-1. Shape: [B, H, T, T]. `.detach()` keeps
           # the cross-block gradient structurally bounded to
           # `score_share_alpha_raw`'s 0-dim scalar.
           self._prev_block_scores = scores.detach()
           if prev_block_scores is not None:
               # Layer l ≥ 1 — blend the previous block's pre-
               # softmax scores with the current block's:
               #   α = σ(score_share_alpha_raw)         # ~4.5e-5 at init
               #   scores_eff = (1-α)·scores_self + α·prev_block_scores
               # α=0 init ⇒ scores_eff = scores_self exactly
               # within fp32 noise of one extra multiply-add
               # (the `(1-α)` multiply is `1 - 4.5e-5 ≈ 1`, the
               # `α·prev_block_scores` term is `4.5e-5·prev`,
               # which rounds to a single ulp at most).
               alpha = torch.sigmoid(self.score_share_alpha_raw)
               scores = (1.0 - alpha) * scores + alpha * prev_block_scores
       # else: lever is off — the stash is a no-op (the branch
       # is gated and `self._prev_block_scores` keeps its
       # initial `None` value; the model loop's `if self.use_…
       # and i == 0` guard skips the read). Baseline path bit-
       # identical.
       ```

       Note: the blend is applied to the SCALED scores (after
       `* scale = * 1/sqrt(d_k)`) so `scores_self` here is exactly
       the `Q·K^T / √d_k` logit that would normally enter softmax.
       The blend happens BEFORE the per-head logit bias (152),
       alibi (Q1), per-head τ (Q3), Q4 / Q10 / Q-bilinear
       transforms, etc. — so it composes as a "pre-softmax
       smoothing" of the block-local attention pattern.
  - `MultiHeadAttention.forward` signature: add `prev_block_scores=None`
    kwarg. Add the parameter to the existing forwarding chain (the
    MHA forward already has many other cross-block kwargs: `q_carry`,
    `av_carry`, `v_residual`, `shared_kv`, `prev_W_K`, `prev_W_V`).
  - `TransformerBlock.__init__`: accept
    `use_cross_block_score_share: bool = False`, forward to inner
    MHA.
  - `TransformerBlock.forward`: add `prev_block_scores=None` kwarg,
    pass through to `self.attention(...)` along with the existing
    `q_carry=`, `av_carry=`, `prev_W_K=`, `prev_W_V=`, etc.
- `models/llm.py`
  - `MinimalLLM.__init__`: add
    `self.use_cross_block_score_share = getattr(config,
    "use_cross_block_score_share", False)` (after the
    `use_cross_block_kv_share` block at line 433).
  - Both `TransformerBlock(...)` construction sites (line ~821 and
    ~1166, pre-norm and post-norm paths) pass
    `use_cross_block_score_share=self.use_cross_block_score_share`.
  - `_run_post_embed`: add
    `prev_block_scores: Optional[torch.Tensor] = None` next to the
    existing `prev_W_K=None` / `prev_W_V=None` (line ~1670).
    Capture after layer 0:
    ```python
    if self.use_cross_block_score_share and i == 0:
        # After layer-0 MHA forward, the post-base-scores tensor
        # (shape [B, H, T, T], post `/sqrt(d_k)`, pre-softmax /
        # pre-mask) is stashed at `block.attention.
        # _prev_block_scores` (`.detach()`-ed inside MHA.forward).
        # Capture for layers 1..N-1. Same `not self.use_gau` guard
        # as v_residual / q_carry / av_carry / prev_W_K / prev_W_V
        # — GAUBlock has no `.attention` attribute and a fused
        # MHA+FFN operator doesn't compose with the score blend.
        if not self.use_gau:
            prev_block_scores = block.attention._prev_block_scores
    ```
    Pass `prev_block_scores=prev_block_scores` into the `block(...)`
    call (alongside `prev_W_K=prev_W_K`, `prev_W_V=prev_W_V`).
- `autoresearch/ideas/204-cross-block-attn-score-share/run.json` —
  the daemon descriptor.
- `_arq_204-cross-block-attn-score-share.py` — repo-root bootstrap
  with top-level `C` subclass.

## Why `σ(-10)`, not 0-init
A literal `α_raw = 0` would require `torch.sigmoid` or a
`nn.functional.softplus`-style softplus trick to keep α bounded in
[0, 1]; `σ(-10) ≈ 4.5398e-5` is one (a) the same trick 188 uses
for the K/V projection blend and (b) identity at step 0 in fp32
within one multiply-add of the no-flag baseline. The spec note
in `idea.md` says "α_b init 0" colloquially; the implementation
uses σ(-10) per the 188 / 161 / `unet-skips-gate-fix` family
precedent (sigmoid-gated scalar at -10 init ⇒ `sigmoid(-10) ≈ 0`
in fp32 noise). Per `idea.md`'s "Step-0 ≈ baseline when off"
assertion: the forward graph is PRACTICALLY baseline at step 0
(max-abs-diff across all 12 blocks < 1e-4) — the plan gate
asserts this rather than the literal "bit-identical" wording
that 188's reviewer (finding D) flagged as imprecise.

## Why Muon, not AdamW (per review.md finding E)
The 12 `score_share_alpha_raw` scalars are 1-D scalar gain
parameters, consistent with 021's `lambda_v` (the only-learned-
scalar pattern in the closed WIN family). They ride in the
Muon optimizer group (the 1-D gain parameter class that lives
in Muon at the same LR scale as `qk_norm_scale`, `head_window_
logit`, `lambda_v`, etc.). The plan calls this out; the
implementation uses `nn.Parameter` registration and lets the
existing optimizer-group machinery in `train_llm.py` pick it up
through the standard 1-D-params-to-Muron routing.

## Control
- **Control**: `configs.llm_config.Tiny1M3MConfig`, seed 42,
  `--warmup false`, dataset `processed_data/pretrain_1B`,
  val_mean from the cache reference
  (`autoresearch/baseline-cache.json`; re-pull on run day). The
  daemon owns the ctrl.
- **Treatment**: `Tiny1M3MCrossBlockScoreShareConfig` (one flag
  flip: `use_cross_block_score_share=True` plus the inherited
  ALiBi flag from the champion config). Seed 42, same
  warmup/dataset. The lever is *only* the pre-softmax score
  blend across adjacent blocks; nothing else changes.
- **Tier**: tiny1m3m only. Single seed 42 per the one-seed-only
  rule.

## Cost
- **Params**: 1 scalar `score_share_alpha_raw` per block ×
  n_layers=12 = 12 new scalars — +0.001% of the 0.94M base. The
  cheapest cost profile in the cross-block family (021 = 12
  scalars, 164 = 12, 168 = 12, 188 = 24).
- **FLOPs (per layer, per forward)**: one extra `(1-α)·scores
  + α·prev` elementwise add+mul on a `[B, H, T, T]` tensor =
  `2 · B · H · T · T = 2 · 2 · 4 · 2048 · 2048 ≈ 67 MFLOPs` per
  layer on layer l ≥ 1; layer 0 is a no-op (just a `.detach()`
  stash). Total ≈ 11 × 67 MFLOPs = ~0.74 GFLOPs/forward added
  vs ~1.5 GFLOPs/forward FFN cost (the manual-path attention
  itself is also forced by the flag, which adds the unmaterialized
  SDPA-flash overhead — net wall-clock impact is ~+5% on the
  attention compute, dominated by SDPA no longer being used).
  Within the 12m `job_timeout` budget; bump to `14m` to mirror
  the 186 plan if needed.
- **Memory**: `scores` is `[B, H, T, T] = [2, 4, 2048, 2048]`
  in fp32 = 128 MB per layer; `_prev_block_scores` is the same
  shape = 128 MB per layer (forward-pass-local, released after
  the forward pass). The standard manual attention path already
  materializes the scores tensor, so the additional cost is one
  extra 128 MB tensor per layer. Within the RTX 3060 12GB budget.
- **Wall-clock**: tiny1m3m training is ~12 min on the box; the
  manual-path overhead is ~+5-10% wall-clock (SDPA flash kernel
  is bypassed for all 12 blocks). Within the 12m `job_timeout`
  budget; bump to `14m` to be safe (matches 186's plan).

## Run
- **Command** (treatment):
  `python _arq_204-cross-block-attn-score-share.py`
  → `train_llm.main()` with `--config_class __main__.C --seed 42
   --dataset_path processed_data/pretrain_1B --warmup false`.
- **Tier**: tiny1m3m, seed 42 (one seed only per the one-seed-only rule).
- **Expected wall-clock**: ~12-14m on the box (treatment), ctrls
  ~3× from daemon's MEASURE path.
- **Pass/fail bar** (copied from `idea.md`, locked by reviewer
  finding A):
  - **WIN**: `trt_val ≤ ctrl_val_mean − 0.02` AND clears the
    two-ctrl rule, train right-sign, no NaN through 92 steps,
    val_acc right-sign.
  - **NULL**: `|trt_val − ctrl_val_mean| < 0.02` (most likely
    outcome per the 164/168/188 cross-block null pattern at
    this tier — but `α=0` init is a sharper lever than 021's
    `λ=0` so the bar is at the upper edge of the ±0.04 noise
    band).
  - **DRIFT**: `trt_val > ctrl_val_mean + 0.02`.
  - Sub-noise is inconclusive per one-seed-only rule.
- **Self-check** (§5): flag OFF reproduces baseline (asserted
  via the `if self.use_cross_block_score_share:` gate — branch
  never taken ⇒ forward bit-identical). Treatment path runs the
  score blend and `MinimalLLM(C())` build-smoke constructs on
  CPU. Step-0 max-abs-diff across all 12 blocks < 1e-4 (the
  reviewer-precise wording for `sigmoid(-10) ≈ 0`, not the
  literal "bit-identical" wording that 188's reviewer flagged).
- **α readout**: append
  `trt.transformer_blocks[i].attention.score_share_alpha_raw.detach().cpu().sigmoid()`
  to the run artifact (12 values, one per block) so a null is
  interpretable (did the optimizer not move α? did it move to
  the same value across blocks? did it diverge per-block?).
