# Plan — 181 cross-head-rmsnorm

## Flag
`use_cross_head_rmsnorm: bool = False` on `LLMConfig` (default OFF —
baseline path bit-identical). Treatment subclass
`Tiny1M3MCrossHeadRMSNormConfig(Tiny1M3MConfig)` with
`use_cross_head_rmsnorm: bool = True`, `@dataclass`-decorated (per the
162/165/155/161/176 precedent that bare-class annotation breaks
dataclass field inheritance).

## Change

- **`configs/llm_config.py`** — add `use_cross_head_rmsnorm: bool = False`
  field on `LLMConfig` immediately after `use_head_gain` (sibling of
  the 160 flag at `:176`). Add `@dataclass class
  Tiny1M3MCrossHeadRMSNormConfig(Tiny1M3MConfig)` with
  `use_cross_head_rmsnorm: bool = True`, sibling of
  `Tiny1M3MVPreAVNormConfig` (the 176 subclass is the closest
  geometric match — also H × (1 α + d_k γ) per block).

- **`models/layers.py`** — `MultiHeadAttention.__init__`:
  - Add `use_cross_head_rmsnorm: bool = False` kwarg immediately
    after `use_head_gain` (line ~809).
  - Register two new parameters when the flag is on:
    - `self.cross_head_rmsnorm_alpha_raw = nn.Parameter(torch.full((self.n_heads,), -1e-3))`
      — per-head gate-α, init `−1e-3` ⇒ `relu(−1e-3) = 0` exactly ⇒
      step-0 forward is byte-identical to baseline (the relu
      clamps negative to zero — bit-exact zero at init, not ≈ 0).
    - `self.cross_head_rmsnorm_gain_raw = nn.Parameter(torch.zeros(self.n_heads, self.d_k))`
      — per-(h,k) post-normalization gain, init 0 ⇒
      `γ_h[k] = 1 + tanh(0) = 1.0` exactly at step 0.
  - When the flag is off, set `self.cross_head_rmsnorm_alpha_raw =
    None, self.cross_head_rmsnorm_gain_raw = None` (attribute
    lookups stay valid even when the flag is off; the forward
    `if` guard short-circuits).

- **`models/layers.py` — `MultiHeadAttention.forward`** (apply site,
  immediately BEFORE the existing `if self.use_head_gain:` branch at
  line 3890, i.e. on `attn_output` of shape `[B, H, T, d_k]` after
  the SDPA call):
  ```python
  if self.use_cross_head_rmsnorm:
      # 181 — Cross-Head Channel RMSNorm. Normalize each token's
      # attention output ACROSS HEADS (rather than across the
      # concatenated d_model axis used by standard post-AV
      # RMSNorm) so all H heads land on the same per-(t, k)
      # scale before the W_O projection. Apply the gated blend:
      #   α_h = relu(α_raw_h)            # 0 at init ⇒ identity blend
      #   rms = sqrt(mean(out² along H) + ε)   # one scalar per (b, t, k)
      #   gain = 1 + tanh(γ_raw_h[k])    # 1 at init ⇒ identity gain
      #   out = (1 − α_h)·out + α_h·(out / rms)·gain
      # Init α=0, gain=1 ⇒ out unchanged exactly ⇒ byte-identical
      # to baseline at step 0. Composes with the post-AV per-head
      # scalar gain (`use_head_gain`, line 3890) by being
      # multiplicative in series, but the mutual-exclusion asserts
      # forbid combining them in a single run.
      rms = (attn_output.pow(2).mean(dim=1, keepdim=True) + 1e-6).sqrt()
      alpha = F.relu(self.cross_head_rmsnorm_alpha_raw).view(
          1, self.n_heads, 1, 1
      )
      gain = 1.0 + torch.tanh(
          self.cross_head_rmsnorm_gain_raw
      ).view(1, self.n_heads, 1, self.d_k)
      attn_output = (1.0 - alpha) * attn_output + alpha * (
          attn_output / rms
      ) * gain
  ```
  This sits between the SDPA output (line ~3878) and the existing
  `use_head_gain` branch (line 3890), so the two compose by being
  multiplicative in series; the mutual-exclusion asserts forbid
  both-on in a single run.

- **`models/layers.py` — `MultiHeadAttention.forward`** (mutual
  exclusion asserts, sibling of the `use_cope ∧
  use_qk_norm_post_rope` block at line 2597 and the
  `use_v_rmsnorm` asserts at line 2631):
  ```python
  assert not (self.use_cross_head_rmsnorm and self.use_head_gain), (
      "use_cross_head_rmsnorm=True is mutually exclusive with "
      "use_head_gain=True (both post-AV; the composition restructures "
      "the lever — turn 160 OFF to isolate 181)."
  )
  assert not (self.use_cross_head_rmsnorm and self.use_attn_output_gate), (
      "use_cross_head_rmsnorm=True is mutually exclusive with "
      "use_attn_output_gate=True (closed-045 per-head scalar gain; "
      "the composition restructures the lever)."
  )
  assert not (self.use_cross_head_rmsnorm and self.use_gated_attn), (
      "use_cross_head_rmsnorm=True is mutually exclusive with "
      "use_gated_attn=True (closed-024 input-conditional gate; "
      "the composition restructures the lever)."
  )
  ```

- **`models/layers.py` — `TransformerBlock.__init__`** — add
  `use_cross_head_rmsnorm: bool = False` kwarg, sibling of
  `use_head_gain` (line ~4107). Read from kwargs, pass through to
  the inner `MultiHeadAttention(...)` constructor at the threading
  site.

- **`models/llm.py` — `MinimalLLM.__init__`** — add
  `self.use_cross_head_rmsnorm = getattr(config,
  "use_cross_head_rmsnorm", False)` immediately after the existing
  `self.use_head_gain` read. Thread into both
  `TransformerBlock(...)` constructor sites (the standard block
  site at line ~991 where `use_head_gain=getattr(...)` lives, and
  the parallel/duplicated site at line ~1166 where the
  `use_v_rmsnorm` plumbing mirrors).

**Step-0 identity**: at init `α_raw_h = -1e-3` for all heads ⇒
`relu(-1e-3) = 0` exactly ⇒ `attn_output = (1 − 0)·attn_output +
0·... = attn_output` *exactly* ⇒ **byte-identical to baseline at
step 0 (max-abs-diff = 0.0)**.

## Control

- **Control**: unmodded `Tiny1M3MConfig` (no cross-head RMSNorm).
  The daemon owns the ctrl (per `RUN-CONTRACT.md` §"Control is the
  daemon's, not the idea's"); we ship only the treatment stub.
- **Treatment**: `Tiny1M3MCrossHeadRMSNormConfig` with
  `use_cross_head_rmsnorm: bool = True`. A/B at tiny1m3m, seed 42,
  single seed per the one-seed-only rule.
- **Cached baseline** (per `autoresearch/baseline-cache.json`,
  box `5b8a7fea8963`, measured 2026-06-15T05:58:50Z):
  `val_mean = 6.3988 ± 0.0088`, `noise_band = 0.04`.

## Cost

- **Params**: H=4, d_k=16, n_layers=12. Per block: `H × (1 α + d_k
  γ) = 4 × (1 + 16) = 68` params. Across 12 blocks: `12 × 68 = 816`
  params. That's +0.087% of the 0.94M baseline (949,056 params).
  Mirrors 176-v-pre-av-norm's exact param count. Well under the
  per-lever budget.
- **FLOPs**: ~1 RMSNorm per (b, t, k) per block (mean of H=4 squares
  + sqrt + division + per-(h,k) tanh + blend). Negligible vs the
  dominant FFN/AV matmul cost.
- **Memory**: two new tensors per MHA of shape `[H]` and `[H, d_k]`,
  both fp32, ~272 bytes/block. Trivial.

## Run

- **Artifact**: `_arq_181-cross-head-rmsnorm.py` at repo root,
  imports `Tiny1M3MCrossHeadRMSNormConfig as C` from
  `configs.llm_config`, dispatches `train_llm.main()` with
  `--config_class __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false` (mirror the
  162/165/169/170/176 `_arq_*.py` pattern). The stub's
  docstring should include a step-0 byte-identity check
  (`trt_step0_logits == ctrl_step0_logits` byte-exact, expected
  max-abs-diff = 0.0) — not run by default; it documents the
  expected step-0 behavior.
- **Job**: `/venv/main/bin/python _arq_181-cross-head-rmsnorm.py`
  with `JOB_TIMEOUT=12m` (tiny1m3m runs in ~2-6 min; the cap keeps
  a hung treatment from burning the box).
- **Descriptor**: `autoresearch/ideas/181-cross-head-rmsnorm/run.json`
  — `{"name": "181-cross-head-rmsnorm", "arq_file":
  "_arq_181-cross-head-rmsnorm.py", "job_timeout": "12m"}`.
- **Val loss**: read from
  `~/arq/logs/181-cross-head-rmsnorm.log` (`grep "val_loss"`).
- **Pass/fail bar** (from `idea.md` §"Pass / fail bar"):
  - **WIN**: treatment val ≤ `val_mean − 0.005` (≤ **6.3938**)
    AND clears the ±0.04 noise band by ≥8×. Mirrors 016's bar.
  - **NULL**: |treatment val − val_mean| < 0.005 (inside
    [6.3938, 6.4038]).
  - **DRIFT**: treatment val ≥ `val_mean + 0.005` (≥ **6.4038**).
  - **Crash / NaN / OOM** → `needs-recode` (round 1, inside
    budget).
  - **Sub-noise** (|Δval| < 0.005 but not DRIFT) is
    **INCONCLUSIVE** on one seed per the one-seed-only rule — do
    **not** re-run with extra seeds.

**LoC budget**: ~20 lines normalization + plumbing + apply site,
~12 lines config subclass + threading, ~3 lines asserts. Total
~50 LoC, well under the 200 LoC cap. No new dependencies.

## Coordination note (per prompt §2)

Working tree has many sibling-flag hunks landed. `git status`
confirms no overlap with `models/layers.py` /
`configs/llm_config.py` / `models/llm.py` for the
`use_cross_head_rmsnorm` flag. No conflict.

## Non-blocking note (mirrors 176 review precedent)

The line-number references in `idea.md` for the existing
`use_head_gain` apply site and the `use_cope ∧ use_qk_norm_post_rope`
assert pattern drifted against the current working tree (the
reviewer flagged this in r2). The semantic targets are
unambiguous: locate via `grep -n "if self.use_head_gain" models/layers.py`
and `grep -n "use_cope and use_qk_norm_post_rope" models/layers.py`,
and the implementation places 181's apply site and asserts at
those exact semantic positions.
