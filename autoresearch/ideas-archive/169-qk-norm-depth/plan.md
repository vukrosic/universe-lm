# Plan — 169 qk-norm-depth

## Flag
`use_qk_norm_depth: bool = False` — default OFF.

- `configs/llm_config.py` field on `LLMConfig` (declared after `use_qk_layernorm` /
  `use_v_layernorm` / `use_q_only_norm` / `use_k_only_norm` block, around the
  closed-016 siblings at lines ~1683-1695).
- `models/layers.py` kwarg `use_qk_norm_depth: bool = False` on
  `MultiHeadAttention.__init__` (alongside the 016 / 162 / 165 knobs near
  lines ~682-895) and on `TransformerBlock.__init__` (pass-through near
  lines ~3256-3298).
- `models/llm.py` reads via `getattr(config, "use_qk_norm_depth", False)`
  (next to `use_qk_layernorm` read at line ~460) and threads it into the
  four `TransformerBlock(...)` / `MultiHeadAttention(...)` construction
  sites at lines ~702-705 and ~961-965.

## Change

### Option A — locked (per idea.md)
Keep 016's per-head `self.q_norm = RMSNorm/LayerNorm(d_k)` and
`self.k_norm = RMSNorm/LayerNorm(d_k)` modules intact (existing 016
setup at `models/layers.py:1071-1072`). Add one new scalar parameter
per MHA — `self.qk_norm_scale = nn.Parameter(torch.ones(()))` — and
in `forward`, after the existing per-head RMSNorm call and BEFORE the
QK matmul, multiply both `Q` and `K` (and the MoA `extra_K` for
consistency) by `self.qk_norm_scale`.

The `α_l = 1.0` init ⇒ the multiplicative gain is exactly the identity
in fp32 (`1.0 * x = x` exactly) ⇒ **byte-identical forward to 016's
step-0 (max-abs-diff = 0.0)** — no tolerance needed for the 169-vs-016
comparison.

The scalar is applied as a SINGLE statement after the norm+RoPE block
(at line ~2288, before the q_gain block), so it works uniformly across
the three norm+RoPE branches (`use_nope_or_cope`, `use_qk_norm_post_rope`,
and the default pre-RoPE branch) — at α=1.0 the multiplication commutes
with the post-norm post-RoPE ops (q_gain, GQA repeat, q_temp_token, etc.)
in fp32.

For the MoA `extra_K` (lines ~2580-2617): multiply `extra_K` by the
same `self.qk_norm_scale` after the MoA's per-K RoPE+norm and before
the `K_all = torch.cat([K.unsqueeze(1), extra_K], dim=1)` concat, so
the MoA's extra K sees the same per-block normalization strength as
the standard K.

Three `assert not (self.use_qk_norm_depth and self.X)` lines at the top
of `MultiHeadAttention.forward` (mirroring the existing
`assert not (self.use_cope and self.use_qk_norm_post_rope)` at line
~2040), for `use_q_only_norm`, `use_k_only_norm`, and
`use_qk_norm_post_rope`. Combining per-block scaling with any of these
restructures the lever and must fail loud at construction/runtime.

## Control
- Control: `Tiny1M3MQKNormConfig` (the 016 WIN config — sets
  `use_qk_layernorm=True`).
- Treatment: `Tiny1M3MQKNormDepthConfig` (= `Tiny1M3MQKNormConfig` with
  `use_qk_norm_depth=True`).
- Seed: **42** (always one seed — see `code-implementer.md` §"ONE SEED
  ONLY").
- Tier: **tiny1m3m** (standard recipe).
- The Δ is depth-conditional-on-016, NOT "QK-norm yes/no" — the
  control is the 016 WIN, not unmodded-nanogpt.

## Cost
- Params: **+12 scalars total** (one per block × 12 blocks, +0.001%
  of 0.94M). Single `nn.Parameter(torch.ones(()))` per MHA.
- FLOPs: ~+2 elementwise multiplies per block per forward (negligible
  at tiny1m3m — one extra multiply on Q and one on K, broadcasted
  against the `[B, H, T, d_k]` tensor).
- Memory: 0 (the scalar is a single fp32 weight, lives in the MHA's
  Parameter dict alongside `q_gain` / `k_gain` / etc.).

## Run

Standard tiny1m3m seed-42 command (per `prompts/runner.md`):

```bash
cd /root/universe-lm && \
LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
/venv/main/bin/python -m training.trainer \
  --config_class autoresearch.configs.tiny1m3m.Tiny1M3MQKNormDepthConfig \
  --activations "use_qk_norm_depth=True" \
  --seed 42 --steps 3000 --batch_size 32
```

Expected wall-clock: ~6 min on V100 (standard tiny1m3m baseline).

## Pass / fail bar (copied from idea.md)

- **Control**: `Tiny1M3MQKNormConfig` (016 WIN, `trt=6.3906`).
- **PASS**: 169 trt val < 016 control val by ≥ **0.005** (mirrors 016's
  own plan bar; clears the tiny1m3m ±0.04 box noise band by ≥2×).
- **NULL**: |169 val − 016 val| < 0.005 ⇒ per-head-shared scale is
  sufficient at 0.94M, depth-conditional not the binding axis.
- **DRIFT**: 169 val ≥ 016 val + 0.005 ⇒ per-block DOF adds noise.
- **CRASH / NaN / OOM** → `needs-recode` (round 1, inside budget).

## LoC budget
~40 lines total (well under the 200 LoC cap):
- `models/layers.py`: ~25 lines (kwarg + register + 3 asserts + single
  forward multiply + MoA extra_K multiply + TransformerBlock pass-through).
- `configs/llm_config.py`: ~15 lines (LLMConfig field + dataclass
  subclass).
- `models/llm.py`: ~6 lines (read + 4 thread sites).

## Step-0 identity
- **Flag OFF**: no `qk_norm_scale` Parameter registered, no branch
  taken, baseline path bit-identical.
- **Flag ON at step 0**: `qk_norm_scale = 1.0` ⇒ `Q = Q * 1.0 = Q` and
  `K = K * 1.0 = K` exactly in fp32 ⇒ **byte-identical forward to
  016's step-0 (max-abs-diff = 0.0)**. The chosen control is 016
  (the WIN config), so this comparison is the relevant one. The
  `fp32 max-abs-diff < 1e-3` tolerance only applies to the
  016-vs-unmodded RMSNorm rescaling (already accepted by 016's spec),
  not the 169-vs-016 comparison.