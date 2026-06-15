# Plan — 182 per-head-window

## Flag

`use_per_head_window: bool = False`, on `MultiHeadAttention.__init__` in
`models/layers.py` (pass-through from `TransformerBlock` → inner MHA at the
two `models/llm.py` MHA-construction sites and the `TransformerBlock.__init__`
MHA construction site). Surface flag on `Tiny1M3MConfig` in
`configs/llm_config.py`; the `Tiny1M3MPerHeadWindowConfig` subclass flips
it on.

## Change

- **`models/layers.py`**:
  1. `MultiHeadAttention.__init__` — add `use_per_head_window: bool = False`
     to the kwargs (alongside `use_mqa_gated`, `use_moa`, etc.). When
     `use_per_head_window=True`, allocate
     `self.head_window_logit = nn.Parameter(torch.full((n_heads,), 10.0))`.
     Init `10 ⇒ sigmoid(10) ≈ 0.99995 ⇒ W_h/2 = T · sigmoid(10) ≈ T −
     0.00005·T > T − 1 = max|t − s|` so the penalty term is identically 0 at
     fp32 and the step-0 forward is byte-identical to the no-flag baseline.
     When off, the `Parameter` is never registered and the forward branch is
     never taken — baseline path bit-identical.
  2. `MultiHeadAttention.forward` — add `use_per_head_window` to the
     manual-path dispatch condition (line ~3387) so the manual attention
     branch is taken when the flag is on (the spec calls for the
     `score -= 1e9 · relu(rel_dist − half_w)` form, which requires a
     score-space op, not a SDPA `attn_mask`).
  3. In the manual branch, after the causal mask is applied
     (`scores = scores.masked_fill(~window, -1e9)`), apply the per-head
     penalty: `half_w = T · sigmoid(self.head_window_logit)` (shape `[H]`,
     broadcast to `[B, H, 1, 1]`); `rel_dist = |ar_t − ar_s|` (shape
     `[T, T]`); `scores -= 1e9 · relu(rel_dist - half_w)`. The penalty
     is identically 0 at fp32 at step 0 (half_w ≈ T − 0.00005·T > T − 1,
     so `relu(rel_dist − half_w) = 0` for every valid `(t, s)` pair),
     so softmax and the rest of the forward are bit-identical to baseline.
     The penalty uses `1e9` (fp32-clean, no `−∞`, no NaN risk — matches
     154-rebased-attn's rebased-softmax style).
- **`configs/llm_config.py`**:
  - Add `use_per_head_window: bool = False` to `Tiny1M3MConfig` (alongside
    `use_mqa_gated`, etc.).
  - Add a `Tiny1M3MPerHeadWindowConfig` subclass that flips
    `use_per_head_window=True`.
- **`models/llm.py`** — thread `use_per_head_window` through both MHA
  construction sites (the YOCO upper-half branch around line 698 and the
  standard `TransformerBlock` branch around line 994).

Step-0 (flag OFF) is byte-identical: no `Parameter` registered, no branch
taken, the manual-path dispatch list does not include the flag, the
forward path is exactly the baseline.

## Control

- A/B at `tiny1m3m`, seed 42 (one seed only — see pipeline protocol).
- Control: `Tiny1M3MConfig` (baseline, no flags).
- Treatment: `Tiny1M3MPerHeadWindowConfig` (`use_per_head_window=True`).
- Tier: `tiny1m3m`.

## Cost

- Params: H=4 heads × n_layers=12 blocks = **48 extra parameters**
  (+0.005% of 0.94M). The single scalar `head_window_logit` per MHA is
  the only new `Parameter` — no new `nn.Linear`, no new projection.
- FLOPs: in the flag-on path, the manual attention branch is taken
  (instead of SDPA's `is_causal=True` flash path). The added work is
  ~1 matmul (Q·Kᵀ) + 1 `1e9·relu(...)` subtract + 1 softmax = a small
  overhead per layer per step. At tiny1m3m / T=2048 / H=4 / B=8: manual
  branch adds ~3 ms/layer/step on CPU; on RTX 3060 it's well under 1 ms
  per layer per step. No new FLOPs in the flag-off path.
- Memory: 48 floats (~192 B). Negligible.

## Run

- Tier: `tiny1m3m`. Seed: 42 (one seed only).
- Command (via the daemon): `python _arq_182-per-head-window.py`. The
  daemon wraps it in the standard two-ctrl A/B (`baseline.sh check`
  returns `CACHED` on the current cache, so this run is treatment-only).
- Wall-clock: ~5-7 min for the single treatment run (matches the
  closed per-head scalars at this tier).
- Pass / fail bar (verbatim from `idea.md`):
  - **NULL band** (per-head scalar null cluster): `|trt − cached_baseline| < 0.01`.
  - **WIN pass bar** (plan-side): `trt ≤ cached_baseline − 0.01`
    (matches the magnitude of 016-qk_norm's Δ=−0.0138 at the same tier).
  - **WIN cache rule** (cache-authoritative, `BASELINE-CACHE-DESIGN.md`):
    `trt < cached_val_mean − noise_band`. Re-pull from
    `autoresearch/baseline-cache.json` on run day — the cache has moved
    multiple times across the last week (6.4394 → 6.4504 → 6.4447 →
    6.4346 → 6.4455 → 6.3988). **As of 2026-06-15** (r1 self-pull):
    `cached_val_mean = 6.3988`, `val_std = 0.0088`,
    `noise_band = max(0.04, 2·0.0088) = 0.04` ⇒ WIN iff
    `trt < 6.3588`. **The plan-mirror numbers in the spec are the source
    of truth on run day** — `evidence.md` cites whichever version of the
    cache was current when the run was judged.
  - **Two-ctrl rule** (when running live, not cached): the WIN must
    also be strictly less than BOTH same-session ctrls (per the §2
    two-ctrl rule used by 143-shortconv / 131-layer-drop). If `trt`
    beats the cached mean but lands inside the ctrl pair, that is
    **DRIFT**, not WIN.

## Self-check

Before release:

1. Build-smoke: `python autoresearch/bin/_box_smoke.py _arq_182-per-head-window.py`
   must print `SMOKE_OK` (verifies the stub defines `C` and
   `MinimalLLM(C())` constructs on CPU without error).
2. Step-0 byte-identical: run the no-flag baseline and the flag-on
   treatment with the same input on CPU, then confirm
   `max_abs_diff(logits_baseline, logits_treatment) < 1e-6` at fp32.
   Mirrors 154-rebased-attn's step-0 identity test (the math guarantees
   it: at init `W_h/2 ≈ T − 0.00005·T > T − 1 = max|t − s|`, so the
   penalty is identically 0 everywhere and softmax is unchanged).