# Plan — 020 Forgetting Transformer (FoX)

## Flag
- `use_fox: bool = False` on `LLMConfig` (`configs/llm_config.py:179`),
  threaded through `TransformerBlock` (`models/layers.py:1809`,
  `models/layers.py:1847`) into `MultiHeadAttention.__init__`
  (`models/layers.py:447`, `models/layers.py:605`).
- Trt config class: `Tiny1M3MFOXOnFireConfig`
  (`configs/llm_config.py:713-735`) — sets
  `use_fire_pe = True` AND `use_fox = True`.
- Built module: `models/fox.py:FoX(d_model, n_heads)` (instantiated
  lazily in MHA only when `use_fox=True`; never called when off →
  baseline path bit-identical).

## Change
**1 new file + 3 wiring touch points.** No new dependencies.

- `models/fox.py` (new, 131 LoC incl. docstring + comments):
  - Per-head gate projection `gate_w: Parameter[H, d_model]` (zero
    init) + `gate_b: Buffer[H]` set to `FOX_BF_INIT = +10.0`.
  - `forward(x)`: `z = einsum("btd,hd->bth", x, gate_w) + gate_b`
    → `log_f = logsigmoid(z)` (≤ 0) → `cum = cumsum(log_f, dim=T)`
    → pad a zero along T → build `D[b,h,i,j] = exp(cum_pad[i+1] −
    cum_pad[j])` via slicing the padded axis → `permute` heads to
    dim 1 → `exp` → mask upper-tri to 0. Output `[B, H, T, T]`,
    strictly lower-tri, ≤ 1 by construction.
  - Math-corrected init `b_f = +10` (NOT r1's `+5`) gives
    `D[0, 2047] = exp(−0.0929) ≈ 0.911` at the real T=2048 — ≤ 9%
    worst-case decay over the full context, verified at smoke
    (`D[0,0,2047,0] = 0.9112`).
- `models/layers.py:447` — add `use_fox: bool = False` kwarg to
  `MultiHeadAttention.__init__` with the design comment.
- `models/layers.py:605-607` — store `self.use_fox = use_fox`; if
  on, build `self.fox = FoX(d_model, n_heads)`.
- `models/layers.py:1422-1432` (FIRE branch) and
  `models/layers.py:1533-1541` (manual branch) — after `softmax`,
  when `self.use_fox`: `d = self.fox(x)` → `attn_w = attn_w * d` →
  `attn_w = attn_w / attn_w.sum(−1, keepdim=True).clamp_min(1e-9)`.
  Then continue to `@V`. Order is `scores += fire_bias → softmax →
  attn_w *= D → row-renorm` when both flags on — strictly
  orthogonal axes (FIRE additive on logits, FoX multiplicative on
  probs).
- `models/layers.py:1441` — add `or self.use_fox` to the manual-
  branch OR (`or self.use_fire_pe` already forces manual for
  `use_cope`/`use_fox`/etc — FoX needs manual because the
  post-softmax multiply doesn't go through SDPA's flash kernel).
- `models/layers.py:1809, 1847` — pass `use_fox` from
  `TransformerBlock` to its MHA child.
- `configs/llm_config.py:179` — declare `use_fox: bool = False` on
  `LLMConfig` (sits next to `use_fire_pe`, `use_cope`).
- `configs/llm_config.py:713-735` — `Tiny1M3MFOXOnFireConfig`
  (`use_fire_pe=True, use_fox=True`); ctrl is
  `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True` (the 009
  WIN signature from `closed.md:40`).
- `tests/test_fox.py` (new, 156 LoC): 5 invariants — no NaN/Inf,
  causal lower-tri, identity-init ≤ 1 and within D_min bound,
  W_f-perturbation makes head 0 differ (wiring live), MHA step-0
  output within `1e-2` of `use_fox=False` baseline (the reviewer's
  r2 nit on tolerance — see §Self-check).

## Control
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (the 009 WIN FIRE-equipped baseline, val 6.3234 in `closed.md:40`).
- **Trt**: `Tiny1M3MFOXOnFireConfig` — same recipe as ctrl + `use_fox=True`.
- **Seed**: 42 (one seed only — see `feedback-one-seed-only.md`).
- **Tier**: tiny1m3m.
- **Pass bar** (copied from `idea.md:88-96`):
  - **Win**: `trt_val < ctrl_val − 0.02`.
  - **Null**: `|trt_val − ctrl_val| < 0.02`.
  - **Fail**: `trt_val > ctrl_val + 0.01`.

## Cost
- **Params**: `gate_w` = `H · d_model` (e.g. 8·256 = 2,048 / layer at
  tiny1m3m). `gate_b` = `H` (8 / layer). Total `+8 · d_model + 8`
  per layer → at 6 layers, ~12.3K params (+0.2% over the
  tiny1m3m's ~6M params). Negligible.
- **FLOPs**: attention path is dominated by the softmax. The FoX
  path adds (a) one `[B, T, d] · [H, d] → [B, T, H]` matmul (1
  einsum, ~`B · T · d · H` FLOPs), (b) one `cumsum` over T, (c) one
  `[B, H, T, T] · [B, H, T, T]` elementwise multiply, (d) one
  row-renorm `[B, H, T, T] / [B, H, T, 1]`. At B=8, T=2048, H=8,
  d=256: ~67M extra FLOPs / layer → ~400M / 6 layers (≈ 1% of
  the attention forward). Plan's "5% attention forward" estimate
  is an upper bound; actual is closer to 1%.
- **Memory**: D is `[B, H, T, T]` in fp32 = `8·8·2048·2048·4B ≈
  1 GB` *per layer* if materialized. We allocate it once per
  forward via `cum_pad` slicing + `permute().exp()` — same
  activation cost as the attention scores tensor, so the
  attention path's memory grows by ~50% (scores + D, both
  `[B, H, T, T]`). The runner uses gradient checkpointing
  already; no additional change.

## Run
- **Command**: `python3 train_llm.py --config
  Tiny1M3MFOXOnFireConfig --seed 42 --tier tiny1m3m` (ctrl drops
  the `use_fox=True` — pass `--no_fox` once the runner supports
  that flag, or set `use_fox=False` on the config dataclass and
  re-run with `use_fire_pe=True` for the ctrl). The config class
  is the A/B handle, not a CLI flag — see `vast-runner-harness.md`.
- **Tier**: tiny1m3m (single seed, no sweep).
- **Expected wall-clock**: ≈ 4–6 hours on the Vast box (the
  Tiny1M3MConfig tier baseline). With the FoX path active, add
  ~5% to attention compute → ~6.5 h worst case.
- **Pass/fail bar**: copied from `idea.md:88-96` (Win Δ ≤ −0.02,
  Null |Δ| < 0.02, Fail Δ > +0.01). A null is informative — it
  partitions "FIRE's additive bias already saturates the
  relative-position benefit at our scale; multiplicative mass
  control is sub-threshold at tiny1m3m depth/length."

## Self-check (per code-implementer.md §5)
- [x] `use_fox=False` path: FoX module is not instantiated
  (`if self.use_fox: self.fox = FoX(...)` in `models/layers.py:606`),
  forward never calls `self.fox`, no extra params allocated, no
  extra FLOPs. Baseline path is bit-identical to a pre-flag build.
- [x] `use_fox=True` path at step 0: tested in
  `tests/test_fox.py::test_step0_attention_output_unchanged` —
  MHA output is within `1e-2` of the `use_fox=False` baseline
  (1e-2 chosen per the reviewer's r2 nit: the r1 plan's 1e-5 was
  too tight; the smoke test pins 1e-2 as the defensible ceiling,
  and the actual measured drift is well below it on a uniform
  softmax).
- [x] `tests/test_fox.py::test_identity_init_close_to_ones`:
  verifies the 9% identity bound (`D_min = exp(T·log f)`); verified
  numerically: `D[0,0,2047,0] = 0.9112` at T=2048.
- [x] `tests/test_fox.py::test_wiring_live_with_Wf_perturbation`:
  a +1 perturbation on head 0's `gate_w` changes head 0's D and
  leaves heads 1..5 unchanged. Confirms the projection is wired
  into the kernel (the lever is trainable, not a constant).
- [x] `tests/test_fox.py::test_causal_lower_triangular`: D upper-
  triangle is exactly 0 (the attention mask in MHA already zeros
  the post-softmax A there; the explicit mask is belt-and-suspenders
  to avoid any fp32 edge case leaking upper-tri mass into the
  row-renorm).
- [x] All 5 tests pass: `pytest tests/test_fox.py -v` → 5 passed.

## Coordination note
`git diff models/layers.py configs/llm_config.py` is empty at the
start of this pass — no other worker has touched the shared files
in parallel. No conflict. Per `project-parallel-ai.md`, this is the
expected state for idea 020.
