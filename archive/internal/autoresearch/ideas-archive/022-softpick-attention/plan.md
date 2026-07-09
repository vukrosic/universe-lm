# Plan — 022 Softpick (rectified-softmax attention, sink-free normalization)

## Flag
- `use_softpick: bool = False` on `LLMConfig` (`configs/llm_config.py`,
  sits next to `use_fox` at line 179), threaded through
  `TransformerBlock.__init__` (`models/layers.py:1809-1847`) into
  `MultiHeadAttention.__init__` (`models/layers.py:447-605`).
- Trt config class: `Tiny1M3MSoftpickOnFireConfig`
  (`configs/llm_config.py:773-780` neighborhood, new dataclass right
  after `Tiny1M3MVQGainSWAHighRoPE250KConfig`) — sets
  `use_fire_pe = True` AND `use_softpick = True`.
- No new module. The softpick op is a ~12 LoC helper function
  embedded in `models/layers.py` near the swap site (mirrors the
  `models/fox.py` "separate module" pattern but smaller — no
  parameters, no init, no learned state — so a helper is the right
  shape).

## Change
**1 helper in `models/layers.py` + 4 wiring touch points.** No new
dependencies. ~45 LoC total, well under the 200 ceiling.

- `models/layers.py` (new, near the swap site at line 1421, ~12
  LoC): `softpick(scores, mask, eps=1e-6)` — computes
  `exp(x) - 1` in **fp32** (overflow guard for large positive
  scores in fp16/bf16), then
  `num = relu(z) * mask_float` and
  `den = |z| * mask_float + eps`, returns
  `num / den` cast back to model dtype. Both numerator and
  denominator are multiplied by the SAME 0/1 mask so masked
  positions contribute zero to both (the bug class the spec
  calls out at `idea.md:32-45`). The canonical mask is the
  `[B, 1, T, T]` boolean already computed in the FIRE branch
  (`scores.masked_fill(~window, -1e9)` produces the
  `mask_float = window.to(z.dtype)` by reusing the existing
  `window` tensor at line 1417-1419).
- `models/layers.py:1421` (FIRE branch only) — replace
  `attn_w = torch.softmax(scores, dim=-1)` with
  `attn_w = softpick(scores, window.view(1, 1, T, T))`. The
  same `window` tensor that's already been used to fill the
  scores is reused as the mask argument. This is the ONLY
  softmax call reached in the A/B (ctrl and trt both have
  `use_fire_pe=True` → manual path; SDPA and the manual-FoX
  branch at `layers.py:1435+` never fire).
- `models/layers.py:1435-1444` — add `or self.use_softpick` to
  the manual-branch OR list as a **defensive** fallback. The
  trt uses FIRE, so the swap site at 1421 is reached; if some
  non-FIRE path ever tried to use softpick (e.g. a future
  combination with another flag), it would fall back to softmax
  here. Same shape as the `or self.use_cope` and
  `or self.use_fox` entries in the same OR list.
- `models/layers.py:447, 605` — add `use_softpick: bool = False`
  kwarg to `MultiHeadAttention.__init__` with the design
  comment, and store `self.use_softpick = use_softpick`.
- `models/layers.py:1809, 1847` — add `use_softpick` to
  `TransformerBlock.__init__` and pass to its MHA child.
- `models/llm.py:224-225` neighborhood — read
  `self.use_softpick = getattr(config, "use_softpick", False)`
  and pass through to `TransformerBlock`.
- `configs/llm_config.py:179` neighborhood — declare
  `use_softpick: bool = False` on `LLMConfig` (sits next to
  `use_fox`).
- `configs/llm_config.py:773-780` neighborhood — new dataclass
  `Tiny1M3MSoftpickOnFireConfig` (sets
  `use_fire_pe=True, use_softpick=True`).
- `configs/__init__.py:13, 102` neighborhood — export the new
  config class.
- `tests/test_softpick.py` (new, ~80 LoC): 5 invariants —
  (i) no NaN/Inf on a non-trivial random input,
  (ii) all-`window=True` mask gives a valid row-stochastic
  result that sums to 1 (or ≤ 1 when at least one score ≤ 0,
  per `idea.md:122`),
  (iii) step-0 MHA: build trt model, run one fwd+bwd, assert
  loss is finite AND grads on `q_proj.weight`,
  `k_proj.weight`, `v_proj.weight` are non-zero (the
  lever-is-dead-on-arrival guard from `idea.md:54-58, 117-122`),
  (iv) mask interaction: place a real key inside the SWA window
  and several masked keys outside it; assert the softpick
  output is zero on masked positions AND the denominator is
  not polluted by masked positions (the bug class from
  `idea.md:32-45`),
  (v) identity-when-off: trt model with the flag forcibly
  turned off must produce a row-renormed softmax equivalent
  to the plain baseline (the bit-identical-when-off pin from
  `idea.md` and the prompt §4).

## Control
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (the 009 WIN FIRE-equipped baseline, val 6.3234 in
  `closed.md:40`).
- **Trt**: `Tiny1M3MSoftpickOnFireConfig` — same recipe as ctrl
  + `use_softpick=True`.
- **Seed**: 42 (one seed only — see `feedback-one-seed-only.md`).
- **Tier**: tiny1m3m.
- **Pass bar** (copied from `idea.md:99-108`):
  - **Win**: `trt_val < ctrl_val − 0.005`. The −0.005 bar (not
    −0.02) reflects the bet shape — softpick is a
    parameter-free normalization tweak, not a structural
    change, so a modest gain is the realistic win shape and we
    don't want a real effect lost in the noise floor.
  - **Null**: `|trt_val − ctrl_val| < 0.01` (sub-noise; the
    lever does not fire on top of FIRE at this scale).
  - **Fail**: `trt_val > ctrl_val + 0.01` (worse than baseline
    by more than half the ctrl-gap — sink-removal is hurting).
- **Step-0 smoke check (pre-run, runnable on the box)**:
  build the trt model (`Tiny1M3MSoftpickOnFireConfig`), run one
  fwd+bwd on a tiny batch, assert (a) loss is finite, (b)
  grads on `q_proj.weight`, `k_proj.weight`, `v_proj.weight`
  are non-zero, (c) `attn_w` sums per row to ≤ 1 (≤ 1, not
  == 1, because softpick permits zero total mass). If (b)
  fails the lever is dead on arrival (zero attn output ⇒
  zero grad on Q/K/V) and the A/B is malformed — the runner
  must NOT proceed to a full training run.

## Cost
- **Params**: 0. Softpick is a function swap on
  `exp(x) - 1` / `|exp(x) - 1|`. No new weights, no new
  buffers, no new init. Bit-identical to baseline at every
  init.
- **FLOPs**: roughly equivalent to `torch.softmax`. The
  `exp(x) - 1` op adds one subtract and one `relu`; the
  denominator does an absolute-value and a sum. The `mask
  * z` and `mask * |z|` are two extra elementwise multiplies
  on a `[B, H, T, T]` tensor — at tiny1m3m (B=32, H=8,
  T=2048) that's ~1M extra ops per layer per step, ~6 layers
  ≈ 6M ops. Negligible vs. the 2·d·T·T = ~8M QK-product ops
  per layer. The fp32 cast on the `exp - 1` is the only
  meaningful overhead, and it's pinned in the spec.
- **Memory**: one extra fp32 tensor of shape `[B, H, T, T]`
  for the `exp - 1` intermediate, cast back to model dtype
  before returning. At tiny1m3m (B=32, H=8, T=2048) that's
  `32 * 8 * 2048 * 2048 * 4 bytes = 4.3 GB` worst case
  (fp32) — but the test runs at much smaller T (e.g. T=64 in
  `test_fox.py`) and the training tier is also smaller
  per-batch than this worst case. If memory becomes an
  issue, the helper can be applied per-head in a loop, but
  the plan doesn't anticipate needing that for tiny1m3m.

## Run
- **Command**: invoke the Vast runner harness (see
  `vast-runner-harness.md`) with the new config class on
  `/venv/main/bin/python`, seed 42, tier tiny1m3m. The
  control's prior val 6.3234 is the baseline anchor.
- **Tier**: tiny1m3m (one tier only — no tier promotion, no
  multi-seed sweep).
- **Seed**: 42 (one seed only).
- **Expected wall-clock**: same as the FIRE-equipped ctrl
  (no new FLOPs in the attention path to first order; the
  fp32 cast on `exp - 1` is the only meaningful addition,
  and it's amortized over the same matmul-flash budget).
- **Pass/fail bar**: copied from `idea.md:99-108` and from
  the Control block above.

## Self-check (prompt §5)
- Flag OFF reproduces the control (no numeric drift) —
  `tests/test_softpick.py::test_identity_when_off` builds the
  trt config, monkey-patches `mha.use_softpick = False`, and
  asserts the resulting attention weights are bit-equivalent
  to the plain `torch.softmax` baseline on a fixed input.
- The treatment path actually exercises the new code —
  `test_step0_grads_live` runs an fwd+bwd on the trt model
  and asserts non-zero grads on Q/K/V; the
  `test_mask_does_not_pollute_denominator` test places
  masked keys outside the SWA window and asserts the
  denominator is unaffected.
- `plan.md`'s pass/fail bar matches `idea.md` verbatim
  (Win −0.005, Null ±0.01, Fail +0.01).

## Open coordination note
No conflicts in `models/layers.py`, `models/llm.py`, or
`configs/llm_config.py` at the time of claim — `git diff`
across these files is clean.

## r1 recode (2026-06-10)
- Codereview r1 blocking finding: `Tiny1M3MSoftpickOnFireConfig`
  extended `Tiny1M3MConfig` (vanilla), silently dropping VQ-gain +
  SWA(512) + RoPE 250K from the ctrl recipe (4-field HP drift).
- Fix applied in `configs/llm_config.py:856` — changed parent
  dataclass from `Tiny1M3MConfig` to
  `Tiny1M3MVQGainSWAHighRoPE250KConfig`. Kept only the two override
  fields (`use_fire_pe=True`, `use_softpick=True`). Updated
  docstring to call out the parent-class choice.
- Re-verified: instantiated ctrl and trt; `use_value_embed`,
  `use_q_gain`, `use_sliding_window=True`, `sliding_window_size=512`,
  `rope_base=250000` now agree across both. Only differences are
  `use_fire_pe` (ctrl gets True at runtime per plan; trt has it in
  the class) and `use_softpick` (the lever).
- Re-ran `pytest tests/test_softpick.py -v` → 8/8 pass.
- Re-ran step-0 smoke on trt: 0.9522M params (slightly higher than
  the prior 0.9491M because V-embed and q_gain are now wired in,
  matching the ctrl), loss=10.80 finite, `qkvo_proj`/FIRE/V-embed
  grads all non-zero across all layers. Lever is alive.

## r2 recode (2026-06-10) — fp32-overflow stabilization
- Runner bounced from `running`: both runs (`022-soft.log` and
  `022-soft-r.log`) NaN'd mid-training at step ~400/732 (last finite
  loss 6.5851). `evidence.md` correctly identified an impl bug, not
  a mechanism failure (the paper is reported drop-in).
- Root cause: `softpick` computed `exp(scores.to(fp32)) - 1.0`
  *without* a per-row max-subtraction. fp32's exp ceiling is x ≈
  88.7 (e^88.7 = fp32 max ≈ 3.4e38). When even a single attention
  score grew past that mid-training, `exp(score) = +inf`, then
  `relu(inf)` in the numerator and `|inf|` in the denominator both
  blow up — `inf / inf = NaN` propagates through `matmul(attn_w, V)`
  to the loss. The mask multiply doesn't save it because the
  overflow is on UNMASKED entries.
- Fix in `models/layers.py:39-86` (softpick helper) — apply the
  closed-form identity:

    relu(exp(x) − 1) / Σ|exp(x) − 1|
    ≡ relu(exp(x − M) − exp(−M)) / Σ|exp(x − M) − exp(−M)|

  for M = per-row max over UNMASKED positions (the masked_fill
  −inf + amax pattern), clamped to ≥ 0. The clamp_min(0) keeps the
  identity exact when M_true ≤ 0 (subtracting 0 is a no-op) and
  bounds `exp(x − M) ≤ 1` and `exp(−M) ≤ 1` when M > 0 — overflow
  becomes impossible. No spec change (the function value is
  mathematically identical at every input, just stably computed).
  No new params, no LR/schedule change, no init drift.
- New regression test `test_no_nan_under_fp32_exp_overflow` in
  `tests/test_softpick.py` — plants scores up to 200 (well past
  fp32 exp ceiling 88.7) and asserts softpick stays finite with
  valid row sums. Pinned so a future refactor can't re-introduce
  the bug.
- Verified `tests/test_softpick.py` → 9/9 pass (8 prior + new
  overflow regression). Numerical equivalence against the naive
  form on normal-range scores: max-diff ≈ 1e-6 (float precision
  noise; mathematically identical). Adversarial check with a
  single planted score=95.0: naive emits NaN, stable returns 1.0
  (correct — only positive score, all mass there).
- Re-ran step-0 smoke on trt: 0.9522M params, loss finite, grads
  finite, finite under 20× weight inflation (synthetic over-stress
  to exercise overflow regime end-to-end).
- LoC delta: +28 LoC inside the softpick helper (mostly comments
  explaining the identity); +35 LoC for the new regression test.
  Active softpick code remains under the 50-LoC cap.
- Coordination: parallel agent moved FoX op to pre-softmax
  (logit-add) at `models/layers.py:1596-1608` and added 029-V-norm
  wiring. Neither conflicts with the softpick fix — the change is
  purely internal to the `softpick` helper at lines 39-86; the
  swap site at line 1622 is unchanged.
