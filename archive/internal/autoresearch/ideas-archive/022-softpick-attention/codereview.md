# Code-review log — 022 softpick-attention

## r3 — 2026-06-10 — verdict: accept

**Round 3, forced call** (3-round cap: accept or reject only). Recoded after
the r2 NaN at step ~400/732 (`evidence.md` — both runs `022-soft.log` and
`022-soft-r.log`). Implementer's fix is a closed-form numerical-stability
identity, NOT a spec change.

**Identity is mathematically correct.** Verified algebraically + empirically:

```
relu(exp(x − M) − exp(−M))            = exp(−M) · relu(exp(x) − 1)
|exp(x − M) − exp(−M)|                = exp(−M) · |exp(x) − 1|
```

Both numerator and denominator scale by `exp(−M)`, so the ratio is invariant
under any M (`exp(−M) > 0` preserves the relu sign). The implementer picks
`M = amax(scores over UNMASKED) clamp_min 0`:

- `M_true ≤ 0` ⇒ clamp activates, subtracting 0 is a no-op, formula reduces
  to `exp(x) − 1` exactly (safe because `x ≤ 0` ⇒ `exp(x) ≤ 1`, no overflow).
- `M_true > 0` ⇒ clamp inactive, `exp(x − M) ≤ 1` AND `exp(−M) ≤ 1`, both
  bounded, fp32 overflow becomes impossible at any score magnitude.
- Fully-masked row: `masked_fill(−inf)` ⇒ `amax = −inf` ⇒ `clamp_min(0) ⇒ 0`,
  then mask-multiply zeroes the row. No NaN, no division by 0.

**Empirical agreement & overflow check** (manual run against a naive
reference at `models/layers.py:74-85`):

| input | naive | stable | ok? |
|---|---|---|---|
| randn × 5.0, causal mask | finite | finite, max-diff 1.13e-6 | ✓ identical up to fp32 precision noise |
| scores=[95.0, 0, 0] | NaN | [≈1.0, 0, 0] | ✓ stable recovers correct (only positive ⇒ all mass) |
| fully-masked row | — | 0 | ✓ no NaN |
| all-neg scores | finite | identical (max-diff 0.0) | ✓ clamp-min(0) is a no-op as designed |

**Tests are green.** `pytest tests/test_softpick.py -v` → 9/9 pass on the
local box (8 prior + new `test_no_nan_under_fp32_exp_overflow` which plants
scores ~mean 100 σ 50 — many > fp32 exp ceiling 88.7 — and asserts the
stable form stays finite with row-sums ≈ 1).

**Spec faithfulness re-checked:**

- *Helper math* (`models/layers.py:74-85`) — `s.fp32`, `M = max over unmasked
  ≥ 0`, `z = exp(s − M) − exp(−M)`, `num = relu(z) · m`, `den = |z| · m`,
  return `num / (sum(den, dim=−1) + ε)` cast back to model dtype. ε=1e-6
  pinned, fp32 cast pinned, mask multiplied into BOTH numerator and
  denominator. Functionally identical to the spec's
  `relu(exp(x) − 1) / Σ|exp(x) − 1|` at every input; differs only in being
  overflow-safe. No mechanism change.
- *Swap site* (`models/layers.py:1642-1650`) — single `if self.use_softpick:`
  in the FIRE manual-attention branch, calls
  `softpick(scores, window.view(1,1,T,T))` reusing the same `window`
  tensor that already produced `masked_fill(−1e9)` (so the softpick mask
  matches the causal/SWA mask exactly). Else branch is the original
  `torch.softmax(scores, dim=-1)` — identity-when-off holds.
- *OR-list defensive fallback* (`models/layers.py:1660`) — `or
  self.use_softpick` present with comment noting the swap site is the FIRE
  branch above; a non-FIRE path with `use_softpick=True` falls back to
  plain softmax (the spec's intended behavior).
- *Flag wiring* end-to-end (verified line-by-line):
  `configs/llm_config.py:190` (LLMConfig) →
  `models/llm.py:228` (LLM.__init__ getattr) →
  `models/llm.py:376` (Block kwarg) →
  `models/layers.py:2070` (Block.__init__ kwarg) →
  `models/layers.py:2127` (Block→MHA pass-through) →
  `models/layers.py:532, 751` (MHA kwarg + stored on self). No constructed
  module on the off-path ⇒ bit-identical baseline.
- *Trt config* (`configs/llm_config.py:918-945`)
  `Tiny1M3MSoftpickOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig)` —
  parent is the FIRE-equipped 009-WIN config; only two overrides
  (`use_fire_pe=True`, `use_softpick=True`). Verified empirically:
  `use_value_embed=True`, `use_q_gain=True`, `use_sliding_window=True`,
  `sliding_window_size=512`, `rope_base=250000` all agree with the ctrl;
  only `use_fire_pe` (False on the bare ctrl class, set at runtime per
  plan) and `use_softpick` differ. MRO confirms parent chain. Exported
  from `configs/__init__.py:14, 111`.
- *LoC budget* — softpick-helper diff against r2: +28 LoC (mostly comments
  explaining the identity); active code remains ~7 LoC inside the helper.
  New regression test +35 LoC. Total softpick active code: well under the
  50-LoC cap and the 200-LoC ceiling.
- *No HP drift in the softpick code path.* LR/schedule/init/seed
  untouched. Only changes are inside the softpick helper and the new
  regression test.
- *Coordination.* The diff includes the parallel agent's FoX move
  (post-softmax multiply → logit-add at `models/layers.py:1621-1633,
  1753-1759`) and 029 V-Norm wiring. The FoX move is at lines 1632 and
  1757 — adjacent to but not inside the softpick swap site at line 1647;
  the softpick helper at line 43 is untouched by parallel work. No
  reverts, no stomps. Untouched: `models/llm.py`, `configs/llm_config.py`
  softpick lines.

**Carry-over from r2 codereview** (still hold, no new findings):
- Identity-when-off holds (off-path on `torch.softmax`, no new ops).
- Step-0 smoke gate passes (test_step0_finite_loss_and_nonzero_qkv_grads).
- Mask interaction correct (test_mask_does_not_pollute_denominator,
  test_masked_mask_zeroes_denominator_term).

**3-round cap context.** Round 3 forces accept or reject. The recode is a
mathematically-exact stability fix with empirical agreement to naive on
safe inputs and correct behavior in the overflow regime that NaN'd r2.
Re-running this trt against the FIRE-equipped ctrl now exercises the
mechanism rather than the bug. Accept.

---

## r2 — 2026-06-10 — verdict: accept

**r1 blocking finding (parent-class HP drift) is fully fixed.** Empirical
verification (instantiated `Tiny1M3MSoftpickOnFireConfig` and
`Tiny1M3MVQGainSWAHighRoPE250KConfig`, walked the MRO and field-by-field
diff):

- `type(trt).__mro__[1].__name__ == "Tiny1M3MVQGainSWAHighRoPE250KConfig"` ✓
- `use_value_embed=True`, `use_q_gain=True`, `use_sliding_window=True`,
  `sliding_window_size=512`, `rope_base=250000` — all agree ctrl↔trt ✓
- `use_fire_pe` is `False` on the ctrl class and `True` on the trt class;
  this is the by-design runtime pattern (`plan.md:185-186` — ctrl gets FIRE
  at runtime, trt has it baked into the class) and matches how 020-FoX
  handles the same offset. Not a drift.
- `use_softpick` is the only lever delta. ✓

**Tests are green.** `python -m pytest tests/test_softpick.py -v` →
8/8 pass on the local box (test_softpick.py::test_no_nan_or_inf_random_input,
test_no_nan_or_inf_under_low_precision_cast,
test_all_true_mask_row_sums_le_one_with_positive_scores,
test_all_nonpos_scores_yield_zero_mass,
test_step0_finite_loss_and_nonzero_qkv_grads,
test_mask_does_not_pollute_denominator,
test_masked_mask_zeroes_denominator_term,
test_use_softpick_false_runs_softmax_unchanged).

**Spec faithfulness re-checked (r1's non-blocking observations still hold):**

- *Helper math* (`models/layers.py:39-60`) — `z = exp(scores.fp32) − 1`,
  `num = relu(z) * m`, `den = |z| * m`, `out = num / (den.sum(−1)+ε)`
  cast back to model dtype. ε=1e-6 default pinned, fp32 cast pinned,
  mask multiplied into BOTH numerator and denominator. Matches
  `idea.md:18-28` and the `idea.md:30-44` mask-interaction fix.
- *Swap site* (`models/layers.py:1590-1593`) — single `if self.use_softpick:`
  branch in the FIRE manual-attention path, calls
  `softpick(scores, window.view(1,1,T,T))` reusing the same `window`
  tensor already used for `masked_fill(−1e9)` (so the mask and the
  softpick mask are exactly the same). The else branch is the original
  `torch.softmax` — identity-when-off holds, no reordering, no new ops.
- *OR-list defensive fallback* (`models/layers.py:1614`) — added
  `or self.use_softpick` with a comment that the swap site is the
  FIRE branch above; a non-FIRE path with `use_softpick=True` falls back
  to plain softmax, which is the spec's intended behavior.
- *Flag wiring* end-to-end: `configs/llm_config.py:190` (LLMConfig) →
  `models/llm.py:228` (getattr) → `models/llm.py:370` (Block kwarg) →
  `models/layers.py:2019` (Block kwarg) → `models/layers.py:2076`
  (pass to MHA) → `models/layers.py:507, 707` (MHA kwarg + stored).
  No new module; off-path is bit-identical.
- *Trt config* (`configs/llm_config.py:855-882`) — dataclass with
  parent = `Tiny1M3MVQGainSWAHighRoPE250KConfig`, only two overrides
  (`use_fire_pe=True`, `use_softpick=True`). Docstring names the
  parent-class choice, the 009 WIN anchor, the lever's category, and
  the step-0 smoke gate.
- *LoC budget* — non-test softpick-only diff is ~30 LoC active code
  (helper ~10, swap site 4, OR-list 1, flag plumbing ~10, trt config
  ~3, comments heavy). Well under the 50-LoC cap and the 200-LoC
  ceiling. Test file is 283 LoC but doesn't count.
- *No silent HP drift in the softpick code path.* No LR / schedule /
  init / seed changed. The only drift was the r1 parent-class
  inheritance, which is now fixed.
- *Coordination* — the diff includes 020/023/024/025 wiring from the
  parallel agent; no reverts or stomps. Softpick-only diff is the
  helper, the swap site, the OR-list entry, the flag plumbing, and the
  new config class. No collisions.

**One-thing-I-would-track-but-not-block:** the spec describes the
softpick normalizer as one-line, but the actual diff is ~30 LoC of
active code plus ~50 LoC of comments / design notes. Comments are
useful (they explain the mask-interaction fix and the FIRE-branch
constraint) and are not on the LoC budget; mentioning it for the
record so a future `simplify` pass can decide whether to compress.

**Round 2, accept allowed.** Frontmatter `round: 2` — the 3-round cap
permits accept/reject/revise, and there's nothing left to revise.

---

## r1 — 2026-06-10 — verdict: revise

### 🔴 BLOCKING — silent ctrl/trt recipe drift (the A/B is malformed as wired)

- **`configs/llm_config.py:840-863`** — `Tiny1M3MSoftpickOnFireConfig`
  extends **`Tiny1M3MConfig`** (vanilla tiny1m3m), but the ctrl per
  `idea.md:86-93` and `plan.md:87-91` is
  **`Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`** (the 009
  WIN signature, val 6.3234 in `closed.md:40`). The trt therefore
  silently drops VQ-gain + SWA(512) + RoPE 250K — four fields of HP
  drift, not a "function swap on top of FIRE". Verified empirically:

  ```
  CTRL  use_value_embed=True   use_q_gain=True   use_sliding_window=True   rope_base=250000
  TRT   use_value_embed=False  use_q_gain=False  use_sliding_window=False  rope_base=10000
  ```

  The plan's Control block (`plan.md:87-91`) is explicit: "**Trt**:
  `Tiny1M3MSoftpickOnFireConfig` — *same recipe as ctrl* +
  `use_softpick=True`." The current trt is not the same recipe. Any
  Δval signal will be dominated by removing V-embed/Q-gain/SWA/high-
  RoPE, not by softpick — the lever being measured is invisible.

  **Fix:** change the dataclass parent from `Tiny1M3MConfig` to
  `Tiny1M3MVQGainSWAHighRoPE250KConfig` and keep only the two
  override fields:

  ```python
  @dataclass
  class Tiny1M3MSoftpickOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
      """...docstring stays..."""
      use_fire_pe: bool = True
      use_softpick: bool = True
  ```

  After the fix, re-verify by diffing the resolved instance:
  `for f in ['use_value_embed','use_q_gain','use_sliding_window','sliding_window_size','rope_base','use_fire_pe','use_softpick']:`
  ctrl and trt must agree on every field except `use_softpick`.

  Note: the analogous `Tiny1M3MFOXOnFireConfig` (020) has the same
  bug and was incorrectly green-lit in 020's r1 codereview; that
  doesn't make it correct here. 022's ctrl is the same FIRE-equipped
  009-WIN config, so the same fix shape applies.

### Non-blocking observations (no findings, recorded for the next pass)

- **Softpick helper math is faithful to the spec.**
  `models/layers.py:36-58` — `z = exp(scores.fp32) − 1`, `num =
  relu(z) * mask`, `den = z.abs() * mask`, return `num / (den.sum(-1)
  + eps)` cast back to model dtype. ε=1e-6 default matches paper.
  fp32 cast for the `exp − 1` op is in place (large positive scores
  overflow in fp16/bf16, as the spec warns). Output shape and dtype
  round-trip cleanly. Matches `idea.md:18-28`.
- **Mask interaction is correctly handled.** Swap site at
  `models/layers.py:1538-1546` passes the same `window` tensor used
  for `masked_fill` into softpick. Inside softpick, `m = mask.to(z.dtype)`
  is multiplied into BOTH numerator (`num = relu(z) * m`) AND
  denominator (`den = z.abs() * m`). Masked positions contribute zero
  to both — the `idea.md:32-45` bug class is closed. Confirmed by
  `test_mask_does_not_pollute_denominator` and
  `test_masked_mask_zeroes_denominator_term`.
- **Identity-when-off holds.** `use_softpick=False` keeps the
  baseline path on `torch.softmax(scores, dim=-1)` with no
  reordering, no new ops, no extra params. The `if self.use_softpick:
  attn_w = softpick(...)` branch is exactly one statement and is
  cleanly gated. No drift to the `else` branch. No softpick params
  are constructed when the flag is off (the helper is a pure
  function, no module).
- **Step-0 smoke gate passes.** Built
  `Tiny1M3MSoftpickOnFireConfig` (0.9491M params — matches the
  tiny1m3m budget, no extra params from softpick), ran one fwd+bwd
  on a batch of token ids: loss is finite, output finite, Q/K/V
  grads on `qkvo_proj.weight[:qkv_size]` are non-zero. The lever is
  alive, not dead-on-arrival. `tests/test_softpick.py::test_step0_finite_loss_and_nonzero_qkv_grads`
  is the persistent guard.
- **OR-list defensive fallback is present and well-commented.**
  `models/layers.py:1567` adds `or self.use_softpick` to the manual-
  branch trigger, with a comment noting the swap site is the FIRE
  branch above; if `use_softpick=True` and `use_fire_pe=False`, the
  manual path is forced and falls through to plain softmax (the
  spec's intended behavior — the defensive entry is belt-and-braces).
- **Tests are green.** `pytest tests/test_softpick.py -v` → 8/8 pass:
  finite on random input, finite under fp16/bf16 cast, all-True mask
  row-sums ≤ 1 with positive scores, all-non-pos scores → zero mass,
  step-0 finite loss + non-zero Q/K/V grads, mask does not pollute
  denominator, direct masked denominator-term check, off-flag path
  differs from on-flag path (wiring is live).
- **No silent ε/dtype drift inside the helper.** `eps=1e-6` is the
  default kwarg; `scores.to(torch.float32)` is explicit. Both pinned
  per `idea.md:23-26`.
- **Flag wiring is end-to-end.** `configs/llm_config.py:186`
  declares `use_softpick: bool = False` on `LLMConfig` (next to
  `use_fox`); `models/llm.py:225-228` reads via `getattr` and passes
  to `TransformerBlock`; `models/layers.py:1969-1971` receives at the
  Block kwarg; `models/layers.py:2022` passes to MHA; MHA stores at
  `self.use_softpick` at `models/layers.py:683-687`. No constructed
  module for the off-path → bit-identical baseline.
- **Coordination with parallel agent's work is clean.** The diff
  includes wiring for 023 (canon-conv), 024 (gated-attn), 025
  (SSMax) alongside 022 — the parallel agent's flags. These coexist
  with the softpick swap site without conflict (SSMax multiplies
  `scores` before softpick, but trt config has `use_ssmax=False`, so
  it doesn't fire; canon-conv and gated-attn are on different code
  paths). Softpick changes are not reverted or stomped.
- **LoC budget respected.** Non-test softpick-only diff is ~30 LoC
  (helper ~10, swap site 4, OR-list 1, flag plumbing ~10, trt config
  ~3). Comments are extensive but the active-code budget is under
  the 50-LoC cap and well under the 200-LoC ceiling. Test file is
  283 LoC but tests don't count against the LoC budget for the
  research lever.
- **No HP smuggling in the softpick code path itself.** No LR,
  schedule, init constant, or seed changed by the softpick diff. The
  drift called out as the blocking finding above is in the *config-
  class inheritance*, not in the softpick op code.
- **Round 1, revise allowed.** Frontmatter `round: 1` — revise is in
  budget; only one finding to apply, ~3-line patch in
  `configs/llm_config.py`.
