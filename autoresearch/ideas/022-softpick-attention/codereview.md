# Code-review log — 022 softpick-attention

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
