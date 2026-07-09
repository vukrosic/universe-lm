## r3 — 2026-06-10 — verdict: accept

r2 blocking finding closed cleanly. Both r2 nits also addressed. All 6 tests
pass. No new findings. Round 3 cap — only accept/reject available; the diff
is correct and faithful, so accept and go to run.

### r2 findings verified closed

- **BLOCKING (closed) — trt parent class fixed.** `configs/llm_config.py:894`
  now reads `class Tiny1M3MFOXOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):`
  (was `Tiny1M3MConfig` at r2's line 850). Class body unchanged
  (`use_fire_pe = True; use_fox = True`). Class was relocated to line 893–919
  (immediately after the parent), matching the
  `Tiny1M3MSoftpickOnFireConfig` / `Tiny1M3MCanonOnFireConfig` /
  `Tiny1M3MUNetSigmoidOnFireConfig` / `Tiny1M3MQKNormOnFireConfig` sibling
  pattern. Docstring (`configs/llm_config.py:898`) calls out the parent so
  the next reviewer can verify by inspection. **Empirically verified**:
  instantiated `Tiny1M3MFOXOnFireConfig()` and `Tiny1M3MVQGainSWAHighRoPE250KConfig()`
  + `use_fire_pe=True` and diffed all dataclass fields — exactly **one**
  flag differs: `use_fox: ctrl=False trt=True`. The four previously-drifted
  levers (`use_value_embed`, `use_q_gain`, `use_sliding_window`, `rope_base`)
  now match. The A/B is the promised 1-lever ablation of FoX on top of
  the 009 WIN signature.

- **Nit (closed) — `test_step0_attention_output_unchanged` parametrized over
  FIRE branch.** `tests/test_fox.py:143` now wraps the body in
  `for use_fire_pe in (False, True):` so both the SDPA-vs-manual path
  (`use_fire_pe=False`) and the FIRE-branch FoX hookup at
  `models/layers.py:1632-1633` are exercised. Both variants assert
  `diff < 1e-2`; both pass.

- **Nit (closed-by-runner) — `evidence.md` staleness.** Flagged but out of
  this gate's scope; next successful run overwrites it.

### What's still correct (no findings, r1/r2 holds verified)

- **Log-add math identity.** `softmax(s + log_D)_i = softmax(s)_i · D_i /
  Σ_j softmax(s)_j · D_j` — equivalent to the paper's multiply-after-softmax
  form, but softmax's max-subtraction absorbs arbitrarily-negative log_D so
  the r1 NaN failure mode (post-softmax row-renorm underflowing to ≈0/≈0)
  is mathematically impossible here.
- **Identity init preserved.** `models/fox.py:65` `FOX_BF_INIT = +10.0`;
  `gate_w` Parameter zero-init at `models/fox.py:107`; `gate_b` Buffer
  set to `FOX_BF_INIT` at `models/fox.py:112`. Math: `logsigmoid(10) ≈
  −4.54e−5`, cumsum over T=2048 → `log_D[0, 2047] ≈ −0.093`, softmax
  drift ≤ ~9% on the worst-case row. `test_identity_init_close_to_ones`
  passes.
- **Off-path bit-identical.** `models/layers.py:744-746` guards
  construction of `self.fox` behind `if self.use_fox:`; the FIRE-branch
  hook at `models/layers.py:1632` and manual-branch hook at
  `models/layers.py:1760` are both `if self.use_fox: scores = scores +
  self.fox(x).to(scores.dtype)`. With `use_fox=False`, no module is
  built, no FLOPs are added, state_dict is fox-free.
- **Flag forces manual attention path.** `models/layers.py:1659`
  (`or self.use_fox`) OR's `use_fox` into the manual-branch condition so
  the pre-softmax additive bias actually fires (SDPA's flash kernel
  doesn't expose pre-softmax logits).
- **Trained-gate regression covered.** `test_trained_gate_does_not_blow_up`
  drives `b_f=-3` at T=2048 (the exact r1 NaN scenario) and verifies
  `log_D` stays finite and `softmax(random + log_D)` rows sum to 1.
- **All 6 tests pass.** `pytest tests/test_fox.py -v` → `6 passed, 3
  warnings in 5.96s`.
- **No silent HP drift on LR/schedule/init.** Only `use_fox` toggles between
  ctrl and trt; no LR, schedule, init, or seed constants touched in
  `configs/llm_config.py` for this idea.
- **Pass/fail bar unchanged.** `idea.md:88-96` Win Δ ≤ −0.02, Null
  |Δ| < 0.02, Fail Δ > +0.01. Seed 42 only. Tier tiny1m3m. Plan ↔ idea
  consistency holds.
- **LoC budget respected.** `models/fox.py` 151 LoC incl. docstring,
  ~85 LoC code. Layers wiring ~10 LoC across two branches + the kwarg.
  Well under the 200 LoC ceiling.
- **Coordination clean.** `git diff` shows FoX edits at
  `models/layers.py:744-746` (init), `:1621-1633` (FIRE), `:1659`
  (manual-branch OR), `:1753-1761` (manual) — none overlap with the
  parallel-AI's V-Norm / Moonlight×QK-Norm / U-Net-sigmoid / QK-Norm-on-FIRE
  edits at the other line ranges. No rebase, no stomp, no push.

### Routing

Accept → `needs-run`. Queue row at `autoresearch/queue.md:150` shows the
prior NaN failure; the runner will repopulate on the next pass after the
log-add fix + parent-class fix land in the rerun.

---

## r2 — 2026-06-10 — verdict: revise

The NaN fix is mathematically sound and all 6 tests pass, but a previously-
missed silent HP drift in the trt dataclass makes the A/B 4-lever-deep, not
the 1-lever-deep comparison the plan promises. Single blocking finding +
two nits to clean up while you're in there.

- **BLOCKING — Trt config parent is wrong → silent 4-lever HP drift.**
  `configs/llm_config.py:850` declares
  `class Tiny1M3MFOXOnFireConfig(Tiny1M3MConfig):` but `plan.md:113` (and
  `idea.md:81-86`) explicitly promise the trt is "same recipe as ctrl +
  `use_fox=True`" where ctrl is `Tiny1M3MVQGainSWAHighRoPE250KConfig +
  use_fire_pe=True`. The trt instance differs from ctrl on **four** levers
  besides `use_fox`, verified by instantiating both:

  | flag                 | ctrl | trt   |
  |----------------------|------|-------|
  | `use_value_embed`    | True | False |
  | `use_q_gain`         | True | False |
  | `use_sliding_window` | True | False |
  | `rope_base`          | 250000 | 10000 |
  | `use_fox`            | False | True |

  This is exactly the "silent HP drift smuggled in alongside a mechanism
  change" that `prompts/code-reviewer.md §2` calls out as a finding. The
  A/B partitions "FIRE + FoX vs VQGain+SWA+RoPE250K+FIRE", not "FIRE+FoX
  vs FIRE". Even after the r2 NaN fix completes a clean run, the trt vs
  ctrl Δ confounds FoX with dropping V-embed + Q-gain + SWA(512) + RoPE
  base 250k. The 009 WIN signature (val 6.3234, `closed.md:40`) is on
  VQGain+SWA+RoPE250K+FIRE; comparing a FIRE-only-but-no-VQGain-SWA-RoPE
  trt against it is not a clean ablation of FoX. R1 codereview accepted
  this without raising the finding (the r1 review literally noted
  `use_fire_pe=True, use_fox=True` on trt + the VQGain ctrl side-by-side
  and called it "matches the plan" — that was wrong, the asymmetry was
  there in r1 already and was never settled). **Fix (one line):** change
  `configs/llm_config.py:850` from
  `class Tiny1M3MFOXOnFireConfig(Tiny1M3MConfig):` to
  `class Tiny1M3MFOXOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):` —
  the body (`use_fire_pe = True; use_fox = True`) is already correct;
  inheriting from the VQGain+SWA+RoPE250K parent will pick up
  `use_value_embed=True, use_q_gain=True, use_sliding_window=True,
  sliding_window_size=512, rope_base=250000` automatically. Verify
  post-fix by instantiating `Tiny1M3MFOXOnFireConfig()` and confirming
  all five flags above match the ctrl row. (Note: 021-value-residual,
  024-gated-attention, 027-moonlight-x-qknorm sibling configs have the
  same parent bug — not scoped to this idea, but flag it to the queue.)

- **Nit — `test_step0_attention_output_unchanged` doesn't exercise the
  trt path.** `tests/test_fox.py:135-141` builds both MHAs with
  `use_fire_pe=False`, so `mha_no` goes through SDPA (fast path) and
  `mha_fox` goes through the manual branch (FoX forces it via
  `models/layers.py:1659`). The test passes — but it's measuring "SDPA
  vs manual + FoX log_D", not "manual vs manual + FoX log_D" or
  "FIRE-branch vs FIRE-branch + FoX log_D". The ~1e-2 ceiling absorbs
  both the FoX drift and the SDPA-vs-manual numerical drift. The trt
  config actually exercises the FIRE branch (`use_fire_pe=True`
  → `models/layers.py:1574-1652`). **Fix (low priority):** add a second
  variant with `use_fire_pe=True` on both sides, or set
  `use_fire_pe=False, use_fox=False` on `mha_no` AND force the manual
  branch by adding `use_cope=True` then comparing (clunky). Cleanest:
  parametrize the test over `(use_fire_pe ∈ {False, True})` so the FIRE
  branch's FoX hookup at `models/layers.py:1632-1633` is also covered.
  Not strictly blocking — the math identity
  `softmax(s + log_D) = softmax(s) ⊙ D / row_sum` holds in both
  branches and the FoX module itself is branch-agnostic, but the
  current test gives weaker coverage than its name implies.

- **Nit — `evidence.md` is stale (records the failed r1 run).** Not your
  job to rewrite, but call it out for the runner: the existing
  `evidence.md` still says "Verdict: FAIL — NaN, needs-recode" and the
  WIN/NULL bookkeeping. On the next successful run the runner will
  overwrite it.

### What looked correct (no findings)

- **Log-add identity holds.** Pen-and-paper:
  `softmax(s + log_D)_i = exp(s_i)D_i / Σ_j exp(s_j)D_j
  = softmax(s)_i · D_i / Σ_j softmax(s)_j · D_j`
  — exactly the multiply-after-softmax + row-renorm form in the paper
  sketch. Softmax's internal max-subtraction absorbs arbitrarily
  negative `log_D` cumsums (the exact failure mode of r1).
- **Identity init preserved.** `gate_w=0` (Parameter, zero-init) +
  `gate_b=+10` (buffer, `FOX_BF_INIT`). `logsigmoid(10) ≈ -4.54e-5`,
  cumsum over T=2048 → `log_D[0, 2047] ≈ -0.093`, softmax drift ≤ ~9%
  worst-case row. Verified numerically in
  `test_identity_init_close_to_ones` (passes, asserts
  `exp(log_D) ∈ [D_min, 1+1e-6]` and `|diag - 1| < 1e-3`).
- **Off-path bit-identical.** `use_fox=False` ⇒ `self.fox` is not
  constructed (`models/layers.py:744-746`), the `if self.use_fox:` guard
  at `models/layers.py:1632` (FIRE) and `:1760` (manual) is skipped, no
  extra params, no extra FLOPs. State dict is fox-free.
- **Upper-tri masking hygiene.** `models/fox.py:139-141` masks the
  upper triangle of `log_D` to 0 — the caller's `-1e9` causal mask
  dominates regardless, but zeroing keeps the tensor safe to inspect.
- **Trained-gate regression covered.** `test_trained_gate_does_not_blow_up`
  drives `b_f=-3` at T=2048 (the exact NaN scenario from r1 evidence.md)
  and verifies `log_D` stays finite + `softmax(random + log_D)` rows
  sum to 1. This is the test r1 was missing.
- **All 6 tests pass.** `pytest tests/test_fox.py -v` →
  `6 passed, 3 warnings in 5.16s` locally.
- **No silent HP drift on the LR/schedule/init axes.** Only `use_fox`
  + (the existing) `use_fire_pe` flags are touched in the trt dataclass.
  Per-layer FoX module adds ~2k params/layer; cost discussion in
  `plan.md:124-136` checks out.
- **LoC budget respected.** `models/fox.py` is 150 LoC incl. docstring
  (~85 LoC code). Wiring in `models/layers.py` is ~10 LoC across the
  two branches + the kwarg. Under the 200 LoC ceiling.
- **Coordination clean.** `git diff models/layers.py configs/llm_config.py`
  shows unstaged V-Norm + Moonlight×QK-Norm + U-Net-sigmoid + QK-Norm-on-FIRE
  edits from the parallel-AI at lines 552-577, 711-720, 744-758,
  1006-1052, 1128-1163, 1990-1995, 2144-2147. None of these overlap with
  the FoX edits at 744-746 (init), 1621-1633 (FIRE branch hook),
  1659 (manual-branch OR), 1750-1761 (manual-branch hook). No
  rebase/stomp, no push.

## r1 — 2026-06-10 — verdict: accept

- **Mechanism is implemented faithfully.** `models/fox.py` does exactly what `idea.md:23-28` prescribes: per-head gate `z = einsum("btd,hd->bth", x, gate_w) + gate_b` → `log_f = logsigmoid(z)` (≤ 0) → `cumsum(log_f, dim=1)` → pad a zero along T → `D = exp(right − left).permute(0,3,1,2)` → explicit `causal` mask (line 129) → output `[B,H,T,T]`. The cumsum / padded-cumsum / slice trick correctly produces `D[i,j] = exp(cum[i] − cum[j-1])` for `j≥1` and `D[i,0] = exp(cum[i])`, matching the paper's cumulative-product definition. `FOX_BF_INIT = +10.0` is the r2-corrected init (not the r1 wrong `+5`).
- **Identity-init math checks numerically.** Smoke-tested at T=2048 with the real init: `f = sigmoid(10) ≈ 0.999955`, `log f ≈ −4.54e-5`, expected `D[0,0,2047,0] = exp(2047 · log f) ≈ 0.9113`, actual measured `0.9112` — within fp32 rounding. The whole lower-tri is in `[D_min ≈ 0.911, 1.0]`; the upper triangle is exactly 0 (sum = 0.000e+00). The corrected `b_f=+10` (vs the r1 wrong `+5` that would have given `D[0,2047] ≈ 1e-6`) is preserved end-to-end — pinned in `models/fox.py:FOX_BF_INIT` and called out in the file docstring.
- **Off-path is bit-identical.** `use_fox=False` → `self.fox` is not constructed (`models/layers.py:606`), no `fox.*` entries in `state_dict()`, no extra params allocated, no extra FLOPs. Verified empirically: `hasattr(mha_no, 'fox') == False`; state_dict is fox-free. The branch wiring is a single `if self.use_fox: d = self.fox(x); attn_w = attn_w * d; attn_w = attn_w / attn_w.sum(...).clamp_min(1e-9)` at lines 1429-1432 (FIRE branch) and 1538-1541 (manual branch) — the `clamp_min(1e-9)` correctly handles fully-masked rows (smoke-tested: 0-row gives 0, no NaN).
- **Step-0 MHA drift is well within tolerance.** Re-ran the plan's test (e) check: `use_fox=False` vs `use_fox=True` (with shared Q/K/V via state_dict copy) gives `max|y_no - y_fox| = 8.9e-6`. The plan's tolerance is `1e-2` (the r2 nit tightened from the r1 `1e-5` after the reviewer flagged it as too tight); 8.9e-6 is below both numbers. The test `test_step0_attention_output_unchanged` in `tests/test_fox.py:111-156` pins this with `assert diff < 1e-2` and currently passes (`pytest tests/test_fox.py` → 5 passed).
- **All 5 tests pass.** `pytest tests/test_fox.py -v` → 5/5 green: `test_no_nan_or_inf` (finite + correct shape), `test_causal_lower_triangular` (D upper-tri = 0 within 1e-6), `test_identity_init_close_to_ones` (D ∈ [D_min, 1+1e-6], diagonal within 1e-3 of 1), `test_wiring_live_with_Wf_perturbation` (head 0 changes by >1e-4 when W_f is perturbed, heads 1..5 unchanged within 1e-5 — the projection is wired), `test_step0_attention_output_unchanged` (drift < 1e-2).
- **Wiring is complete end-to-end.** `configs/llm_config.py:179` declares `use_fox: bool = False`; `models/llm.py:224,338` reads via `getattr(config, "use_fox", False)` and passes to `TransformerBlock`; `models/layers.py:1809,1847` receives and passes to MHA; `models/layers.py:447,605,606` stores and conditionally builds `FoX(d_model, n_heads)`. No silent HP drift — only the new `use_fox: bool = False` field is added; no LR/schedule/init/seed constants are touched.
- **Flag forces the manual attention path.** `models/layers.py:1441` OR's `self.use_fox` into the manual-branch condition (post-softmax multiply can't go through SDPA's flash kernel). Consistent with `plan.md:46` and the FIRE-branch's existing path.
- **Trt config matches the plan.** `Tiny1M3MFOXOnFireConfig` (`configs/llm_config.py:713-735`) sets `use_fire_pe: bool = True, use_fox: bool = True`. Ctrl is `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True` — the 009 WIN signature (val 6.3234 vs ctrls 6.3875/6.4050, cited in `closed.md:44`, called out in the dataclass docstring). The A/B partitions the orthogonal-axis question (FIRE additive on logits + FoX multiplicative on probs) rather than "does FIRE win?". Pass/fail bar matches `idea.md:88-96`: Win `trt < ctrl − 0.02`, Null `|trt − ctrl| < 0.02`, Fail `trt > ctrl + 0.01`. Seed 42 only (no sweep).
- **LoC budget respected.** `models/fox.py` is 131 LoC including the docstring and comment lines (84 lines of pure non-blank non-comment code). The plan's ~47 LoC code-only estimate is accurate; the file's full 131 LoC is still under the 200 LoC ceiling. Wiring in `models/layers.py` adds ~15 LoC for the kwarg, conditional build, and two branch handlers; `configs/llm_config.py` adds the field and the `Tiny1M3MFOXOnFireConfig` dataclass. Tests file is 156 LoC.
- **Coordination clean.** `git diff models/layers.py configs/llm_config.py models/llm.py` is empty (the 020 changes are already committed in 128502a alongside 013/015/016/017/011, but the 020 surface in those files — `use_fox` field, `use_fox` kwarg in MHA/TB, the FIRE+manual branches' post-softmax wiring — is what the plan promised). No unstaged edits to stomp, no parallel-AI rebase to reconcile, no push.
- **No findings.** Code is correct, faithful to the spec, identity-safe, ready to run.
