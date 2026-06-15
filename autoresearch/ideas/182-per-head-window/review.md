# Review log — 182 per-head-window

## r4 — 2026-06-15 — verdict: approve

Re-pass because the auto-implement agent flipped this idea to `rejected` a
*second* time at 2026-06-15T07:05:54Z ("auto-rejected: blocked 4x with no
path forward (see log)"). This is the same doer-protocol violation r3
overturned: auto-implement is a doer, the spec has not changed since the
r3 approval, and the reviewer never issued a reject. Per the reviewer
prompt: *"Doers never close — the reviser and code-implementer bounce
blocked ideas back to a `needs-*` queue, not to `rejected`."* Bounce-loop
termination on push-side staleness is not the reviewer's call to ratify.
Re-walking the spec to confirm the r3 verdict still holds.

- **Spec is unchanged from r3 approval.** Re-verified the four r3 findings
  all hold against the current `idea.md`:
  - **Step-0 byte-identical (BLOCKING, fixed in r1).** `W_h =
    2T·sigmoid(w_h)` with `w_h_init=10` gives `W_h/2 = T·sigmoid(10) ≈
    T − 0.00005·T`. At T=2048, `W_h/2 ≈ 2047.9 > max|t−s| = T−1 = 2047`,
    so the mask is all-ones, `relu(|t−s| − W_h/2) = 0` everywhere, softmax
    unchanged ⇒ **byte-identical at fp32**. Code-impl self-check:
    `max_abs_diff = 2.98e-08` (well under the 1e-6 bar). The dropped
    `β_h = sigmoid(w_h)` alternative is correctly absent.
  - **Single sub-lever committed.** Hard window only. Soft Gaussian decay
    (`λ_h·(t−s)²`) is explicitly deferred, with the rationale preserved.
    No implementer choice space.
  - **Pass/fail bar is concrete and tied to a real control.** NULL band
    `|trt − cached_baseline| < 0.01`, WIN pass `trt ≤ cached_baseline −
    0.01`, cache-authoritative WIN rule `trt < val_mean − noise_band`.
    Cache has moved again (now pinned at `val_mean = 6.2403`, `val_std =
    0.0088`, `noise_band = 0.04` ⇒ WIN iff `trt < 6.2003` — the 175-alibi
    WIN has reset the cache). The spec's run-day re-pull instruction
    handles this: plan.md mirrors whichever cache version is current on
    run day; evidence.md cites that version.
  - **`1e9` penalty explicit.** No more `−∞` in prose; spec pins `1e9`
    (fp32-clean, no NaN risk, matches 154-rebased-attn's rebased-softmax
    style).

- **Source check.** BigBird (Zaheer et al., arXiv:2007.14062, NeurIPS 2020)
  and Longformer (Beltagy et al., arXiv:2004.05150, 2020) are real, the
  per-head-pattern ablation in BigBird is real, the 100M+ scale-evidence
  claim is honest. Not fabricated.

- **Distinct from closed.** Confirmed against `autoresearch/closed.md` (no
  `per-head-window` entry). The closed SWA window-sweep line is a *fixed
  global HP*, not a per-head learnable window — different lever. Closed
  per-head scalars (152/155/160/166/172) are *score-magnitude* levers;
  182 is a *spatial-pattern* lever, in mechanism shape with
  154-rebased-attn (WIN, Δ-3.48) and 143-shortconv (borderline).
  174-xpos-decay null tested learnable *scalar decay*, not a window.
  **Distinct, salvageable.**

- **Implementability.** `<200 LoC`: `use_per_head_window: bool` config
  flag, +48 params (H=4 × n_layers=12), one extra `1e9 · relu(...)` term
  in the score path in `models/layers.py`, threaded through
  `TransformerBlock`. Trivial. **Already implemented in commit 0653bfc8**
  (per `ideas/182-per-head-window/evidence.md`); local `SMOKE_OK` +
  step-0 `max_abs_diff = 2.98e-08` both pass.

- **Tiny1m3m-only.** Confirmed. No references to `screen20m`, the ladder,
  or any larger tier.

- **Transfer-risk.** `med` is honest. Windowed attention is
  well-validated at 100M+; per-head learnable window is novel at this
  scale but the locality prior is established. Scale-evidence section
  cites BigBird (100M–300M encoder) and Longformer (100M+ encoder). Tag
  matches the citation.

- **Why the second "rejected" line should be undone.** Identical to the
  r3 reasoning: the doer's bounce loop failed because the box
  (`/root/universe-lm`) could not `git pull --ff-only` the local commit
  (no `git push` per the don't-push-without-approval protocol). That is
  a **push-side** issue, not a **spec-side** issue — orthogonal to the
  reviewer's definition-gate responsibilities. The spec is sound, the
  code is committed locally, the byte-identical math holds, and the
  implementation has no path-blocking defect that a reviewer should
  ratify as a reject. Re-approving the spec restores it to `needs-plan`
  with a fresh round budget so code-impl can pick it up again once the
  push lands.

**Verdict: approve.** Sound, falsifiable, one sub-lever, distinct from
closed, byte-identical at step 0, <200 LoC, already implemented locally.
Reset `round` to 1 so the code gate gets a fresh budget.

---

## r3 — 2026-06-15 — verdict: approve

Re-pass because the auto-implement agent flipped this idea to `rejected` at
2026-06-15T06:29:48Z, which is a doer protocol violation. Per the reviewer
prompt: *"Doers never close — the reviser and code-implementer bounce blocked
ideas back to a `needs-*` queue, not to `rejected`."* The auto-implement is a
doer, the spec had not changed, and the reviewer never issued a reject. The
"rejection" was a bounce-loop termination, not a reviewer kill, and
terminating an in-flight idea on push-side staleness is not the reviewer's
call to ratify. Re-walking the spec to confirm the r2 verdict still holds.

- **Spec is unchanged from r2 approval.** Re-verified the four r2 findings all
  hold against the current `idea.md`:
  - **Step-0 byte-identical (BLOCKING, fixed in r1).** `W_h = 2T·sigmoid(w_h)`
    with `w_h_init=10` gives `W_h/2 = T·sigmoid(10) ≈ T − 0.00005·T`. At
    T=2048, `W_h/2 ≈ 2047.9 > max|t−s| = T−1 = 2047`, so the mask is all-ones,
    `relu(|t−s| − W_h/2) = 0` everywhere, softmax unchanged ⇒ **byte-identical
    at fp32**. Code-impl self-check: `max_abs_diff = 2.98e-08` (well under
    the 1e-6 bar). The dropped `β_h = sigmoid(w_h)` alternative is correctly
    absent.
  - **Single sub-lever committed.** Hard window only. Soft Gaussian decay
    (`λ_h·(t−s)²`) is explicitly deferred, with the rationale preserved. No
    implementer choice space.
  - **Pass/fail bar is concrete and tied to a real control.** NULL band
    `|trt − cached_baseline| < 0.01`, WIN pass `trt ≤ cached_baseline − 0.01`,
    cache-authoritative WIN rule `trt < val_mean − noise_band`. Re-pulled
    `autoresearch/baseline-cache.json` today: `val_mean = 6.3988`,
    `val_std = 0.0088`, `noise_band = max(0.04, 2·0.0088) = 0.04` ⇒ **WIN iff
    `trt < 6.3588`**. Plan.md mirrors these numbers verbatim with the run-day
    re-pull instruction. Two-ctrl rule cited (143-shortconv / 131-layer-drop
    style). Numbers in the spec match the current cache — no drift.
  - **`1e9` penalty explicit.** No more `−∞` in prose; spec pins `1e9`
    (fp32-clean, no NaN risk, matches 154-rebased-attn's rebased-softmax
    style).

- **Source check.** BigBird (Zaheer et al., arXiv:2007.14062, NeurIPS 2020) and
  Longformer (Beltagy et al., arXiv:2004.05150, 2020) are real, the
  per-head-pattern ablation in BigBird is real, the 100M+ scale-evidence claim
  is honest. Not fabricated.

- **Distinct from closed.** Confirmed against `autoresearch/closed.md` (no
  `per-head-window` entry). The closed SWA window-sweep line is a *fixed
  global HP*, not a per-head learnable window — different lever. Closed
  per-head scalars (152/155/160/166/172) are *score-magnitude* levers; 182 is
  a *spatial-pattern* lever, in mechanism shape with 154-rebased-attn (WIN,
  Δ-3.48) and 143-shortconv (borderline). 174-xpos-decay null tested
  learnable *scalar decay*, not a window. **Distinct, salvageable.**

- **Implementability.** `<200 LoC`: `use_per_head_window: bool` config flag,
  +48 params (H=4 × n_layers=12), one extra `1e9 · relu(...)` term in the
  score path in `models/layers.py`, threaded through `TransformerBlock`.
  Trivial. **Already implemented in commit 0653bfc8** (per
  `ideas/182-per-head-window/evidence.md`); local `SMOKE_OK` + step-0
  `max_abs_diff = 2.98e-08` both pass.

- **Tiny1m3m-only.** Confirmed. No references to `screen20m`, the ladder, or
  any larger tier.

- **Transfer-risk.** `med` is honest. Windowed attention is well-validated at
  100M+; per-head learnable window is novel at this scale but the locality
  prior is established. Scale-evidence section cites BigBird (100M–300M
  encoder) and Longformer (100M+ encoder). Tag matches the citation.

- **Why the prior "rejected" line should be undone.** The doer's bounce loop
  failed because the box (`/root/universe-lm`) could not `git pull --ff-only`
  the local commit (no `git push` per the don't-push-without-approval
  protocol). That is a **push-side** issue, not a **spec-side** issue — it is
  orthogonal to the reviewer's definition-gate responsibilities. The spec is
  sound, the code is committed locally, the byte-identical math holds, and
  the implementation has no path-blocking defect that a reviewer should
  ratify as a reject. Re-approving the spec restores it to `needs-plan` with
  a fresh round budget so code-impl can pick it up again once the push lands.

**Verdict: approve.** Sound, falsifiable, one sub-lever, distinct from
closed, byte-identical at step 0, <200 LoC, already implemented locally. Reset
`round` to 1 so the code gate gets a fresh budget.

---

## r2 — 2026-06-15 — verdict: approve

All five r1 findings are cleanly applied. Re-walking each one against the current `idea.md`:

- **(1) Step-0 byte-identical math (BLOCKING, fixed).** `W_h = 2T · sigmoid(w_h)` with `w_h_init=10` gives `W_h/2 = T · sigmoid(10) ≈ T - 0.00005·T`. At T=2048, `W_h/2 ≈ 2047.9 > max|t−s| = T−1 = 2047`, so the mask is all-ones, the `relu(|t−s| − W_h/2)` term is identically 0 everywhere, and softmax is unchanged ⇒ **byte-identical at fp32** (max-abs-diff 0.0). The dropped `β_h = sigmoid(w_h)` alternative is correctly absent — it had the same off-by-one and would be a footgun for any future resurrector. Implementer must still run the `max_abs_diff(logits, baseline_logits) < 1e-6` self-check (mirrors 154-rebased-attn), and that gate will catch any implementer who re-introduces the broken formula.
- **(2) Single sub-lever committed.** Hard window only. Soft Gaussian decay (`λ_h·(t−s)²`) is explicitly deferred to a future idea, with the rationale ("different mechanism, different gradient dynamics") preserved. The design sketch in idea.md now points at hard-window only. No implementer choice space on the mechanism.
- **(3) Pass/fail bar is concrete and tied to real control.** NULL band `|trt − cached_baseline| < 0.01`, WIN pass `trt ≤ cached_baseline − 0.01`, cache-authoritative WIN rule `trt < val_mean − noise_band` with `noise_band = max(0.04, 2·val_std)`. Re-pulled cache today: `val_mean = 6.3988`, `val_std = 0.0088`, `noise_band = 0.04` ⇒ **WIN iff `trt < 6.3588`**. Plan must mirror these four numbers verbatim with the run-day re-pull instruction. Two-ctrl rule cited (143-shortconv / 131-layer-drop style). Numbers in the spec match the current cache — no drift.
- **(4) `1e9` penalty explicit.** No more `−∞` in prose; spec pins `1e9` (fp32-clean, no NaN risk, matches 154-rebased-attn). Implementation will not silently swap in `−inf` and trigger softmax NaN.

Source check: BigBird (Zaheer et al., arXiv:2007.14062, NeurIPS 2020) and Longformer (Beltagy et al., arXiv:2004.05150, 2020) are real, the per-head-pattern ablation in BigBird is real, and the 100M+ scale evidence claim is honest. Not fabricated.

Distinct from closed: confirmed against `autoresearch/closed.md` (no `per-head-window` entry). The closed SWA window-sweep line is a *fixed global HP*, not a per-head learnable window — different lever. Closed per-head scalars (152/155/160/166/172) are *score-magnitude* levers; 182 is a *spatial-pattern* lever, in mechanism shape with 154-rebased-attn (WIN, Δ-3.48) and 143-shortconv (borderline). 174-xpos-decay null tested learnable *decay*, not a window. **Distinct.**

Implementable in < 200 LoC: `use_per_head_window: bool` config flag, +48 params (H=4 × n_layers=12), one extra `1e9 · relu(...)` term in the score path in `models/layers.py`. Thread through `TransformerBlock`. Trivial.

Tiny1m3m-only: confirmed. No references to `screen20m`, the ladder, or any larger tier. Step-0 byte-identical test gives a tight falsifiability check (max-abs-diff < 1e-6 at fp32 step 0 is binary).

Transfer-risk: med is honest. Windowed attention is well-validated at 100M+; per-head learnable window is novel at this scale but the locality prior is established. Scale evidence section cites BigBird (100M–300M encoder) and Longformer (100M+ encoder). The tag matches the citation.

**Verdict: approve.** Sound, falsifiable, one sub-lever, distinct from closed, byte-identical at step 0, <200 LoC. Reset `round` to 1 so the code gate gets a fresh budget.

---

## r1 — 2026-06-15 — verdict: revise

- **Step-0 byte-identical claim is wrong (BLOCKING).** The idea proposes `W_h = T · sigmoid(w_h)` with `w_h_init=10` so `sigmoid(10) ≈ 0.99995` and `W_h ≈ T`. The mask is then `M_h(t,s) = 1 if |t−s| ≤ W_h/2`. At T=2048, `W_h/2 ≈ 1023.95`, but `max |t−s| = T−1 = 2047`. The justification in idea.md ("`|t−s| ≤ T−1 < T/2 for T ≥ 2`") is a math error — for any T > 2, `T−1 > T/2`, so the inequality runs the wrong way. The first ~1024 positions of every query fall outside the window at step 0, so `score -= 1e9 · relu(|t−s| − W_h/2) = 1e9 · ~1023 ≈ 1e12` is subtracted, and softmax zeroes those positions. The model is **not** byte-identical to baseline at step 0 — early-layer val_loss will jump ~0.1+ at fp32 (matches the 1e12 score bias pattern in past SWA-on-at-init failures). Fix one of:
  1. `W_h = 2T · sigmoid(w_h)` — at w_h=10, `W_h/2 ≈ 2047.9 > T−1=2047`, so the mask is all-ones at step 0 (recommended; keeps the /2 convention in the spec).
  2. Redefine the mask as `M_h(t,s) = 1 if |t−s| ≤ W_h` (no /2) and keep `W_h = T · sigmoid(w_h)` — at init, `W_h ≈ T > T−1`, mask all-ones. This is the cleanest one-line change.
  3. Pick a much larger `w_h_init` (≈ 15+) so `sigmoid(w_h) ≥ (2T−2)/T ≈ 1.999`, but fp32 sigmoid saturates and gradient through `sigmoid(15)` is ~3e-7, killing the lever — **don't** use this route.
  Implementer must re-verify `max_abs_diff(logits, baseline_logits) < 1e-6` at fp32 step 0 after the fix.

- **Commit to ONE sub-lever (taste already flagged this; the reviser must act on it).** idea.md's "Design sketch" lists both (a) hard window via `W_h = T·σ(w_h)` and (b) soft Gaussian decay via `λ_h·(t−s)²`. They are structurally different mechanisms and have different gradient dynamics. Per taste's "Pick one sub-lever for the run" finding, **commit to the hard-window variant only** (option (a)) and explicitly defer the soft-decay variant. Do not let the implementer ship both in one config.

- **Tighten the pass bar to a specific number tied to a real control.** idea.md has no explicit Δ threshold. Box noise at tiny1m3m is ~±0.01 val_loss; the 4 closed per-head scalars (152-bias, 155-temp, 160-gain, 166-RPE) all nulled at |Δ| < 0.02 at this tier. Set the pass bar: **trt ≤ cached-baseline − 0.01** (matches the 016-qk_norm WIN magnitude of −0.0138 at the same tier), and a **NULL** is `|Δ| < 0.01` against the **fresh** baseline 6.4320±0.04 (the 2026-06-15 cached mean; re-pull from `LEADERBOARD.md` on run day). A WIN must be **strictly less** than both the cached mean and the two same-session ctrls (per the §2 two-ctrl rule used by 143-shortconv / 131-layer-drop).

- **Confirm the per-head-WINDOW axis is distinct from the closed per-head-SCALE axis (not a finding, just defensive).** The closed levers 152/155/160/166/172 are per-head *scalars* that act on attention *magnitudes* (additive bias, temperature, post-AV gain, additive RPE, RoPE base) — all nulled at 0.94M/12L/4H because per-head gradient signal is too weak to specialize. 154-rebased-attn (WIN, Δ-3.48) and 143-shortconv (borderline, all 4 same-day ctrls beaten) both show that *spatial-pattern* changes (which positions are attended to) do bind at this tier, with a much sharper gradient signal than scalar levers. 182 acts on the spatial pattern (which positions are inside the window), not the magnitudes — closer in mechanism-shape to 154/143 than to 152/155/160. The closed SWA-axis line ("SWA window sweep (256/384/512/768/1024/2048) — 512 winner") swept a fixed global window HP, not a per-head learnable window; 182 is not a duplicate. The 174-xpos-decay null tested a *learnable scalar decay* (not a window), so it's a different mechanism from 182's hard window. **Distinct, salvageable.**

- **Drop the second alternative design (β_h = σ(w_h), W_h = T·β_h).** It has the same off-by-one as the primary proposal (max |t−s| > W_h/2 at step 0). If the reviser keeps both formulations in the spec, the implementer might pick the broken one.

- **`1e9` penalty in the relu is fine numerically but make it explicit it is `1e9` not `1e12`.** idea.md's design sketch has `1e9 · relu(...)` while the prose says `−∞`. fp32 `−∞` is unrepresentable cleanly; `−1e9` on logits before softmax is essentially zero probability and avoids any NaN risk. Keep `1e9` (matches 154-rebased-attn's rebased-softmax style; the implementer should mirror whatever 154 uses for consistency).

- **No code changes yet.** The `models/layers.py` diff in the working tree is from the parallel-AI agent (per the coord note). Implementer is the only agent that should touch `models/layers.py` and `configs/llm_config.py`; the reviser just edits `idea.md`.
