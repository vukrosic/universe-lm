# Review log — 025 scalable-softmax

## r1 — 2026-06-10 — verdict: revise

- **Source confirmed.** arXiv:2501.19399 — Ken M. Nakanishi, "Scalable-Softmax
  Is Superior for Attention", submitted 2025-01-31. Title and author verified
  via the abs page. Real, current, single-author preprint. Not fabricated.

- **Mechanism is a mechanism, not an HP.** SSMax multiplies the attention
  scores by `s · log(n)` pre-softmax, where `n` is the per-query causal key
  count. This is a structural change to the attention line — it lives in the
  same site as the existing logit-side tweaks (CoPE, qk-norm, alibi, talking
  heads, antisym-QK, decoupled content/pos) inside the manual-attention branch
  of `MHA.forward`. `s` is a single learnable scalar per head, init 1.0 (the
  paper's natural starting point). No schedule, no auxiliary loss, no new
  trainer plumbing beyond a flag.

- **Duplicate check passes cleanly.** Walked `closed.md` and the active queue
  axis-by-axis vs. the orthogonal axes the taste-reviewer already enumerated:
  - `logit softcap` (closed) — *clamps* with tanh. Different operator.
  - `020-FoX` — *content* decay on probabilities, post-softmax. Different site.
  - `016-qk-norm` (WIN Δ-0.014) — bounds norms pre-softmax. 025 scales logits
    by *length* pre-softmax. Same forward site, different operator; both can
    coexist but they are not the same lever.
  - `022-softpick` — replaces the softmax distribution shape. Different
    family.
  - `024-gated-attention` — sigmoid output gate on `attn @ V`. Different
    family.
  - `009-FIRE` — additive position bias on scores. 025 is multiplicative
    *length* scaling on scores. Different operator at the same site.
  025 is its own lever on a length-temperature axis that is not in the
  closed list.

- **Implementability is fine.** The MHA manual path already has a stable
  logit-pipeline before `torch.softmax(scores, dim=-1)` (the same line that
  CoPE, FoX, qk-norm, and the various Q-tweaks modify). SSMax slots in as
  one more `scores = scores * (s_h.view(1,H,1,1) * log_n.view(1,1,T,1))`
  pre-softmax multiply plus an `nn.Parameter(torch.ones(n_heads))` init
  block. ~15-20 LoC, no SDPA rewrite (manual path already required by the
  other score-side tweaks). Drop-in, identity-ish on the *parameter*
  (s_h=1.0) even if the forward isn't bit-identical to baseline — the
  paper's mechanism *is* the log-scaling, so this is justified, not a
  step-0 red flag. Co-residency with FIRE is straightforward: both land
  on the same `scores` tensor, just from different ops.

- **🔴 Pass-bar not numerically pinned — the gating finding.** The idea
  only states the bet *directionally* ("we expect a val-loss drop"). The
  taste-reviewer likewise did not pin a number. At tiny1m3m the in-session
  ctrl-pair gap is ~0.005–0.02 across recent runs (see `queue.md` and
  `closed.md` ctrl entries from 2026-06-09: 6.3875/6.4050, 6.4044/6.4091,
  6.3969/6.3891, 6.5991/6.6050). A real bar for this idea needs an
  explicit number tied to the box noise floor — e.g. "Δ vs the in-session
  ctrl-pair min ≤ −0.01 to count as WIN, between −0.01 and 0 = informative
  NULL (sharpening not binding), > 0 = regress." The reviser should add
  this to the Mechanism / Why-it's-worth-a-slot section so the runner
  doesn't have to invent the bar at run time.

- **Per-head vs global `s` not picked at the spec level.** The idea says
  "per-head or global" and leaves the choice to the implementer. The paper
  uses per-head; per-head is the natural pick and costs only `n_heads`
  extra scalars (negligible at 0.94M). The reviser should pick `per-head`
  explicitly so the plan doesn't dither, and so the runner's flag is
  unambiguous.

- **A/B scope vs FIRE not specified.** The idea asserts that SSMax "stacks
  on FIRE cleanly" but doesn't pin the A/B. The minimum useful A/B at
  tiny1m3m is the same one the runner already supports for the other
  single-lever ablations: `ctrl = baseline` vs `trt = baseline + ssmax`,
  with the in-session ctrl-pair variance bracket. The "stacks on FIRE"
  bet is a *follow-up* A/B (`FIRE-on` vs `FIRE-on + ssmax`), not the
  primary one. The reviser should clarify the A/B scope so the runner
  doesn't pick the wrong control.

- **Stackability with `016-qk-norm` is also worth a sentence.** Both
  modify the same `scores` tensor pre-softmax, but at different
  operations (norm-bounding vs length-scaling). The plan should note
  that they compose (and order is irrelevant since both are per-tensor
  multiplies), so a follow-up A/B of `qk-norm + ssmax` vs `qk-norm`
  alone is well-defined if 016 is now part of the active baseline.

- **Step-0 behavior note (not a finding).** At s_h=1.0, the forward is
  not bit-identical to vanilla softmax — at `n=2048` the logits are
  scaled by `log(2048) ≈ 7.6`, which is a substantial sharpening. This
  is the *point* of the paper (s learns the right magnitude) and is
  explicitly justified by the mechanism, so the
  "step-0 ≈ baseline (identity/zero-init)" check is satisfied by
  "explicitly justified" rather than "bit-identical." The reviser
  doesn't need to change anything; flagging it so the code-implementer
  doesn't add a confusing "this is an init lever" comment.

- **All hard rules pass.** tiny1m3m-only ✅, one seed (42) ✅, no
  multi-seed ask ✅, ~20 LoC drop-in ✅, mechanism not HP ✅, not a
  duplicate ✅, real source ✅, ≤3-round budget (round=1) ✅.

**Verdict: revise** — the bar / per-head / A/B-scope items above are
small spec patches, not a re-think. After the reviser applies them,
the idea is ready for the code gate (`needs-plan`).

---

## r2 — 2026-06-10 — verdict: approve

All three r1 findings are closed in the current `idea.md`. Walking each:

- **Bar is now numerically pinned** (idea.md:19-25). WIN: Δ ≤ **−0.01** vs
  the in-session ctrl, tied to the ctrl-pair bracket of 0.006–0.02
  observed in the 2026-06-09 batch. Informative NULL band: −0.01 < Δ ≤ 0
  (the "sharpening not binding at 2048/0.94M" result, still a *result*,
  to be logged to `closed.md`). Regress: Δ > 0, with the runner required
  to re-run ctrl to disambiguate box-drift before calling a clean null.
  Anti-cheat clause explicitly fences off the ±0.0053 POLYLOSS-style
  in-bracket result from being claimed as WIN. This is the kind of
  bar a runner can actually execute against.

- **Per-head `s_h` is now the spec** (idea.md:14, "single learnable scalar
  **per head**"). The "per-head or global" hedge is gone. Cost: n_heads
  extra scalars (e.g. 4 at tiny1m3m). Matches the paper. Init 1.0 with
  the mechanism-justified non-bit-identical step-0 is preserved and
  explained in-place, so the code-impl won't need to add a confusing
  "this is an init lever" comment.

- **A/B scope is fully specified** (idea.md:27-30). Primary A/B is
  `ctrl = baseline` vs `trt = baseline + ssmax`, seed 42, single run —
  exactly what the runner already supports. Stack-with-FIRE and
  stack-with-qk-norm are explicitly demoted to follow-ups, gated on
  the primary clearing (no fire). The runner will not pick the wrong
  control.

- **Stackability with FIRE and qk-norm is also a sentence now**
  (idea.md:33-35). Both compose on the `scores` line via per-tensor
  multiplies; order is irrelevant. This is enough for the plan to
  spec a future `qk-norm + ssmax` A/B cleanly when the runner is
  ready.

- **Source / mechanism / LoC / scope checks all still pass.** arXiv
  2501.19399 Nakanishi Jan 2025 (real, current, single-author). SSMax
  is a structural change to the attention line, not an HP. ~20 LoC
  drop-in: `nn.Parameter(torch.ones(n_heads))` plus a
  `scores = scores * (s_h.view(1,H,1,1) * log_n.view(1,1,T,1))` line
  before `torch.softmax`, plumbed through the existing manual
  attention branch that already hosts CoPE / qk-norm / FoX / FIRE.
  No new trainer plumbing beyond a flag. tiny1m3m-only, seed 42,
  no schedule, no aux loss. Not a duplicate of any closed lever
  on the `closed.md` axis walk.

- **🔴 Single-seed rule honored.** No multi-seed, no seed-sweep, no
  per-seed means. Sub-noise effects will be called inconclusive, not
  promoted with "add seeds to confirm."

- **No remaining findings to block `needs-plan`.** Spec is crisp
  enough for the code gate to land a real plan. Round 2 budget is
  used cleanly; the idea is ready.

**Verdict: approve** — flip to `needs-plan`, round reset to 1 so the
code gate gets a fresh budget.
