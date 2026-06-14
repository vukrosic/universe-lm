# Review log — 140-sophia

## r1 — 2026-06-14 — verdict: revise
- **Spec gap: pass/fail bar not quantified.** The "Why it's worth a slot" section
  frames the bet qualitatively ("paradigm shift if it works, closes the
  second-order axis if it doesn't") but does not commit to a Δ number. Add a
  `### Pass bar` subsection to the plan that pins the WIN bar to **Δ < -0.01
  val_loss vs the cached baseline (6.4394 ± 0.04)** and the null band to
  **|Δ| < 0.01** — same convention used by 152-attn-logit-bias, 153-relu2-ffn,
  155-per-head-temp, 156-moa. Without a number, the runner cannot resolve
  a 6.43 ± 0.05 result (Sophia at tiny1m3m will be inside the noise band
  per the existing optimizer-axis nulls: 119-sam, 122-tiger, 124-radam,
  125-psgd, 128-spectral-decoupling, 134-mega-ema, 135-adan, 139-lion all
  closed null at this tier).
- **Code bug: `adamw_params` is local to `setup_optimizer()` (line 478) but
  the Sophia Hutchinson block inside `train_model()` references it at
  lines 1746, 1785, and 1796 — `NameError: name 'adamw_params' is not
  defined`.** This is the same crash the auto-implement tried 3× to fix
  and gave up on (see `log.jsonl` 2026-06-13T21:32:10Z and
  2026-06-13T21:32:14Z). The fix is one of:
  1. Add `adamw_params: List[nn.Parameter]` to `train_model()`'s signature
     and thread it through from the caller in `main()` (cleanest, matches
     the existing `optimizers` / `schedulers` threading).
  2. Derive it inside the Hutchinson block from
     `sophia_opt.param_groups[*]['params']` (least invasive, keeps the
     function signature unchanged).
  Either is fine; pick (2) to keep the diff small. Whichever path is
  chosen, document it in the plan so the next code-implementer pass
  doesn't repeat the same 3 recode attempts.

**Verdict rationale:** the idea is sound — Liu et al. 2023 (arXiv:2305.14342)
is a real second-order optimizer with a substantively different mechanism
from the 110–139 AdamW-variant wave (diagonal Hessian preconditioner vs
adaptive-LR scaling). The transfer-risk:med tag is justified (paper
validated at 125M and 1.5B; 0.94M is 130×+ below the tested range, and
Hutchinson noise is known to grow at small scale). The plan's LoC budget
is tight (~200 + 80) but feasible, and step-0 identity is honestly
documented (forward is bit-identical when `use_sophia=False`). The two
findings above are both tractable: a one-paragraph bar spec and a
one-line scope fix. `revise` (not `reject`) — the spec is salvageable
and the second-order axis has not been closed at 0.94M yet.

## r2 — 2026-06-14 — verdict: revise
- **Pass bar now quantified** — `### Pass bar` section now pins
  `WIN`/`NULL`/`DRIFT` thresholds to the cached `6.4394 ± 0.04` baseline.
  ✓ finding closed.
- **`adamw_params` scope fix is documented in the plan** but the
  reviser's plan amendment says "build a local `adamw_params` list from
  `sophia_opt.param_groups[*]['params']`" (option 2 from r1). Plan
  amendment is in place; code-level verification is the next gate's job.
  Carry forward.

**Verdict rationale:** both r1 findings are addressed at the plan level.
The code-level `adamw_params` scope fix is the next code-implementer
pass's responsibility; do not bounce back here.

## r3 — 2026-06-14 — verdict: reject
- **🔴 The r1/r2 `adamw_params` scope bug is NOT actually fixed in code.
  The r2 review note flagged that the plan was amended to derive
  `adamw_params` from `sophia_opt.param_groups[*]['params']`, but
  `training/trainer.py` line 1746 (and 1785, 1796, 1805) still references
  the bare name `adamw_params`:**
  ```python
  1746:                                for p in adamw_params
  1785:                                _hess_loss, adamw_params,
  1796:                            hv_list = torch.autograd.grad(
  1796:                                g_u, adamw_params,
  1805:                            hv_list = torch.autograd.grad(
  1805:                                g_u, adamw_params,
  ```
  `adamw_params` is defined inside `setup_muon_optimizer()` at
  `training/trainer.py:478` (local to that function), is NOT in
  `train_model()`'s argument list (verified: args = `model, config,
  train_loader, val_loader, optimizers, schedulers, ...`), and has no
  `global` / module-level binding. The first call into the Hutchinson
  block will raise `NameError: name 'adamw_params' is not defined` —
  exactly the same crash the r2 runner hit (see `log.jsonl` 2026-06-13T21:32:10Z
  and 21:32:14Z). The reviser updated the **plan** but did NOT touch
  the code. `git blame` on `training/trainer.py:1746` is `bd5adf58` (the
  original Sophia commit); no subsequent commit has modified that line.
  This is a provable, in-tree drift between the plan's claim and the
  code's reality.
- **🔴 Round-3 cap forces this decision.** Per the protocol, round 3
  forbids `revise` — the gate must `approve` or `reject`. Approving
  an idea whose documented scope fix is not in the source tree would
  push the same `NameError` back to the runner for a third time. Same
  pattern as 150-xlayer-feedback (r1 11.39, r2 9.77, r3 11.89; all
  spec/code-drift, all closed as `reject: round-3 cap hit at definition
  gate`). The mechanism is sound but the implementation has been broken
  for 2 rounds and a third approval on a known-bad source is not
  defensible.

**Verdict rationale — what is and is not closed.** The plan-level
amendments from r1 (pass bar) and r2 (`adamw_params` derivation
strategy) are both correctly documented in `idea.md`. The mechanism
(Liu et al. 2023, arXiv:2305.14342) is real and the source citation
holds; the diagonal Hessian preconditioner IS a real lever vs the
110–139 AdamW-variant wave; the transfer-risk:med tag is justified
(0.94M is 130×+ below the paper's tested 125M/1.5B range, and
Hutchinson noise grows at small scale); the 200-LoC budget is over by
~25% (sophia.py = 254 LoC) but not the deciding factor. What is NOT
closed is the provable `NameError` that the runner will hit on first
launch — the r1 reviewer explicitly said "Whichever path is chosen,
document it in the plan so the next code-implementer pass doesn't
repeat the same 3 recode attempts", and the reviser did the
documentation step but not the code step. The same class of
plan/code-drift failure as 150-xlayer-feedback and 115-rdrop. Per the
precedent those two set (round-3 cap + unrunnable code = reject), and
per the protocol's 3-round budget, this is `reject`. The Sophia axis
is not closed at 0.94M (could close null at the same tier with a clean
code drop, OR could close as a null result on a working A/B); the
cap is on the **spec/code drift**, not on the mechanism.
