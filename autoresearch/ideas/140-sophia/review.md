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
