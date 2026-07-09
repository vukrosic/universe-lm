# codereview — 011-cautious-lion

## r1 — 2026-06-09 — verdict: accept

- **Faithful to the mechanism.** `optimizers/lion.py` implements Chen et al.
  (2023, arXiv:2302.06675) Lion: `c = β1·m + (1-β1)·g`, `update = sign(c)`,
  `m ← β2·m + (1-β2)·g`, decoupled weight decay. The Cautious branch
  (Liang et al. 2024, arXiv:2411.16085) computes
  `mask = (update * g > 0).to(update.dtype)` and rescales by
  `1 / mask.mean().clamp(min=0.1)`. Matches the idea verbatim; the
  `clamp(min=0.1)` floor is hard-coded in the implementation (not
  deferred), as r1 plan-level note required.

- **Identity / zero-init at step 0 holds.** Verified by smoke test: at
  step 0, `m=0`, `c=(1-β1)·g`, `sign(c)=sign(g)` (since `1-β1 > 0`),
  so the mask is all-ones and the rescale factor is `1/1.0 = 1.0`. The
  cautious update reduces to `sign(g)`, identical to the bare-Lion
  step-0 update. Ran `Lion(cautious=True).step()` vs
  `Lion(cautious=False).step()` with identical `g`:
  `max diff = 0.0` at step 0.

- **Default OFF path is bit-identical.** With `use_lion=False`
  (default), the trainer's `setup_muon_optimizer` returns
  `[Muon, AdamW]`, prints `Lion parameters: 0`, and instantiates
  `muon_optimizer = Muon(...)` exactly as the pre-PR code did. No new
  ops or reordering on the default branch — verified by reading the
  diff and by constructing `MinimalLLM(Tiny1M3MConfig())` end-to-end
  and confirming the optimizer list is unchanged.

- **Wiring is complete and not the recurring break.** `use_lion` and
  `use_cautious_lion` are read via `getattr(config, ..., False)` in
  `training/trainer.py:_setup_optimizers` (lines 106-107). The flags
  flow: `LLMConfig.use_cautious_lion` → `getattr` → `Lion(cautious=
  use_cautious_lion)` → `state['cautious']` in the param group.
  Verified end-to-end: `setup_muon_optimizer(MinimalLLM
  (Tiny1M3MCautiousLionConfig()))` returns `[Lion(cautious=True),
  AdamW]`; `setup_muon_optimizer(MinimalLLM(Tiny1M3MLionConfig()))`
  returns `[Lion(cautious=False), AdamW]`. The `Lion(cautious=)` kwarg
  is *not* dead — both branches instantiate, and the cautious mask
  branch in `lion.py` is gated by `if cautious:` (line 89), so the
  no-cautious path skips the mask computation entirely.

- **Routing matches the plan.** Lion replaces Muon on the 2-D
  non-embedding, non-norm slot. 1-D / `token_embedding` / `emb_proj` /
  `*.norm.weight` stay on AdamW (same `is_muon_candidate` predicate
  the Muon path uses at `trainer.py:110-115`). Verified param counts
  on Tiny1M3M: 541,568 to Lion/Muon, 407,488 to AdamW — the routing
  split is identical. The Lion and SOAP paths are routing-disjoint
  (SOAP fires on `not is_muon_candidate` at `trainer.py:155-162`),
  so co-activation of `use_lion` and `use_soap` is well-defined.

- **LoC budget.** `optimizers/lion.py` is 116 lines (95 non-comment).
  Plan budget was 50-80 LoC; the extras are docstrings (lines 1-24,
  30-43) and a `g.float()` / `m_fp` mixed-precision cast that matches
  the Muon path (lines 78-83, 113-114). Trainer additions: ~50 LoC
  (the `lion_params` list, the `getattr` reads, the routing branch
  at `trainer.py:163-173`, the Lion instantiation at `trainer.py:
  182-198`, the optimizers-list branch at `trainer.py:270-271`).
  llm_config.py additions: 5 fields + 2 dataclasses (~50 LoC).
  `optimizers/__init__.py`: 2-line change. Total is well under the
  200-LoC budget for the idea.

- **No silent HP drift.** `lion_lr=3e-4`, `lion_beta1=0.9`,
  `lion_beta2=0.98` are pinned in `LLMConfig` defaults (matches
  Chen et al.'s values, and matches the plan). No LR/schedule/init
  changes to the baseline path. Seed is 42, one seed (plan.md
  Control section, line 64).

- **Plan ↔ idea consistency.** Pass/fail bar in `plan.md` (lines
  96-100) matches `idea.md` §Wiring / §Hypothesis verbatim: PASS
  ≤ −0.015 vs bare-Lion ctrl, NULL |Δ| < 0.01, DRIFT > +0.01.
  Two-run protocol: `Tiny1M3MLionConfig` (ctrl) + `Tiny1M3MCautious
  LionConfig` (treatment) — both defined as recipe dataclasses in
  `llm_config.py` (lines 574-598). Tier: `tiny1m3m`. Seed: 42.

- **Coordination.** No edits to `models/layers.py` or `models/llm.py`
  (mechanism is at the optimizer level, not the model level). Diff to
  `configs/llm_config.py` and `training/trainer.py` is additive
  (new fields, new branch, new optimizer) and does not stomp on the
  parallel Claude's edits to those files — the changes are scoped to
  the Lion lever and don't touch SOAP / schedule-free / cautious-AdamW
  / cautious-Muon / retnet / poly-loss code paths. No rebase, no push.

Non-blocking observations:
- The plan called for an explicit `use_lion` ↔ `use_muon` mutex
  assertion in the trainer; the current code makes it implicit
  (when `use_lion=True`, `muon_optimizer=None` and the routing
  branch sends the params to Lion). This is r2 plan-level note (a),
  not a blocker — the existing code is safe, just less defensive
  than the plan sketched. Worth a one-line `assert` in a future PR
  if/when `use_muon` becomes a real flag.
- The 2-D / 1-D / embedding routing logic could be factored into a
  helper (`_route_params(model, config) -> (muon, adamw, soap, lion)`)
  if a third optimizer is added — at four optimizers the inline
  branches are still readable, but the file is getting long. Out of
  scope for this PR.
