## r1 — 2026-06-09 — verdict: accept

- **Mechanism match.** `models/layers.py:1862-1865` constructs `sub_ln_attn` /
  `sub_ln_ffn = nn.LayerNorm(d_model, elementwise_affine=True)` per block (γ=1,
  β=0 default). `models/layers.py:2042-2043 / 2052-2053` (post-norm branch)
  and `:2064-2065 / 2082-2083` (pre-norm branch) wrap each sublayer output
  AFTER the layerscale (when present) and BEFORE the residual add / norm
  step. Faithful to the spec: `y = x + LN_post(Sublayer(LN_pre(x)))` for
  pre-norm, `y = x + LN_post(Sublayer(x))` for post-norm. ✓
- **Flag wiring.** `configs/llm_config.py:286` — `use_sub_ln: bool = False`
  (default OFF). Passed through `models/llm.py:241, 349` into
  `TransformerBlock.__init__` (`:1703`), then guarded in forward by
  `if self.use_sub_ln:` at four call sites. Single boolean, default OFF,
  treatment path exercises the new code. ✓
- **Identity at step 0 — divergence, accepted.** The docstring/idea claim
  "γ=1, β=0 ⇒ identity at step 0 ⇒ baseline preserved" is mathematically
  incorrect (verified: `nn.LayerNorm(64)` on a nonzero-mean/-var input gives
  mean diff 4.1, max diff 15.8 — pure mean-centering + unit-var scaling, NOT
  identity). `plan.md` already acknowledges this and confirms via forward
  smoke test (max diff 0.058 at the actual model scale). The mechanism is
  what DeepNet tests — sub-LN takes effect at step 0 — so the A/B is clean:
  ctrl (flag OFF) is bit-identical to pre-norm baseline; trt (flag ON) is
  the lever. Implementation is correct; docstring/idea is just imprecise.
  **Finding (minor, not blocking):** the docstring at
  `models/layers.py:1856-1861` and the implementation comments at
  `:2038-2041, 2060-2063` repeat the "identity at step 0" claim. Suggest
  a one-line edit next pass to "γ=1, β=0 ⇒ standard LayerNorm behavior
  (mean-center + unit-var), the lever takes effect at step 0" — for
  honesty, not for correctness.
- **No silent HP drift.** No LR/schedule/init constants changed in the
  diff. New LN uses default `nn.LayerNorm(d_model, elementwise_affine=True)`
  (γ=1, β=0). ✓
- **Param routing.** `sub_ln_attn.weight` / `.bias` and `sub_ln_ffn.weight`
  / `.bias` are 1-D → routed to AdamW in `training/trainer.py:109-115` (the
  `param.ndim == 2` Muon gate fails, falls through to `adamw_params`).
  Correct. ✓
- **LoC budget.** Net new code in `models/layers.py`: ~10 LoC
  (1 attr + 1 if + 2 LN constructions + 4 if-guards + 4 wrap lines).
  With comments: ~30 LoC. Under the 20-LoC stated budget for the code-only
  portion, well under 200 LoC total. ✓
- **Plan ↔ idea consistency.** PASS bar Δ ≤ −0.005, NULL |Δ| < 0.01,
  DRIFT > +0.01. Single seed 42, tiny1m3m tier. Plan's "−0.005" PASS bar
  is consistent with the taste review's depth-stability framing (lever is
  bounded at 6 layers; null is the more informative outcome). ✓
- **Coordination.** Diff to `configs/llm_config.py`, `models/layers.py`,
  `models/llm.py` is scoped to the `use_sub_ln` slice — no stomp of the
  parallel-AI edits to other flags (CoPE, QK-Norm, Moonlight, Lion,
  Cautious-Lion). Each is gated by its own boolean, all default OFF. ✓

### Minor findings (not blocking — note for the runner / next maintainer)

- **Plan cost-claim math error.** `plan.md` "Cost" § says
  `2 × d_model² = 2 × 64² = 8,192 extra params/block` at tiny1m3m. But
  `nn.LayerNorm(d_model)` has 2 × d_model params (γ + β), not d_model².
  Real cost: 2 LNs × 2 × 64 = 256 params/block × 12 blocks = **3,072
  params** at tiny1m3m (+0.32% of 949k), not +98,304 (+10.4%). Screen10m
  is 576/block × 24 = 13,824, not ~1.0M. Implementation is correct; the
  cost footnote in `plan.md` is off by ~32×. Doesn't gate the A/B — the
  param count is still negligible — but worth a one-line fix next time
  plan.md is touched.
- **Parallel-block gap.** `use_parallel_block=True` forward path
  (`models/layers.py:2015-2028`) returns before the sub_ln guards and
  does NOT wrap attn_out / ff_out. If both `use_parallel_block=True` and
  `use_sub_ln=True` are set, the LNs are allocated but never invoked
  (dead params). The planned A/B is tiny1m3m, no parallel block — not
  a current A/B concern. Suggest a follow-up idea if parallel + sub-LN
  ever needs to be tested together.

### Verdict

**accept** — mechanism faithful, identity-safe (flag OFF ⇒ baseline path
untouched), no HP drift, single boolean default OFF, param routing correct,
LoC under budget. The two minor findings (docstring imprecision, cost-math
error) are non-blocking and can be cleaned up in a future pass.
