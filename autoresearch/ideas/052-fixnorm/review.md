# Review log — 052 FixNorm

## r1 — 2026-06-11 — verdict: revise

- **No falsifiable pass/fail bar.** "Should move loss" is qualitative. Box noise at tiny1m3m is ~±0.01 val loss; add a concrete Δ threshold against a *named* control. Suggested form (match peer ideas 016/021/023/024/025): "WIN if treatment val loss is ≤ ctrl - 0.005 at tiny1m3m, seed 42, vs the current best baseline (FIRE-PE + QK-norm equipped)." Pin the exact control config name the runner will use, not "vs baseline."
- **Step-0 is not bit-identical and the idea does not justify it.** Per-token L2 normalization changes init magnitudes immediately (init rows have ‖row‖₂ ≈ 0.02·√d_model; after FixNorm + radius `g`, every row has ‖row‖₂ = `g`). Pipeline rule: "step-0 ≈ baseline (identity/zero-init) unless explicitly justified." Pick one: (a) initialize `g` so step-0 ≈ baseline (e.g. set `g = 0.02·√d_model` so the mean post-norm magnitude matches baseline), and say so; or (b) add a one-sentence justification that magnitude *equalization across rows* is the mechanism and is what we want to test from step 0.
- **Mechanism is under-specified — name the exact recipe (the reviser writes the spec; this stays out of the plan-stage's hands).** Add a `## Spec` section that fixes all four ambiguities:
  1. **Radius parameterization:** one global learnable scalar `g` (paper default), one per-row, or fixed constant? Default should be the paper's one global scalar.
  2. **Interaction with existing `embedding_scale = √d_model`** in `models/llm.py:558-569` (`x = tok * emb_scale`). Does FixNorm *replace* that scaling (set `embedding_scale = -1` and let `g` carry the radius), *multiply* with it, or *precede* it? Pick replace: `g` is the only magnitude knob. Say so.
  3. **Where it sits w.r.t. low-rank embedding** (`emb_rank`): on the rank-r table (before `emb_proj`), or on the post-projection d_model vector? Default config has `emb_rank=None`, but spec must say (default: post-projection d_model, so the lever applies uniformly across config variants).
  4. **Output side.** `lm_head.weight = token_embedding.weight` is shared by reference at `models/llm.py:509-510` (full-rank case), but FixNorm normalizes *at lookup time*, not on the stored weight. State explicitly: "logits use the raw shared weight (no FixNorm at the output site); the lever is input-side only." If the intent is symmetric clamping at both sites (the paper's "tied" variant), say that instead and own the extra LoC.
- **Name the control config the runner pulls.** Findings 1 and 3 collapse if the spec says: "control = `Tiny1M3MBaselineConfig` (current best baseline with FIRE-PE + QK-norm). Treatment = same + `use_fixnorm=True`, `fixnorm_radius_init=...`." That's also what the code-implementer needs, so write it now.

## r2 — 2026-06-11 — verdict: approve
- The recipe is now fully pinned: control, treatment, radius init, placement, and step-0 rationale are all explicit.
- The pass/fail bar is numeric and tied to a named control.
- This is ready to move into planning.
