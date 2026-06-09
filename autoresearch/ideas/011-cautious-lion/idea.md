---
id: 011-cautious-lion
status: done
round: 1
updated: 2026-06-09T12:39:46Z
---

# 011 — Cautious Lion

## Source
Liang et al., "Cautious Optimizers: Improving Training with One Line of Code" (arXiv:2411.16085, 2024). The "Cautious" mask trick generalized from Adam to *any* sign-based optimizer. Lion (Chen et al., 2023) is a natural target: its update is `update = sign(β₁·m + (1-β₁)·g)` — already a sign — so applying the cautious mask is the cleanest possible one-liner.

## Mechanism
Lion's update direction is already sign-based. The Cautious mask zeroes the update where `sign(update) ≠ sign(g)`. Equivalent to: skip a step when the momentum and the current gradient disagree on sign. Implementation: in `optimizers/lion.py`, add a `use_cautious: bool = False` flag; in the step, after computing `update = sign(...)`, set `update = update * (update * g > 0).float()`, then rescale the update by `1 / mask.mean().clamp(min=0.1)` to keep the effective LR constant. Mask-mean clamp floor is pinned at 0.1 — deferring this to the implementer is forbidden; the plan carries the constant.

## Why it's worth a slot
- **Mechanism is orthogonal to 001/002.** Cautious-Muon and Cautious-AdamW both apply the mask to momentum-based optimizers. Lion has *no momentum buffer equivalent* — its "momentum" is the sign-of-momentum, and masking there is a different signal (it's about the update direction, not the variance).
- **Strong paper results across scales.** Liang et al. report consistent gains for Cautious variants at 7B-70B.
- **Negative result would still be informative** — would tell us Cautious is momentum-variance-specific, not sign-update-general.

## Hypothesis
Δ in [−0.015, −0.03] val loss on tiny1m3m / seed 42 vs **bare-Lion** baseline. Mechanism: skipping disagreement steps reduces wasted LR on noisy gradient signs. (Range re-stated to match the new pass bar — lower bound moved up from −0.01 so a real effect must clear noise.)

## Wiring
This is a two-lever idea and the reviewer is right that without separating them the A/B is uninterpretable. We adopt option (ii): **commit the plan to a Lion-only baseline ctrl as a prerequisite step in the same idea.** The plan produces *two* leaderboard rows: first the bare-Lion run, then Cautious-Lion vs bare-Lion. Both numbers go into the evidence file.

Required code changes:
- `optimizers/lion.py` — `Lion` class (~40 LoC) implementing Chen et al.'s update, with a `cautious: bool = False` flag that applies the mask + `1 / mask.mean().clamp(min=0.1)` rescale in the step. Mirrors `optimizers/cautious_adamw.py` structure. ~50 LoC total.
- `optimizers/cautious_lion.py` — thin subclass exposing `use_cautious_lion` flag (or just expose the `cautious=` kwarg on `Lion` and let `LLMConfig` pick — see Plan).
- `training/trainer.py:_setup_optimizers` — add `Lion` to imports (lines 14-17), add a new branch in `setup_muon_optimizer` (or a sibling `setup_lion_optimizer`) that, when `config.use_lion` is True, routes 2-D non-embedding, non-norm params to `Lion` and 1-D + embeddings to `AdamW` (Lion's standard 2-D / 1-D split — Lion replaces Muon, not AdamW).
- `LLMConfig.use_lion: bool = False` and `LLMConfig.use_cautious_lion: bool = False` (or a single `lion_cautious: "none"|"cautious"` enum).
- Routing table: 2-D `out_proj` / `W_O` + non-embedding, non-norm 2-D params → Lion; `token_embedding` / `emb_proj` / `*.norm.weight` / 1-D scalars → AdamW. Mirrors Muon routing in `trainer.py:92-156`.
- `Lion`-only baseline ctrl must be present in `LEADERBOARD.md` before the Cautious-Lion run — the plan runs the bare-Lion ctrl first, logs it, then runs Cautious-Lion against it.

Pass/fail: PASS ≤ −0.015 vs **bare-Lion** ctrl (not Muon-AdamW). NULL/INCONCLUSIVE = |Δ| < 0.01. DRIFT > +0.01. The Δ-vs-Muon-AdamW ctrl is *not* a pass criterion — it is a secondary number for context. A sub-noise effect is inconclusive, not a win.

## Notes
- Unlike 001/002, no Muon/AdamW hyperparameter conflict — Lion uses its own `lion_lr` separate from `adamw_lr`. Add `LLMConfig.lion_lr: float = 0.0003` (Chen et al.'s default, ~10× smaller than AdamW's) — pin it in the plan, don't let the implementer pick.
- Lion's sign-based update with a fixed LR has known divergence risk on the embedding; the routing above keeps embedding on AdamW specifically to avoid this. Plan must state this explicitly.
- The 2-D routing split means this idea implicitly requires adding Lion to the routing table. The plan carries the full scope, not a flag-and-forget.
