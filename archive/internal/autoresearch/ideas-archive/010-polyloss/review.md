# Review log — 010 polyloss

## r2 — 2026-06-09 — verdict: approve
- **All 6 r1 blockers resolved in spec.** Re-read end-to-end against `idea.md` frontmatter + body:
  1. ✓ ε₁ pinned to **1.0** (idea.md:20), with explicit "no sweep" + on/off flag defaulting off. The principled-next-Taylor-term framing is intact.
  2. ✓ Falsifiable pass/fail bar with three explicit states (PASS / NULL / DRIFT) anchored to leaderboard ≈6.4287 (idea.md:22-27). NULL is framed as informative, not failure. The 0.005–0.01 inconclusive window is explicit.
  3. ✓ Train-only reporting rule spelled out with exact file:line anchors (trainer.py:372-377, :403-408) and explicit "leave evaluation.py:53 on plain CE" — directly comparable to leaderboard.
  4. ✓ `-100` ignore_index mask rule stated with concrete code shape (idea.md:39-44). `-100 -> 0` via `clamp(min=0)` is harmless because the `* mask` zeros the gradient contribution at those positions either way.
  5. ✓ LoC budget called out with diff sites and ~10–20 LoC total estimate, well under 200.
  6. ✓ LabelSmooth disambiguation framed honestly in idea.md:53-54 — mechanism-distinct (model's own gold probability vs fixed Dirichlet prior) but acknowledged empirically inseparable at this tier; result framed as "j=1 polynomial correction vs CE," not "PolyLoss vs LabelSmooth."
- **Source ✓**: arXiv:2204.12511 (Leng et al. 2022) — real, well-cited. Not in `closed.md` (grep confirms zero hits for polyloss / 2204.12511).
- **Mechanism ✓**: structural loss-shape change, identity-safe at step 0 (`ε₁=0` ≡ baseline CE via off-flag). Distinct from existing `label_smooth` in `configs/output_head_ablations.py` (Dirichlet prior over vocab vs model's own `(1-p_t)`).
- **Tier ✓**: tiny1m3m / seed 42 only. No screen20m, no full ladder, no multi-tier promotion.
- **LoC ✓**: 8–12 LoC across trainer.py + a new config dataclass; 0 LoC in evaluation.py.
- **Decision**: approve. Spec is sharp, falsifiable, and identity-safe. Reset round to 1 → code gate.

## r1 — 2026-06-09 — verdict: revise
- **Source ✓**: PolyLoss (Leng et al. 2022, arXiv:2204.12511) is real and well-cited. The j=1 simplification `L = L_CE + ε₁·(1 - p_t)` is the form the paper itself flags as the strong default; implementation is loss-head only, ~3-5 LoC against `training/trainer.py:372-408` and `training/evaluation.py:53`.
- **Mechanism ✓**: structural loss change (content-dependent correction), not a hyperparameter sweep. ε₁=0 ≡ baseline CE, so the flag is identity-safe at step 0. Distinct from `label_smooth` in `configs/output_head_ablations.py:32` (uniform prior over vocab) — PolyLoss's `(1 - p_t)` is a function of the model's own gold probability, not a fixed Dirichlet. Orthogonal to 007-sigmoid-loss's bounded-gradient story (which was closed anyway). Not in `closed.md`; no duplicate lever.
- **Tier ✓**: tiny1m3m / seed 42, no model-shape change, no compute cost, transferable across scale. Fits the niche.

**Findings (must be fixed in the spec, then this passes the gate):**

1. **Pin ε₁ to a single number (BLOCKER).** The idea says "per-task hyperparameter ε₁ (a few j=1 coefficient)" — that is exactly the ambiguity that makes the result unresolvable at box noise ~±0.01 val loss. The paper's "strong default" recommendation for classification (CIFAR, ImageNet, LMs) is **ε₁ = 1.0** — it is literally the next Taylor term in the series expansion of `-log p_t = (1-p_t) + (1-p_t)²/2 + …`, so 1.0 is principled, not tuned. The spec must name **ε₁ = 1.0**, with the on/off flag defaulting off. No sweep. The taste-reviewer's r2 verdict ("pin the single ε value") made this the spec gate's job — pin it.

2. **Name a falsifiable pass/fail bar (BLOCKER).** The idea's stated expected Δ is −0.005 to −0.02, the lower end of which is inside box noise and the upper end of which would be a clean win. A spec that does not pre-commit to a decision rule is uninterpretable. The bar at this tier should be:
   - **PASS** (win): treatment val loss ≤ control val loss − 0.005 (≥0.005 absolute improvement).
   - **NULL** (clean, loggable): |Δ| < 0.005 vs control. Write `evidence.md` with verdict NULL, append one line to `closed.md` per `PIPELINE.md` Run + Evidence rules. The null is informative ("CE's j=1 truncation term is negligible at this scale"), do not pretend it is a failure.
   - **DRIFT**: |Δ| > 0.01 → box validation (`LEADERBOARD.md` baseline ≈ 6.4287), rerun or kill the slot.
   Anything in the 0.005–0.01 window → log inconclusive; do not promote.

3. **Reporting rule: train-only (must be explicit in spec).** The existing loss-side aux family in `configs/output_head_ablations.py` (ZLoss, LabelSmooth, ConfPenalty) is **train-only**; eval stays plain CE for the leaderboard. PolyLoss is a loss-shape change, not a logit op, so it should follow the same reporting rule: apply the ε₁·(1-p_t) correction in `trainer.py:372-377` and `trainer.py:403-408` only; **leave `evaluation.py:53` on plain CE** so the val_loss number is directly comparable to the leaderboard. State this explicitly in the spec — easy to miss, and getting it wrong biases the leaderboard.

4. **Mask handling must match the existing CE path.** `trainer.py:368-370` builds `shift_labels = y[:, 1:]` with the last token set to -100 (and the non-AMP path does the same). The PolyLoss term `ε·(1 - p_t)` must be masked with the same `ignore_index=-100` positions — otherwise the correction leaks into the last token of every sequence and silently biases the gradient. Spec must require `p_t = softmax(logits)[..., t]` only at positions where `t != -100`, or equivalently multiply the per-position correction by `(shift_labels != -100).float()`.

5. **LoC estimate is realistic; flag the diff sites in spec.** Concretely: ~3-5 LoC in each of the two trainer.py loss blocks (extract the j=1 term behind `if getattr(config, "use_poly_loss", False):` guard so the base path is byte-identical), 0 LoC in `evaluation.py` (no change), and 2 LoC in a new `Tiny1M3MPolyLossConfig(LLMConfig)` dataclass (or appended to `configs/output_head_ablations.py` next to the LabelSmooth anchor). Total ≈ 10-20 LoC, well under the 200-LoC cap.

6. **Acknowledge the disambiguation risk in the spec, do not let it kill the idea.** A PolyLoss win and a LabelSmooth win are not mathematically identical, but at tiny1m3m the val-loss delta is unlikely to be large enough to distinguish them empirically. The spec should note this — and frame the result as "the j=1 polynomial correction is X better than plain CE," not "PolyLoss is X better than LabelSmooth." If the result is NULL, the writeup should be honest that a sweep at this scale cannot separate the two.

**Decision**: revise. The mechanism and source are sound; the blockers are all in the *spec*, not the idea. After the reviser applies findings 1-4 (5 and 6 are notes), this is `approve`.
