# Review log — 207 W_O Low-Rank Bottleneck

## r1 — 2026-06-15 — verdict: approve

**Why it clears the definition gate:**
- **Source real and current.** Hu et al. LoRA (arXiv:2106.09685, 2021) — real, foundational. Arora et al. "Linear Algebraic Structure of Word Senses" — real theoretical anchor for low-rank structure in transformer weights. In-repo pointers (194, 197, 160, 142, 161) all resolve.
- **Mechanism is structural, not a hyperparameter.** A learnable rank-r residual correction on `out_proj` is an architectural prior over what attention can write to the residual stream. Identity at step 0 by design. Pass.
- **tiny1m3m only.** Plan and idea both reference the 0.94M / 3M tok / seed-42 tier exclusively. No tier-mismatch.
- **Not already closed.** 171 (DropConnect W_O — regularization axis, closed null +0.0478 wrong-sign), 160 (rms gain per head — post-AV magnitude axis, null inside band), 142 (LayerScale — per-channel diagonal gain), 197 (tied W_O — sharing axis), 199 (spectral W_O reg — Lipschitz axis), 203 (pre-W_O SE — pre-projection axis) all sit on orthogonal axes. 207 is the **rank** axis on W_O — genuinely new. 194 (low-rank FFN) tests the same rank axis on a different sub-block; closed-twin logic does not apply because the placement is different and the lever is "where the rank-r correction sits," not "low-rank correction exists."
- **LoC well under 200.** Roughly: 1 config flag + 1 `α_raw` parameter + 2 small `nn.Parameter` (W_O_A, W_O_B) per block + ~15 LoC of forward code in `models/layers.py` (the `out_proj` apply site at line 6119 is the natural slot; the precedent at line 4407-4421 for 171 DropConnect shows the gate-by-flag pattern). Fits the existing `out_proj` site without architectural disruption.
- **Falsifiable.** The bet is `Δ ≤ -0.01` at tiny1m3m against the box-noise ±0.01 band. WIN would mean W_O has exploitable intrinsic-rank ≤ 16 at 0.94M; NULL would close the rank axis on W_O at this tier. Either is informative and the null branch is well-pre-registered.
- **Transfer-risk: med, defensible.** LoRA literature r=8–256 anchors the absolute rank. Effective-rank of trained projections tends to grow sub-linearly with model size (~√d_model), so r=16 at d_model=64 is a meaningful prior at this tier and not absurd at 135M (d_model~768, equivalent r ≈ 16 × √(768/64) ≈ 56, well within LoRA sweet spot). The plan should not over-claim novelty at scale — there is no published *training-from-scratch* W_O low-rank win, but the lever is well-defined and a NULL here closes a real axis.

**Findings for the plan (tightenings, not blockers):**

1. **Bit-identical is approximate, not literal.** `sigmoid(-10) ≈ 4.5399e-5`, not 0. With W_O_A/W_O_B init std=0.02 (matching the `out_proj` std=0.02 precedent at line 6043) and d_model=64, the per-element perturbation `α · (W_O_A @ W_O_B)` is on the order of `4.5e-5 × 0.02 × √64 × 0.02 ≈ 7e-7`, which is ~6× fp32 epsilon (1.19e-7). The self-check should target `max_abs_diff < 1e-5` (not "bit-identical"), or push `α_raw` to -15/-20 if a tighter bound is wanted. This does not affect val-loss interpretability — well below the box noise — but the plan should not promise exact bit-identity it cannot deliver.

2. **r=16 is hard-coded absolute; document it.** At d_model=64, r=16 = 25% of d_model. At 135M (d_model≈768), the same r=16 is ~2% — a much stronger prior. The plan should fix r=16 absolutely for this run (which is fine for tiny1m3m) and add one line in the post-mortem noting the relative-rank framing for any future scale-up. Don't make r a fraction of d_model now — keep the lever simple for the 0.94M run.

3. **Scale evidence caveat in the plan.** LoRA is *adaptation* (frozen base + trainable A,B). 207 trains A, B, and base jointly from scratch. The plan should say so in 1 sentence and note that NULL here is informative ("W_O is full-rank at 0.94M") rather than a negative for the broader rank-prior thesis. Avoid implying the run validates LoRA-style fine-tuning.

4. **α init choice needs justification.** `α_raw = -10` is a magic number. The plan should pick it from {−5, −8, −10, −15} and justify, or sweep one step (one extra step-0 fp32 max-abs-diff check at each value) and pick the smallest |α_raw| that keeps max-abs-diff < 1e-5. Don't leave it as an undocumented choice.

**Routing:** approve → `needs-plan`, round reset to 1 (definition gate budget used; code gate starts fresh).
