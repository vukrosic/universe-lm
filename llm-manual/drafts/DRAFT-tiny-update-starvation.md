# DRAFT — At tiny1m3m the model is update-starved; update-amount levers compound, not substitute

**Hypothesis.** At tiny1m3m (≈732 optimizer steps / 3M tokens, Muon+AdamW), the model is
*under-optimized*: raising the total effective parameter-update per token (peak LR up,
Muon momentum down, batch size down) lowers val loss, and at least two of these levers
(peak-LR ×2 and momentum 0.95→0.90) **compound super-additively** rather than substituting.

**Believed because.** At 732 steps the optimizer has not reached the loss basin, so more
update-per-token moves it closer; LR and momentum scale different terms of the effective
step, so they can stack. Measured paired Δ vs the combo-296 champion:
- ×2.0 peak-LR alone: **Δ−0.0104** (3-seed, 306) — sub-band.
- Muon momentum 0.90 alone: **Δ−0.0085** (317) — sub-band.
- Both together (323): **Δ−0.0278** point estimate > the −0.0189 you'd get if they merely
  added → **super-additive**. Per-seed −0.0185 / −0.0462 / −0.0188 (seeds 42/123/7), all
  same sign.
- Direction control: `grad_accum=2` (halves the update count) → **Δ+0.0624** (worse), and
  `bs=1` (doubles steps) → **Δ−0.0104** — both confirm "more updates → lower loss."
- Substitution control: not every update-lever pair compounds — 322 (momentum + bs=1)
  **anti-stacked** (Δ ≈ 0 vs the better single lever). So "compounds" is pair-specific, not
  a blanket optimizer law.

**Test.** The paired confirms above (`_arq_confirm_306`, `_arq_confirm_323`, 317/322 logs),
all vs the same combo-296 champion control on the same box/seeds.

**Predicted.** Within this scope, any lever that increases effective update/token gives
Δ<0; LR×momentum compound; LR×batch substitute. Outside the short-horizon regime the sign
should weaken toward 0.

**Promotes to.** **L?** (tentative) — NOT L!. The combined point estimate clears 0.02, **but
with n=3 the t-based 95% CI on Δ−0.0278 is ≈ [−0.067, +0.012] and includes 0** (driven by
the seed-123 outlier −0.0462). The daemon promoted 323 on a point-estimate band (|Δ|≥0.018),
which is *weaker* than the book's "clears the noise" gate (|Δ|≥0.02 **and** CI excludes 0).
The *direction* (under-optimization + compounding) is well-supported; the *significance of
the magnitude* is not yet. → **L!** needs ≥5 seeds bringing the CI off 0.

**Falsifier.** At a ≥10× token budget the same levers give Δ≥0 → bounds the law to the
short-horizon (under-trained) regime. Or: a 5+-seed re-run of 323 whose CI includes 0 at a
smaller mean → demotes the super-additivity to noise.

**Scope.** tiny1m3m only: d_model=64, 12L, ~1.3M params, 3M tokens / 732 steps, seq 512,
Muon(lr 0.024 base)+AdamW(lr 0.006 base), combo-296 champion recipe. A law **strictly inside
this line** — explicitly *not* claimed at any larger size or longer schedule.

**Note (RULE 0).** This entry is the scientific *record* of the optimization axis and why it
is **closed** — the loop does **not** re-mine optimization hyperparameters (operator
directive 2026-06-17: novel architectures only). Recording it here closes the axis with
evidence; it does not reopen HP search.

**Evidence so far.** partial→strong-direction. Champion lineage `autoresearch/champion.json`
(323 entry, val 6.1720), `_arq_confirm_323.py`. CI computed here, not in the daemon.

**Blocked on.** ≥5-seed paired re-run of 323 to move the magnitude CI off 0 (for L!).
