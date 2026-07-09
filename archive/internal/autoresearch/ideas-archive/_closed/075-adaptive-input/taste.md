# Taste log — 075 adaptive-input

## r1 — 2026-06-11 — verdict: reject
- **Scale-dependent mechanism, off-niche for tiny1m3m.** Adaptive Input
  Representations (Baevski & Auli 2018) buys val loss by reallocating embedding
  capacity along the Zipf tail: frequent tokens get high-rank embeddings, rare
  tokens get low-rank. That trade only fires when the long tail has *enough
  gradient signal* to be worth shrinking. Paper validates on WikiText-103
  (~100M tokens) and Billion Word (~1B tokens) — 30-300× our 3M-token budget.
  At tiny1m3m only the top ~500-1500 of the 49,152-token vocab see meaningful
  gradient counts in 733 steps; the rest are effectively random init regardless
  of bucket rank, so allocating *more* capacity to "frequent" or *less* to "rare"
  has nothing to bite on. Niche-fit rule from `idea-taste.md`: "An idea that
  only pays off at larger scale ... has no taste here regardless of how good the
  paper is." `transfer-risk: med` in the frontmatter undercounts the gap — this
  is high transfer-risk in the wrong direction (works at scale, not at tiny).
- **Information value of the A/B is near-zero.** Outcome space: (a) null,
  which we'd attribute to "tail too cold at 3M tokens" — i.e. we'd learn
  nothing the prior already says; or (b) win, which would be hard to attribute
  to bucketed-rank vs the side-effect of giving the top bucket more parameters
  (a confound a single zero-gated A/B can't disentangle). Neither outcome
  changes the 135M recipe — the screen exists to feed `beat-smollm2-135m.md`
  and a tiny-scale null on a scale-dependent lever isn't transferable evidence.
- **Embedding-capacity family is already crowded and largely closed at this
  scale.** `closed.md` notes "V/Q/K/O embeds + combos, q_gain / k_gain
  (screen20m rows 0-17)" as a closed axis; the live 10m winner is already a
  factorized tied embedding (`emb_rank=48`, ~91% of params). A bucketed
  re-allocation of that same embedding capacity is a tweak on a saturated lever
  at our scale, not a fresh bet. Cross-check with [[009-fire-pe]] / [[023-canon-conv]]
  — current attention-side levers are paying off; embedding-side has hit
  diminishing returns at tiny1m3m.
- **Net.** Reject on niche fit + low information value, not on correctness.
  The mechanism is real and well-evidenced — just at the wrong tier for this
  pipeline.
