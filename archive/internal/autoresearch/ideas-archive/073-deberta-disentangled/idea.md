---
id: 073-deberta-disentangled
status: planning
round: 1
updated: 2026-06-13T04:05:12Z
transfer-risk: low
---

# 073 — DeBERTa Disentangled Attention

## Source
DeBERTa: Decoding-enhanced BERT with Disentangled Attention (arXiv:2006.03654). 2020.

## Mechanism
**Replaces FIRE** (does not stack — 013-cope showed stacking PE is destructive).
Adds a *content-position cross-term* `Q_c(H_i) · K_p(P_{i-j})` to attention
logits, where `P_{i-j}` is a learnable relative-position embedding indexed by
clipped distance (clip=64), and `K_p` is a position-key projection **shared
across all 6 layers** (single global kernel). Final logit = `Q_c · K_c +
Q_c · K_p`. Position embeddings zero-init → step-0 identical to baseline
content-only attention. Param overhead: shared `K_p` (192² ≈ 37k) +
`P` (129 × 192 ≈ 25k) = ~62k = **~6.6% of 0.94M** — meaningfully cheaper than
per-layer DeBERTa. Implementation ≈ 60 LoC in `models/layers.py`.

## Scale evidence
DeBERTa scales to 1.5B params with strong SuperGLUE gains. The
content-position decomposition is mainstream at 100M+. transfer-risk: low —
mechanism is a position-branch addition with exact step-0 identity.

## Differentiation from 072-t5-rpe (the binding question)
**What 073 does that 072 cannot, in one sentence:** the cross-term
`Q_c(H_i) · K_p(P_{i-j})` makes the distance-bump *content-conditioned* —
the same relative distance Δ produces different attention boosts for
different query tokens, because the bump is a dot product against
`Q_c(H_i)`. 072's bucketed bias is a single scalar `b_h[bucket(Δ)]` added
to every query at that distance — content-blind by construction. **Concrete
mechanism prediction:** punctuation and function-word queries (which have
sharply position-dependent attention — `)` finds `(`, `."` finds the
sentence opener) can learn per-token distance preferences that 072 must
spread across head specialization. **Read-off:** on tokens whose POS bucket
differs from the local average, 073 − 072 should be measurably negative;
on content-token-only positions the two collapse to similar bias shapes.
If 073 ≤ 072 on val loss, the cross-term carries no content-conditional
signal worth its 6.6% param tax at tiny1m3m.

## Cold-start at tiny1m3m (~92 steps, 3M tokens)
The position branch sees *every* relative distance on *every* training
token — 65k tokens/step × 92 steps × ~64 distance buckets in scope ≈ a
massively oversampled prior. By contrast, content-content projections wait
for rare token co-occurrences. Position embeddings `P` are 129 × 192 (clip
=64 → 2×64+1 buckets), shared `K_p` is 192². At one update per token, each
position bucket is hit ~50k times in the run — comparable to a high-freq
vocab token. Expected to fire well before step 92; if it doesn't, that
*is* the answer (cross-term too cold for this budget).

## Why it's worth a slot
Owns the fight against the live WIN (009-FIRE, Δ −0.064/−0.082) and the
in-pipeline sibling (072, content-blind bucketed bias). Three terminal
outcomes, each informative:
- **Beats 072 (Δ ≤ −0.005 vs 072 ctrl):** content-conditional distance
  prior pays — closes 072 as redundant, opens cross-term family for
  Phase-2 stacking with 023-canon-conv.
- **Ties 072 (|Δ| < 0.005):** cross-term adds nothing over bucketed bias
  at this scale; **close** the disentangled-PE family at tiny1m3m
  (`closed.md` line: "content-position cross-term needs ≫92 steps") and
  let 072 carry the bucketed-bias slot alone.
- **Loses to 072 (Δ ≥ +0.005):** 6.6% param tax not earned — closes
  per-layer position projections as a sub-family at our budget; sharpens
  the lesson that PE wins at tiny1m3m come from *cheap* priors (FIRE's
  fixed kernel, 072's tied table), not learned content×position
  interactions.

All three outcomes are loggable to `closed.md` with a one-line takeaway —
no "inconclusive" tail.
</content>
</invoke>
