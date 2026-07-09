# taste.md — research-taste review log

## r1 — 2026-06-11 — verdict: reject

**Spirit-duplicate of in-queue `049-cosformer`**, weaker framing:

- **Same paper, same arXiv ID, same mechanism, same bet.** Both 049 and 079
  cite Qin et al. arXiv:2202.08791 ("cosFormer: Rethinking Softmax in
  Attention") and both reduce the paper to "swap the softmax kernel for a
  cosine one and add a zero-init gate so the model starts as baseline." There
  is no information in running this twice; the spirit check in
  `prompts/idea-taste.md` §2 explicitly says to dedup on spirit, not just
  exact slug.
- **079 is the inferior framing.** 049 correctly names cosFormer's actual
  contribution (linear-time reweighting kernel + per-head cos/sin features),
  with `transfer-risk: med`. 079 reduces the paper to "angular similarity vs
  dot-product similarity" and tags `transfer-risk: med-high` — that framing
  strips the linear-time point and turns the lever into a generic "swap
  kernel, see what happens" bet, exactly the kind of vibe the taste gate
  exists to push back on.
- **Portfolio is already saturated with this family.** Active needs-taste
  queue contains *five* efficient/linear-attention variants in the same
  spirit cluster: 049-cosformer, 050-performer, 077-fnet, 078-fnetar,
  080-fastformer. The protocol's "5th optimizer-momentum variant" example
  was written for exactly this case — a sixth variant of "replace softmax
  with X" is a slot-allocator's nightmare, and a duplicate slug for the
  same paper is worse than a parallel one.
- **Information value at tiny1m3m is near zero either way.** cosFormer's
  selling point is linear time, which is irrelevant at 0.94M params / short
  context. The only thing a clean A/B would teach is "cosine reweighting is
  a wash at this scale" — that null is already implied by the closed-norm-zoo
  and closed-MHA-reshape axes, and one of the in-queue 049/050/077/078/080
  slugs will get there for free.
- **What the miner should do instead.** If the bet is "angular similarity
  beats dot-product at this scale," that should be a sub-experiment inside
  049-cosformer's plan (a single ablation row, not a fresh slot). If the
  bet is "cosFormer-style linear attention beats softmax," that *is* 049
  and the slot already exists. Do not file a third cosFormer slug.

**Action:** reject 079. Recommend the miner merge any angular-similarity
framing they wanted to test into 049's plan as an ablation row, rather than
spinning up a parallel slot.
