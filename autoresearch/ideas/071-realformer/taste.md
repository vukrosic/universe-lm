# Taste log — 071 realformer

## r1 — 2026-06-11 — verdict: accept
- Sharp bet: pre-softmax score residual across adjacent layers, `scores_l += scores_{l-1}`, zero-init cache → step 0 = baseline. One-sentence mechanism with a clean A/B.
- Not a dupe: 021-value-residual is a residual on **values** (post-attn), RealFormer is a residual on **pre-softmax scores**. Different surface, different prior — complementary, not crowded. Not in `closed.md`.
- Leverage at tiny1m3m: 6L means 5 carryover transitions; small but non-zero. Mechanism does not depend on depth or scale — it imposes a soft cross-layer attention-pattern prior, which should fire even shallow (paper's BERT-base = 12L, but gain doesn't require 12L to register at all).
- Information value either way: a WIN says the tiny model wants attention state to persist; a NULL says recompute-from-scratch is fine at extreme depth/data limits. Both feed the 135M recipe call on whether to keep the residual-attention family.
- Transfer: published wins on BERT/ETC pretraining + many downstream NLP tasks. Identity-init makes it a no-regret carry to 135M if it lands here. transfer-risk: med is reasonable (could shrink at small L), not high.
- Niche fit: pure mechanism, zero-init identity, no extra params, no infra need. Fits parameter-golf-tier cleanly.
