---
id: 204-cross-block-attn-score-share
status: needs-taste
round: 1
updated: 2026-06-15T09:00:00Z
transfer-risk: med
plain: Blend the attention scores from the previous block with the current block's scores using a learnable per-block scalar (init 0 so step-0 is byte-identical), so each block can softly re-use the previous block's attention pattern.
---

# 204 — Cross-Block Attention Score Sharing (Learnable Blend of Adjacent-Block Softmax Inputs)

## Source
- 021-value-residual (in-repo WIN Δ=−0.034) — V-side cross-block carry on residual. Different tensor (V vs scores).
- 168-av-output-carry (closed null) — AV carry across blocks. Different placement.
- 164-q-carry (closed null) — Q carry across blocks. Different tensor.
- 188-cross-block-kv-share (in-repo implementing) — K, V projection sharing across blocks. Different axis (projections vs scores).
- 186-v-carry-block (in-repo needs-run) — within-block V carry. Different axis (within-block).
- Sukhbaatar et al., "Memorizing Transformers" (ICLR 2022, arXiv:2203.08913) — cross-document memory via attention; the lever here is *within-model* cross-block score reuse.

## Mechanism
Standard attention: each block b computes its own `scores_b = Q_b · K_b^T / √d_k`, then `attn_b = softmax(scores_b)`.

Cross-block attention score sharing: each block's attention scores are blended with the previous block's:
```
scores_b = (1 − α_b) · scores_b_self + α_b · scores_{b-1}     # α_b init 0
attn_b = softmax(scores_b)
```
At init α_b = 0, scores_b = scores_b_self exactly (bit-identical baseline). As α_b grows, each block's attention pattern is a weighted blend of its own and the previous block's pattern.

The previous block's scores must be detached (no gradient flow through the previous block's score computation, only through the α parameter and the current block's Q, K).

## Design sketch
- **File**: `models/layers.py` — modify `attention_block` to optionally blend the previous block's scores.
- **Config flag**: `use_cross_block_score_share: bool = False`, `score_share_alpha_init: float = -10.0` (sigmoid ≈ 0).
- **Compute**: per block b, compute `α = sigmoid(α_raw_b)`. `scores_b_eff = (1 − α) · scores_b_self + α · prev_block_scores.detach()`.
- **Bit-identical at step 0**: α ≈ 0 ⇒ `scores_b_eff = scores_b_self` exactly.
- **Params**: 1 scalar per block × 12 blocks = 12 α scalars (+0.001% of 0.94M).
- **Intuition**: attention scores encode "where to look from each query position". Cross-block score sharing lets the model softly preserve the previous block's attention pattern — a *soft persistence* of attention across depth. Different from 021 (which mixes V across blocks via the residual) and 188 (which shares K, V *projections* across blocks); 204 mixes *scores* (the post-QK^T output) across blocks.

## Scale evidence
Memorizing Transformers validated at 0.3B-1.2B with retrieval augmentation. No published "cross-block attention score sharing" within-model paper I'm aware of. Transfer-risk: med (lever is a minor architectural change).

## Why it's worth a slot
**Pattern**: cross-block V-carrying (021) is a WIN; cross-block Q-carrying (164) is a NULL; cross-block AV-carrying (168) is a NULL; cross-block K-V-projection-sharing (188) is implementing. 204 is the *score* axis — yet another cross-block mixing axis. The bet: at 0.94M, the *attention pattern* (encoded in scores) is a key signal that should propagate across depth. A 204 WIN would mean cross-block attention-pattern propagation is a missing lever; a 204 NULL would mean the V-side carry is uniquely binding (the Q/AV/score axes are all dead ends).
