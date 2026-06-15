---
id: 190-w0-wv-tied
status: tasting
round: 1
updated: 2026-06-15T08:23:42Z
transfer-risk: low
plain: Share the weights between the attention "value" projection and the "output" projection, like tying the input and output embeddings — it cuts ~10% of attention params and forces the two projections to share representation.
---

# 190 — Tied W_O and W_V (Attention Output/Value Weight Sharing)

## Source
- Press et al., "T5" (JMLR 2020, arXiv:1910.10683) — T5 shares the input embedding and the LM head (`tied=True`). The output projection `W_O` and the value projection `W_V` are *not* tied in T5, but the conceptual analog of tying two projections that are semantically related is well-established.
- Dehghani et al., "Universal Transformers" (ICLR 2019, arXiv:1807.03819) — shares weights across transformer blocks. Different mechanism (block-level vs projection-level), but the *philosophy* of weight sharing to enforce inductive bias is the same.
- ALBERT (Lan et al. 2019, arXiv:1909.11942) — *cross-layer parameter sharing* in the attention and FFN sub-blocks. The paper shows that sharing W_O across layers (or tying W_O to W_V) is a recognized regularization axis. ALBERT-base shares FFN and attention params across 12 layers; the cross-layer sharing axis was validated at 12M-235M on GLUE / SQuAD.
- Tied QK (closed) — the closed tied-QK lever is similar in spirit (share two projections of one attention head) but is closed at our tier because the *QK identity* is a strong constraint that prevents the head from learning different Q vs K roles. Tying W_O to W_V is *less* constraining: W_V is per-head input projection, W_O is the full-head output projection. They are not "the same role" in the attention mechanism, so tying them is a *soft* constraint that forces the two projections to agree on a shared representation subspace.
- In-repo context: 016-qk-norm, 175-alibi, 154-rebased are recent WINs. No prior lever tests *parameter sharing between W_O and W_V*. The closest in-repo lever is "tied QK" (closed in screen20m + tiny1m), but tying W_O to W_V is mathematically different (different projection shapes, different role in the attention forward).

## Mechanism
Standard attention:
```
V = W_V @ x          # W_V: [d_model, d_model], shared across heads (or [H, d_model, d_k] for per-head)
out = softmax(QK^T / sqrt(d_k)) @ V   # [B, T, d_k]
head_out = out @ W_O   # W_O: [d_model, d_model]
```
The full attention output is `(softmax(QK^T) @ V) @ W_O`. With the substitution `V = W_V @ x`, the full output is `softmax(QK^T) @ W_V @ x @ W_O`. Tying W_O = W_V^T means the output is `softmax(QK^T) @ W_V @ x @ W_V^T`, which is a symmetric bilinear form on the value vectors.

**Two variants**:
1. **W_O = W_V^T** (transpose-tied): the output projection is the *transpose* of the value projection. Shapes: W_V is `[d_model, d_k * H]`, W_O^T is `[d_model, d_k * H]`. This makes the value projection and the output projection transpose of each other; the constraint forces the projection to be (approximately) an orthogonal rotation in the subspace it spans.
2. **W_O = W_V** (direct-tied): W_O and W_V are the same matrix. Shapes: requires `d_k * H = d_model`, which holds for square attention (4 heads × d_k=16 = 64 = d_model). The direct-tied form is more constraining and only works for "square" attention. For tiny1m3m (H=4, d_k=16, d_model=64) the shapes are exactly compatible.

**Step-0 byte-identity**: with W_O and W_V initialized by the same `_init_weights` (the global init in the repo), tying them is just a constraint on the init. If both start at the same random matrix, the tied init is bit-identical to the un-tied init *in distribution* but not in actual values (the un-tied case has two independent random matrices; the tied case has one matrix used twice). To get bit-identity at step 0, the constraint must be that the un-tied model also has W_O = W_V initially, which is not the case for standard init.

The lever is *not* step-0 byte-identical in either variant. It's an architectural change at construction time. The "step-0 ≈ baseline" property is: at step 0, the tied model has the *same number of effective DOF* as the un-tied model (just one projection instead of two), and the gradient flow is different. The training trajectory is materially different; step-0 forward is similar in distribution but not in exact values.

**Why this is OK**: the spec allows "step-0 ≈ baseline" for non-strict-bit-identity levers (e.g., 175-alibi's slope init of 0 produces a non-zero attention logit at step 0; the lever is "step-0 ≈ baseline" rather than strict bit-identity). 190 falls in the same category.

## Design sketch
- **Files**:
  - `models/layers.py` (or `models/llm.py`) — in `MinimalLLM.__init__`, add `use_tied_wo_wv: bool = False` config flag. When `True`, register a single `self.wv_wo = nn.Linear(d_model, d_k * H, bias=False)` and use it for both the value projection and the output projection. The forward path computes `V = self.wv_wo(x)`, then after the attention product, computes `out = attn_out @ self.wv_wo.weight.T` (using the same weight matrix, transposed).
  - `configs/llm_config.py` — add `use_tied_wo_wv: bool = False` to `LLMConfig`. Add `Tiny1M3MTiedWOConfig(Tiny1M3MConfig)` with `use_tied_wo_wv: bool = True`.
- **Config flag**: `use_tied_wo_wv: bool = False`.
- **Param count**: removes `[d_model × d_k × H] = [64 × 16 × 4] = 4096` params (one W_V or W_O) and uses it for the other. Net: **−4096 params (−0.43% of 0.94M)**. Slight param reduction.
- **Intuition (why it might lower val loss)**: the W_O and W_V projections together form a `(d_model, d_k × H)` → `(d_model, d_k × H)` shape — they project the residual stream *up* to the per-head value space (W_V) and then *back down* to the residual stream (W_O). Tying them makes the round-trip projection a *symmetric* operator (`W_V · W_V^T`), which preserves the *norm* of the projected vector (W_V is approximately orthogonal, so ||W_V · x|| ≈ ||x||). This norm-preservation property means the attention output's magnitude is well-bounded regardless of the attention weights, which can stabilize the residual stream's growth. At 0.94M, this stability may help the gradient flow.
- **Why it might bind at 0.94M where tied-QK didn't**: tied-QK (closed) forced the Q and K projections to be the same, which prevented the model from learning the *direction of attention* (Q) and the *key to attend to* (K) as separate concepts. Tied W_O/W_V does not prevent any conceptual separation — W_V projects *into* the per-head value space, W_O projects *out of* it; tying them just forces the two projections to be the same matrix, which is a *symmetry* constraint rather than a *role* constraint.

## Scale evidence
- ALBERT (Lan et al. 2019, arXiv:1909.11942) — cross-layer parameter sharing (W_O shared across layers, W_V shared across layers). Validated at 12M-235M on GLUE, SQuAD, RACE.
- T5 tied embeddings — validated at 60M-11B on SuperGLUE, etc. Tied-embedding is the standard tied-projection analog.
- **Transfer-risk: low** — the lever is a recognized parameter-sharing axis with direct validation at 12M+ (ALBERT) and 60M+ (T5 tied embeddings). The specific W_O ↔ W_V tying is less well-studied than tied embeddings, but the *philosophy* (sharing two semantically related projections) is well-validated.

## Why it's worth a slot
The bet, in one sharp sentence: **tied W_O/W_V is a 0.43% param reduction that enforces a norm-preserving symmetry on the attention round-trip projection, and ALBERT shows that *cross-layer* parameter sharing helps at 12M-235M** — the closed tied-QK lever was a *stronger* constraint (QK identity, which is closed because the *role* of Q vs K must differ), but W_O ↔ W_V tying is a *softer* constraint that just forces the *round-trip* projection to be symmetric; a null at 0.94M would tell us that even soft tying on the attention path is too constraining at this tier, and a win would give a meaningful (4k param) efficiency-accuracy trade-off.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- Tied QK (closed) — different projection, different role. Tying W_O/W_V is a *softer* constraint (symmetric round-trip, not QK identity).
- ALBERT-style cross-layer sharing — different axis (cross-layer vs within-block).
- Tied embedding / tied LM head (T5) — different projections (input/output vs W_O/W_V). T5-style tied embedding is the *standard* decoder config (the 0.94M baseline already uses tied embeddings).
- 021-value-residual (WIN), 164-q-carry (null), 186-v-carry (closed) — cross-block V carries, not W_V/W_O tying. Different mechanism.
