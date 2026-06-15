---
id: 190-w0-wv-tied
status: rejected
round: 2
updated: 2026-06-15T12:21:10Z
transfer-risk: low
plain: Share the attention value and output projections by using a single [d_model, d_model] matrix M, with the V slice reading only the first kv_size columns (GQA-aware); cuts ~0.22% of params and enforces a soft symmetry on the attention round-trip.
---

# 190 — Tied W_O and W_V (Attention Output/Value Weight Sharing)

## Source
- Press et al., "T5" (JMLR 2020, arXiv:1910.10683) — T5 shares the input embedding and the LM head (`tied=True`). The output projection `W_O` and the value projection `W_V` are *not* tied in T5, but the conceptual analog of tying two projections that are semantically related is well-established.
- Dehghani et al., "Universal Transformers" (ICLR 2019, arXiv:1807.03819) — shares weights across transformer blocks. Different mechanism (block-level vs projection-level), but the *philosophy* of weight sharing to enforce inductive bias is the same.
- ALBERT (Lan et al. 2019, arXiv:1909.11942) — *cross-layer parameter sharing* in the attention and FFN sub-blocks. The paper shows that sharing W_O across layers (or tying W_O to W_V) is a recognized regularization axis. ALBERT-base shares FFN and attention params across 12 layers; the cross-layer sharing axis was validated at 12M-235M on GLUE / SQuAD.
- Tied QK (closed) — the closed tied-QK lever is similar in spirit (share two projections of one attention head) but is closed at our tier because the *QK identity* is a strong constraint that prevents the head from learning different Q vs K roles. Tying W_O to W_V is *less* constraining: W_V is per-head input projection, W_O is the full-head output projection. They are not "the same role" in the attention mechanism, so tying them is a *soft* constraint that forces the two projections to share representation.
- In-repo context: 016-qk-norm, 175-alibi, 154-rebased are recent WINs. No prior lever tests *parameter sharing between W_O and W_V*. The closest in-repo lever is "tied QK" (closed in screen20m + tiny1m), but tying W_O to W_V is mathematically different (different projection shapes, different role in the attention forward).

## Mechanism (GQA-aware)
Standard attention with GQA (`n_kv_heads=2`, `n_heads=4`, `d_k=16`, `d_model=64`, so `kv_size=32`):
```
V   = W_V @ x          # W_V: [kv_size, d_model] = [32, 64]  (PyTorch linear-weight convention)
out = softmax(QK^T / sqrt(d_k)) @ V   # [B, T, kv_size]
head_out = out @ W_O   # W_O: [d_model, d_model] = [64, 64]
```

**Chosen variant — shared M (GQA-aware).** Define a single matrix
`M ∈ R^{d_model × d_model} = R^{64 × 64}`. Use it for the W_O projection directly
(`F.linear(av, M)`) and use its first `kv_size` columns as the W_V projection
(`F.linear(x, M[:, :kv_size])`). The V-side is the left sub-matrix of M, the
O-side is the whole of M. Under the hood this is equivalent to the miner's
"transpose-tied" form `W_O = W_V^T` when `n_kv_heads == n_heads` (square
attention); for GQA the natural generalization is a single square M whose
leftmost `kv_size` columns drive V and whose full body drives O. The
`d_model × (d_model − kv_size) = 64 × 32 = 2048` "extra" parameters of M
exist only to keep W_O square; they are not directly trained through the V
gradient, so the lever behaves like a *soft* symmetry between the V-in and
O-out projections of attention rather than a strict algebraic tying.

**Why not the miner's first-draft variants.** At tiny1m3m the actual shapes
are `W_V = [32, 64]` and `W_O = [64, 64]`. Neither direct-tied
(`W_O = W_V → [64, 64] vs [32, 64]`, mismatch on dim 0) nor transpose-tied
(`W_O = W_V^T → [64, 64] vs [64, 32]`, mismatch on dim 1) fits without
introducing a non-square O projection, which would break the rest of the
stack. The shared-M design is the natural GQA fix.

**Intuition (reframed for GQA).** The lever is *not* a strict norm-preserving
operator: `W_V · W_V^T ∈ R^{32 × 32}` operates in the value space, not the
residual space, so the residual stream's magnitude is not directly bounded
by `W_V · W_V^T`. The cleaner story is the **ALBERT/T5 regularization**
intuition: tying two semantically related projections is a *soft symmetry
constraint* on the attention round-trip, which reduces the model's
effective DOF and acts as an implicit regularizer. The same linear
operator maps residual stream → value space and aggregated value space →
residual stream (the V side is just a low-rank sub-matrix of M), so the
two projections cannot drift arbitrarily far apart. Don't oversell
norm-preservation; the bet is "shared params → smaller effective DOF →
mild regularization, validated by ALBERT at 12M+ and T5 at 60M+."

**Step-0 byte-identity for the GQA variant.** To keep the construction-time
RNG consumption aligned with the un-tied baseline: at construction, after
the standard `nn.init.normal_(qkvo_proj, std=0.02)` runs, *replace* the V
slice of `qkvo_proj` with `M[:, :kv_size].clone()` and the O slice with
`M.clone()`. The shape and total parameter count of the model are
unchanged (the V/O slots of `qkvo_proj` are still `[32, 64]` and `[64, 64]`
respectively; the tying just makes them point at / mirror the same M
values), but the *byte values* of V and O at step 0 are now correlated
(one matrix, used twice). The un-tied model has two independent random
draws; the tied model has one draw used twice → "in distribution" equivalent,
not bit-identical. This is the same category as 175-alibi (slope-init 0
produces a non-zero forward) and 016-qk-norm (the tied init is a
constraint, not byte-identity); the spec accepts "step-0 ≈ baseline" for
this class.

## Design sketch
- **Files**:
  - `models/layers.py` (or `models/llm.py`) — in `MultiHeadAttention.__init__`,
    add `use_tied_wo_wv: bool = False` config flag. When `True`, register a
    single `self.wv_wo = nn.Parameter(torch.empty(d_model, d_model))`,
    initialized with `nn.init.normal_(std=0.02)`. The forward path:
    - `V = F.linear(x, self.wv_wo[:, :kv_size].T)` (or equivalent — slice
      the in-features for F.linear's `[out, in]` convention)
    - `O = F.linear(attn_out, self.wv_wo)` (full matrix)
    On the un-tied path the V and O slices of `qkvo_proj` keep their
    standard `nn.init.normal_(std=0.02)` init and are read as before.
  - `configs/llm_config.py` — add `use_tied_wo_wv: bool = False` to
    `LLMConfig`. Add `Tiny1M3MTiedWOConfig(Tiny1M3MConfig)` with
    `use_tied_wo_wv: bool = True`. Mirrors the existing `use_tied_qk`
    pattern (configs/llm_config.py:849, models/layers.py:1596-1644).
- **Config flag**: `use_tied_wo_wv: bool = False`.
- **Param count**: net change is `−(d_model × kv_size) = −(64 × 32) = −2048`
  params, since the V slice of `qkvo_proj` is replaced by a tied sub-matrix
  of M and the O slice uses M directly. **−0.22% of 0.94M.** (The miner's
  first-draft number of −0.43% assumed MHA `d_k·H = d_model = 64` and
  counted a full `[64, 64]` saving; with GQA `d_k·H = 64` but `kv_size = 32`,
  the correct saving is `d_model × kv_size = 64 × 32 = 2048`.)
- **Intuition (why it might lower val loss)**: tying W_V and W_O via a
  shared M forces the two semantically related projections (value-in and
  value-out of the attention block) to live in a 1-DOF sub-space rather
  than two independent DOF sub-spaces. This is a soft regularization axis
  validated at 12M-235M by ALBERT (cross-layer sharing) and at 60M-11B by
  T5 (tied embeddings). At 0.94M the regularization is mild (the model is
  small enough that a few-k-param reduction is meaningful) and the bet is
  that the soft symmetry is a regularizer without being so constraining
  that it cripples learning (unlike tied QK, which forced a *role*
  collision between Q and K).
- **Why it might bind at 0.94M where tied-QK didn't**: tied-QK (closed)
  forced the Q and K projections to be the same, which prevented the
  model from learning the *direction of attention* (Q) and the *key to
  attend to* (K) as separate concepts. Tied W_O/W_V does not prevent any
  conceptual separation — W_V projects *into* the per-head value space,
  W_O projects *out of* it; tying them just forces the two projections to
  share a single square matrix, which is a *symmetry* constraint rather
  than a *role* constraint.

## Scale evidence
- ALBERT (Lan et al. 2019, arXiv:1909.11942) — cross-layer parameter
  sharing (W_O shared across layers, W_V shared across layers). Validated
  at 12M-235M on GLUE, SQuAD, RACE.
- T5 tied embeddings — validated at 60M-11B on SuperGLUE, etc.
  Tied-embedding is the standard tied-projection analog.
- **Transfer-risk: low** — the lever is a recognized parameter-sharing
  axis with direct validation at 12M+ (ALBERT) and 60M+ (T5 tied
  embeddings). The specific W_O ↔ W_V tying is less well-studied than
  tied embeddings, but the *philosophy* (sharing two semantically related
  projections) is well-validated.

## Why it's worth a slot
The bet, in one sharp sentence: **tied W_O/W_V is a 0.22% param reduction
that enforces a soft symmetry on the attention round-trip (one shared
[d_model, d_model] matrix M, with V reading only the first kv_size
columns), and ALBERT shows that *cross-layer* parameter sharing helps at
12M-235M** — the closed tied-QK lever was a *stronger* constraint (QK
identity, which is closed because the *role* of Q vs K must differ), but
W_O ↔ W_V tying is a *softer* constraint that just forces the *round-trip*
projection to share a single matrix; a null at 0.94M would tell us that
even soft tying on the attention path is too constraining at this tier,
and a win would give a meaningful (2k param) efficiency-accuracy trade-off.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40; pinned
  baseline val_mean 6.2403 (box 619cf8059d37), current box val_mean 6.3988
  with noise_band 0.04 (box 5b8a7fea8963).
- **Absolute distance to cache is NOT the binding signal.** The 0.04
  noise_band on the current box is wider than the WIN bar (−0.005), so
  the WIN is gated by the per-ctrl delta plus the two-ctrl rule, not by
  raw distance to the 6.40 cache number. The runner should not argue
  with the cache in evidence.md; the pass/fail bar is the per-ctrl delta
  + two-ctrl rule, both of which the plan already pins.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- Tied QK (closed) — different projection, different role. Tying W_O/W_V
  is a *softer* constraint (symmetric round-trip, not QK identity).
- ALBERT-style cross-layer sharing — different axis (cross-layer vs
  within-block).
- Tied embedding / tied LM head (T5) — different projections
  (input/output vs W_O/W_V). T5-style tied embedding is the *standard*
  decoder config (the 0.94M baseline already uses tied embeddings).
- 021-value-residual (WIN), 164-q-carry (null), 186-v-carry (closed) —
  cross-block V carries, not W_V/W_O tying. Different mechanism.
