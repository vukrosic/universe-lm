# 190 — Tied W_O / W_V (review log)

## r1 — 2026-06-15 — verdict: revise

The mechanism is sound, the sources are real, and the closed-axis defense
holds — but the design sketch has a **GQA shape-compatibility bug** that
breaks the lever as written. Concretely fixable; not a reject.

### Findings

- **GQA shape-compatibility bug (BLOCKER, must fix).** The miner's
  design assumes MHA: claims `W_V` is `[d_model, d_k * H] = [64, 64]`
  and "the shapes are exactly compatible" for the direct-tied variant
  `W_O = W_V`. But `Tiny1M3MConfig` (configs/llm_config.py:2257-2261)
  is **GQA**: `n_kv_heads=2, n_heads=4, d_k=16, d_model=64`. The actual
  shapes (PyTorch linear-weight convention, [out_features, in_features])
  are `W_V = [kv_size, d_model] = [32, 64]` and
  `W_O = [d_model, d_model] = [64, 64]`. Neither variant as written
  fits:
  - **direct-tied** `W_O = W_V` → `[64, 64]` vs `[32, 64]`, mismatch.
  - **transpose-tied** `W_O = W_V^T` → `[64, 64]` vs `[64, 32]`, mismatch.
  Fix: define a **GQA-aware shared matrix** `M ∈ R^{d_model × d_model}`.
  Use `M` as the W_O linear weight directly (`F.linear(av, M)`) and
  `M[:, :kv_size]` as the W_V linear weight (`F.linear(x, M[:, :kv_size])`).
  This is the natural GQA generalization of the "transpose-tied"
  interpretation: the W_V slice is the first `kv_size` columns of the
  shared `M`, and the rest of `M`'s columns are unused for V (they
  exist only so W_O has a square [d_model, d_model] matrix to project
  through). Add a short comment noting that this collapses to the
  miner's "W_O = W_V^T" form when `n_kv_heads == n_heads`.

- **Param-savings estimate is wrong for GQA.** The miner says the lever
  saves `[d_model × d_k × H] = [64 × 16 × 4] = 4096` params (−0.43%).
  With GQA, the actual saving is `d_model × kv_size = 64 × 32 = 2048`
  params, i.e. **−0.22% of 0.94M**. Update the number in `## Design
  sketch`, the `## Mechanism` parenthetical, and the
  `## Why it's worth a slot` sentence ("0.43% param reduction" →
  "0.22% param reduction"). The 0.22% number is still a real
  efficiency-accuracy trade-off; the bet does not die.

- **"Norm-preserving" intuition needs restating for GQA.** The miner's
  argument is that `W_V · W_V^T` is a symmetric operator on
  R^{d_model}, preserving the projected vector's norm. With GQA, the
  shared `M` is [64, 64] but `W_V = M[:, :32]` projects to a
  32-dim value space, so `W_V · W_V^T ∈ R^{32 × 32}` — it preserves
  the norm in the value space, not the residual space. The
  norm-bounded-attention-output argument therefore does NOT directly
  bound the residual stream's magnitude. **Reframe** the mechanism
  intuition as: the lever enforces a learned symmetry between the
  value-in and value-out projections of attention (one shared
  [d_model, d_model] matrix, with V reading only the first
  `kv_size` columns). The intuition that survives is "the same linear
  operator maps residual stream → value space and aggregated value
  space → residual stream"; this is a *soft* symmetry constraint
  (the V-side is a low-rank sub-matrix of M, not a full inverse), not
  strict norm-preservation. Don't oversell the norm-preservation
  property; the ALBERT/T5 *regularization* intuition (shared params
  → smaller effective DOF → implicit regularization) is the cleaner
  defense.

- **Cache reference should pin the current box's noise band.** The
  plan says "champion val ≈ 6.24, cache baseline 6.40" — that matches
  the pinned baseline (val_mean 6.2403, box 619cf8059d37) and the
  current box (val_mean 6.3988, noise_band 0.04, box 5b8a7fea8963).
  The 0.04 noise_band is wider than the WIN bar (`-0.005`), so the
  WIN is gated by the per-ctrl delta and the two-ctrl rule, not by
  absolute distance to the cache. The plan already has the right
  numbers (`WIN: trt_val ≤ ctrl_val − 0.005` with two-ctrl rule;
  `NULL: |Δ| < 0.01`) — just add a one-line note that the
  absolute-distance-to-cache is *not* the binding signal at this
  noise level. Minor, but the reviser should write it down so the
  runner doesn't argue with the cache in evidence.md.

- **Step-0 byte-identity for the GQA variant needs a one-liner.** The
  miner correctly notes the tied model is "in distribution" equivalent
  to the un-tied model at step 0 (the un-tied has two independent
  random matrices; the tied has one shared matrix used twice). For
  the GQA-aware design above, the analogous claim is: register `M`
  with the same `nn.init.normal_(M, mean=0, std=0.02)` init the
  baseline uses for the O slice of `qkvo_proj`, then *replace* the
  V slice of `qkvo_proj` with `M[:, :kv_size].clone()` at construction
  so the construction-time RNG consumption matches the un-tied
  baseline (otherwise the tied model's M is re-init'd and downstream
  slices shift). The 175-alibi precedent (slope-init 0 → non-zero
  forward, but spec accepts "step-0 ≈ baseline" as the same category)
  covers this; just add the construction-time clone detail to the
  plan.

- **Sources are real and current.** T5 (arXiv:1910.10683, JMLR 2020),
  Universal Transformers (arXiv:1807.03819, ICLR 2019), ALBERT
  (arXiv:1909.11942, 2019) all verify. The "ALBERT validates
  cross-layer parameter sharing at 12M-235M on GLUE/SQuAD/RACE"
  claim is accurate. The "T5 tied embeddings validated at 60M-11B"
  claim is accurate. No fabrication.

- **Not a duplicate of a closed lever.** closed.md lists "Tied QK
  (on best baseline)" as a closed axis (the miner correctly cites
  this as a different lever — QK role collision vs W_O↔W_V symmetry
  constraint). The 043-mla reject and the broader "MHA vs GQA" closed
  axis are unrelated. ALBERT-style cross-layer sharing is a closed
  twin (the miner correctly distinguishes it as a different axis:
  cross-layer vs within-block). Tied embedding is the *standard*
  tiny1m3m config, on by default; the lever is not tied-embedding.
  No dedup hit.

- **LoC budget is fine.** The structural change is one shared
  parameter (`M ∈ R^{d_model × d_model}`), one forward branch (when
  the flag is on, read M for both V and O), one config flag on
  `LLMConfig`, and one `Tiny1M3MTiedWOConfig(Tiny1M3MConfig)`
  subclass. Well under 200 LoC; matches the established `use_tied_qk`
  pattern (configs/llm_config.py:796, models/layers.py:1547-1549).

- **Transfer-risk: low is correct.** ALBERT at 12M-235M and T5 at
  60M-11B both validate the *philosophy* of tied projections; the
  W_O↔W_V specific instance is novel but the regularization axis
  is well-validated. Low is the right tag.

### Round

This is round 1; reviser has 2 rounds of budget remaining. The
GQA-shape fix + the param-savings correction + the norm-preservation
reframe are all mechanical — should be a single-round bounce.
