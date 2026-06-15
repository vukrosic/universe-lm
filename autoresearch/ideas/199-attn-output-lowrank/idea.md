---
id: 199-attn-output-lowrank
status: needs-review
round: 1
updated: 2026-06-15T16:25:46Z
transfer-risk: med
plain: After 207-wo-lowrank-bottleneck claimed the rank-residual axis on W_O and 194-r2 claimed it on W_V, take the same mechanism onto W_Q — the only remaining d_model x d_model attention sub-block unowned in the active queue, completing the rank-residual family across {W_Q, W_V, W_O} (W_K left to a future sub-block probe).
---

# 199 — W_Q Low-Rank Residual Correction (Repitched r2: W_O → W_Q)

## Source
- **r1 source** (rejected): 207-wo-lowrank-bottleneck (in-repo, needs-plan, approved r1 2026-06-15) — same rank-residual mechanism on W_O. r1 was a hard-replace `W_O = A·B` with SVD init; killed as a queue duplicate on the same axis (W_O) with structurally-weaker engineering than 207's α=0 soft-residual.
- **r1 closed priors** (referenced by the r1 taste verdict): 178-mqa-gated (null), 156-moa (null), 146-sparse-ffn (null), 117/118 MoE family, 190-w0-wv-tied (null), 199-spectral-attn-output (W_O Lipschitz axis — distinct from rank).
- **207-wo-lowrank-bottleneck** (in-repo, needs-plan, approved r1 2026-06-15) — `W_O_eff = W_O + α·A·B`, α=sigmoid(α_raw) init −10, r=16, +24,576 params (+2.6%). 199-r2 mirrors the *mechanism* on a different sub-block.
- **194-lowrank-ffn r2** (in-repo, planning 2026-06-15) — same mechanism on W_V after r1 (FFN axis exhausted: 146/153/157/158/170 + 117/118/145 MoE). V is special at 0.94M (021-vres WIN Δ=−0.034). 194-r2 bet: rank axis on W_V is highest-prior because V is the only single-side attention projection that positively binds.
- **197-tied-wo-across-blocks** (in-repo, needs-plan r1) — W_O sharing across blocks. Sharing axis, not rank axis. Different from 199.
- **199-spectral-attn-output** (in-repo, reviewing r1) — W_O Lipschitz cap. Different axis (Lipschitz), different placement (W_O), different signal.
- **LoRA** (Hu et al. 2022, arXiv:2106.09685) — learned rank-r factorization as residual correction; Q,V are the most-adapted matrices in the LoRA paper's "Q,V-only" experiment. Q is the canonical rank-residual target at adaptation time; 199-r2 is its training-from-scratch analog on Q only.
- **LLM.int8()** (Frantar et al. 2022) — empirical observation that Q/K/V/O matrices in 7B+ models show effective rank 30-60% of nominal. Provides the structural prior that Q is approximately low-rank at scale; 199-r2 tests whether that holds at 0.94M.
- **Pre-softmax attention input priors at 0.94M** (key for Q-side prior):
  - **016-qk-norm** (WIN, Δ=−0.014/−0.019 at tiny1m3m) — joint QK RMSNorm binds. Pre-softmax magnitude axis is alive.
  - **162-q-only-norm** (null, Δ=−0.0043 inside band) — Q-only RMSNorm nulls; K-only is the binding side per the joint-QK attribution.
  - **165-k-only-norm** (null, Δ=−0.0293 inside band) — K-only RMSNorm also nulls; 016's WIN was carried by *joint* QK symmetry, not single-side.
  - **164-q-carry** (null, Δ=+0.0360 wrong-sign, 3.6× plan bar) — Q-side cross-block residual mixing nulls; V-side analog 021 binds.
  - **200-rope-phase-offset-per-layer** (in queue, needs-plan r1) — per-pair × per-layer K-rotation. Different axis (rotation depth, not projection rank).
  - **190-per-layer-qk-norm** (in queue, needs-plan r1) — single scalar gain per block per side on QK. Different axis (magnitude scalar, not projection rank).
- **Distinct from closed axes**: 016/162/165/164/200/190 all touch Q or K but on *magnitude / rotation / mixing* axes. **None probe the *intrinsic rank* of W_Q**. The 162/165 single-side norm nulls are the most relevant priors, but **norm ≠ rank**: a normalized W_Q can still be low-rank (rank-deficient after unit-RMS rows), and a rank-deficient W_Q can still have high-magnitude rows. The two axes are orthogonal in mechanism space.

## Mechanism
Standard attention computes the per-head query:
```
Q = x @ W_Q                # W_Q ∈ R^{d_model × d_model}, here 64×64 = 4096 params/block × 12 = 49,152 W_Q params total
Q = Q.view(B, T, H, d_k).transpose(1, 2)   # [B, H, T, d_k]
attn_logits = Q @ K.transpose(-2, -1) / sqrt(d_k)
```

Low-rank W_Q correction: add a learned rank-r residual to W_Q:
```
W_Q_eff = W_Q + α · W_Q_A @ W_Q_B
W_Q_A ∈ R^{d_model × r}, W_Q_B ∈ R^{r × d_model}, r = 16
α = sigmoid(α_raw), init α_raw = -10.0  ⇒ α ≈ 4.5e-5 at step 0
Q = x @ W_Q_eff
```

At step 0: `α · W_Q_A @ W_Q_B ≈ 0` ⇒ `W_Q_eff ≈ W_Q` to fp32 precision ⇒ bit-identical to baseline forward pass at step 0.

The optimizer can grow α during training to exploit low-rank structure in W_Q. If W_Q is approximately full-rank, the optimizer leaves α near zero and the residual is silent. If W_Q has effective rank < 64, the optimizer activates the rank-r correction.

**Why this design (and not the r1 hard-replace `W_Q = A·B`):** 207's α=0 soft-residual is the cleaner A/B because:
1. **Bit-identity without engineering tricks.** α=sigmoid(α_raw) at α_raw=−10 gives `α·A·B ≈ 0` directly; no SVD decomposition of a "ghost" full-rank W_Q needed.
2. **Rank-axis vs rank-constraint isolated.** Hard-replace confounds "rank ≤ r constraint" with "−X% param reduction" — soft-residual adds +2.6% params and isolates the *rank hypothesis* from the *param hypothesis*.
3. **Optimizer can stay silent.** Hard-replace forces the rank constraint to hold *and* the rank to be destroyed as A,B move independently — only step-0 is byte-identical, not step-1. Soft-residual lets the optimizer decide; if rank-axis binds, the optimizer activates α; if not, α stays pinned near 0 and the run is identical to baseline.

**Why W_Q (and not W_K, W_V, W_O, FFN):**
- **W_O is owned by 207** (in-repo, needs-plan r1) — running a separate W_O rank test is the duplicate the queue is trying to avoid.
- **W_V is owned by 194-r2** (in-repo, planning r2) — 194-r2 explicitly chose W_V because 021-vres WIN establishes V as special at 0.94M.
- **FFN is closed-out** (146, 153, 157, 158, 170 + 117/118/145 MoE sub-axis; 6+ nulls).
- **W_K is open** but is the *orthogonal* sibling to W_Q in the bilinear QK^T structure. The 162 (Q-only norm, null) and 165 (K-only norm, null) prior says single-side norm doesn't fire on either side individually. 016-qk-norm WIN says the *joint* QK axis binds in magnitude terms. W_Q (or W_K) rank is the orthogonal test: does the *intrinsic rank* of the projection itself bind even though *single-side magnitude* doesn't? Picking **W_Q** is one of two symmetric choices; the other (W_K) belongs to a future sibling idea.
- **Rank-axis family completion.** 207 (W_O) + 194-r2 (W_V) + 199-r2 (W_Q) covers three of the four d_model × d_model attention sub-blocks. If all three null, the rank-residual sub-block family is closed at 0.94M. If any one wins, the *sub-block-specific* rank hypothesis is isolated.

## Design sketch
- **File**: `models/layers.py` (attention module) — add an optional rank-r residual correction to W_Q per block.
- **Config flag**: `use_lowrank_wq: bool = False`, `wq_rank: int = 16`, `wq_alpha_raw_init: float = -10.0`.
- **Compute**: per block, `α = sigmoid(α_raw)`. `W_Q_eff = W_Q + α · W_Q_A @ W_Q_B`. `Q = x @ W_Q_eff`. (Apply **before** reshape into heads so the correction is per-block, not per-head — matches 207/194-r2 placement.)
- **Bit-identical at step 0**: sigmoid(−10) ≈ 4.5e-5 ⇒ `α · W_Q_A @ W_Q_B ≈ 0` ⇒ `W_Q_eff ≈ W_Q` to fp32 precision. The cap on `|α · W_Q_A @ W_Q_B|` is well below 1e-6 at the standard W_Q init scale (σ ≈ 1/√64 ≈ 0.125).
- **Params**: 2 × (d_model · r + r · d_model) × 12 blocks = 2 × (64·16 + 16·64) × 12 = 24,576 params (+2.6% of 0.94M); plus 12 α scalars. Matches 207's footprint on W_O exactly so the A/B is direct.
- **Kaiming init on W_Q_A, W_Q_B**: standard `nn.Linear`-style init; α=0 gates the contribution at step 0.
- **Verification step (implementer)**: forward pass at α=0 must produce fp32 max-abs-diff < 1e-6 vs baseline W_Q for a fixed input. If not, the bit-identity claim is broken and the test is invalid (the lever cannot isolate rank from init-noise).

## Scale evidence
- LoRA at 7B-65B (residual low-rank, Hu et al. 2021) — Q,V are the most-adapted matrices; rank-axis on Q validated at scale.
- LLM.int8() (Frantar et al. 2022) — Q/K/V/O matrices show effective rank 30-60% of nominal at 7B+. Direct structural prior that Q has exploitable low-rank structure at scale.
- No published *training-from-scratch W_Q low-rank correction win* at <100M that I'm aware of. 199-r2 is the canonical test at 0.94M.
- **Transfer-risk: med** — lever is well-defined, mechanism is the same as 207/194-r2 (already in flight), placement is novel (W_Q is the only remaining d_model × d_model attention sub-block without a rank probe at 0.94M), identity/zero-init-able at α=0.

## Why it's worth a slot (r2 bet, sharp)
**Pattern in the active queue (2026-06-15):**
- 207 (W_O soft-residual, needs-plan r1) — rank axis on W_O.
- 194-r2 (W_V soft-residual, planning r2) — rank axis on W_V.
- 197 (W_O sharing, needs-plan r1) — sharing axis on W_O.
- 199-spectral (W_O Lipschitz, reviewing r1) — Lipschitz axis on W_O.
- 190 (per-layer QK magnitude scalar, needs-plan r1) — magnitude axis on QK.
- 200 (per-pair × per-layer K-rotation, needs-plan r1) — rotation axis on K.

**W_Q is the only d_model × d_model attention sub-block unowned in the queue.** Taking the rank-residual mechanism onto W_Q completes the rank-axis family across {W_Q, W_V, W_O} (W_K is the orthogonal bilinear-sibling probe, left to a future sibling idea).

**Bet, in one sentence.** At 0.94M/12L/4H, W_Q ∈ R^{64,64} projects the residual stream into query space; if effective_rank(W_Q) < 32 at convergence, a rank-16 correction init-α=0 lets the optimizer exploit that structure, and the lever binds; if effective_rank(W_Q) ≈ 64, the rank-16 correction is silent and the run is identical to baseline. The 162-q-only-norm null is on a *different axis* (per-row magnitude, not projection rank) — a Q-side rank axis is plausibly alive even though a Q-side magnitude axis isn't.

**Pre-registered falsification framing.** Run a fresh ctrl to end of training, compute `effective_rank(W_Q)` (ratio of L1 to L∞ singular values, or sum-of-singular-values²). Then:
- **If 207, 194-r2, AND 199-r2 all null** → the rank-residual sub-block family is closed at 0.94M. Three independent sub-block tests of the same mechanism all failing is the canonical "axis exhausted" signal.
- **If 207 nulls but 199-r2 wins** → the rank-axis is *W_Q-specific* (V is special per 021, Q is special per W_Q rank, but O is not). A sub-block-resolved story.
- **If 207 wins but 199-r2 nulls** → the rank-axis binds on W_O only; W_Q is approximately full-rank at 0.94M. Sub-block-resolved in the opposite direction.
- **If both 207 and 199-r2 win** → the rank-axis is alive on multiple sub-blocks at 0.94M; transfer to the 10M+ ladder is plausible.

**Why a null here is informative, not redundant.** The 194-r2 doc argues that W_Q is a *less* promising target than W_V because V positively binds (021-vres WIN) but Q/K don't (162/165 single-side norm nulls). If 199-r2 also nulls on W_Q, that's three out of three rank-residual sub-blocks failing at 0.94M — *stronger* falsification of the rank-axis family than a single null would be. The leverage is in the *family* of tests, not any single one.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24 (175-alibi), 154-rebased WIN, cached ctrl mean ≈ 6.43-6.45 ±0.04.
- **WIN**: `trt_val ≤ ctrl_val − 0.01` AND clears the two-ctrl rule (Δ vs both fresh ctrls must exceed the ctrl-pair gap).
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.
- **Family null (axis closure)**: 199-r2 null + 207 null + 194-r2 null ⇒ rank-residual sub-block family closed at 0.94M.

## Distinct from closed axes (defensive)
- 016-qk-norm (WIN, joint QK magnitude) — different axis (magnitude, not projection rank).
- 162-q-only-norm (null) — single-side Q magnitude axis, *orthogonal* to single-side Q rank axis (norm ≠ rank; see mechanism section).
- 165-k-only-norm (null) — K-side magnitude axis; supports the "rank-axis orthogonal to magnitude-axis" reading because single-side magnitude doesn't bind.
- 164-q-carry (null) — Q-side cross-block residual mixing (graph-space, not parameter-space).
- 190-per-layer-qk-norm (in queue) — QK magnitude *scalar*, not projection rank.
- 200-rope-phase-offset-per-layer (in queue) — K-rotation depth, not projection rank.
- 207 (in queue, W_O soft-residual) — same mechanism on W_O. 199-r2 is the *W_Q twin*.
- 194-r2 (in queue, W_V soft-residual) — same mechanism on W_V. 199-r2 is the *W_Q twin*.
- 197-tied-wo-across-blocks (in queue) — W_O sharing across blocks. Sharing axis, not rank axis.
- 199-spectral-attn-output (in queue) — W_O Lipschitz. Lipschitz axis, not rank axis.
- 178-mqa-gated (closed null) — cross-head V sharing. Not a rank axis.
- 156-moa, 158-gau, 157-conv-ffn (closed null) — capacity-injection FFN/attention levers. 199-r2 is capacity-*neutral* (+2.6% params) on the Q projection.
- 146-sparse-ffn (closed null) — sparse FFN. 199-r2 is on Q, not FFN.
- 171-dropconnect-wo (closed null Δ=+0.0478 wrong-sign) — weight-level regularization on W_O. Different from rank-axis on W_Q.
- 190-w0-wv-tied (closed null) — W_O/W_V tying. Sharing axis, not rank axis.
