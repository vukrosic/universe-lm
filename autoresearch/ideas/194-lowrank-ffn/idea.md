---
id: 194-lowrank-ffn
status: reviewing
round: 1
updated: 2026-06-15T08:31:54Z
transfer-risk: med
plain: Move the rank-r residual-correction lever off the closed FFN axis and onto W_V (the value projection), so the same mechanism is tested on a sub-block where V is special (021-vres WIN) and the W_V matrix hasn't been probed for low-rank structure at 0.94M. Pre-register: a null closes the entire low-rank-residual sub-block family at this tier.
---

# 194 — Low-Rank W_V Residual Correction (Re-pitched from FFN Bottleneck)

## Source
- 194-r1 (closed at taste gate, FFN axis exhausted: 146, 153, 157, 158, 170 + 117/118/145 MoE sub-axis) — original FFN low-rank pitch.
- **207-wo-lowrank-bottleneck** (in-repo, needs-taste r1) — *same mechanism* on W_O. 194-r2 pivots to W_V so we don't ship a duplicate.
- **197-tied-wo-across-blocks** (in-repo, needs-repitch r1) — W_O sharing across blocks. Different axis (sharing, not rank).
- **199-spectral-attn-output** (in-repo, needs-repitch r1) — W_O Lipschitz regularization. Different axis (Lipschitz, not rank).
- Arora et al., "Linear Algebraic Structure of Word Senses" + follow-up transformer-rank analyses — observations that attention/FFN matrices are approximately low-rank at scale.
- Hu et al., "LoRA" (arXiv:2106.09685, 2021) — learned low-rank factorization as residual correction; W_V-low-rank at init-α=0 is the training-from-scratch analog.
- **Closed priors on V-side attention (key context):**
  - 021-value-residual (WIN Δ=−0.034) — V *cross-block* residual binds at 0.94M; **V is special**.
  - 184-v-carry-block (rejected 3 recode rounds, 2026-06-15) — V cross-block re-attempt failed (cross-block attention path structurally hostile at d_model=64).
  - 164-q-carry (null Δ=+0.0360) — Q-side cross-block *doesn't* bind; V-bind is not symmetric to Q.
  - 151-rov-gated (null Δ=+0.0114) — intra-V rotary gated nulls.
  - 176-v-pre-av-norm (null Δ=+0.0303) — pre-AV V normalization nulls.
  - 016-qk_norm (WIN) — pre-softmax QK *magnitude* axis binds; orthogonal to W_V *rank* axis.
  - 160-rms-gain-per-head (null) — post-AV *magnitude* axis on attention output nulls; not a rank axis.

**No W_V low-rank correction has been tested at 0.94M** — none of the closed V-side levers probe the rank structure of the W_V projection itself.

## Mechanism
Standard attention: `out = (softmax(QK^T/√d_k) ⊙ V) · W_O`, where V is computed per head as `V = x · W_V` with W_V ∈ R^{d_model × d_model} (here 64×64 = 4096 params per block × 12 = 49,152 W_V params total).

Low-rank W_V correction: add a learned rank-r residual to W_V:
```
W_V_eff = W_V + α · W_V_A @ W_V_B
W_V_A ∈ R^{d_model × r}, W_V_B ∈ R^{r × d_model}, r = 8
α = sigmoid(α_raw), init α_raw = -10  ⇒ α ≈ 4.5e-5 at step 0
```
At step 0: W_V_eff = W_V exactly (bit-identical to baseline — α contributes ~zero).

The optimizer can grow α during training to exploit low-rank structure in W_V; if W_V is approximately full-rank, the optimizer can leave α near zero and the residual is silent.

## Design sketch
- **File**: `models/layers.py` (attention module) — add an optional rank-r residual correction to W_V per block.
- **Config flag**: `use_lowrank_wv: bool = False`, `wv_rank: int = 8`, `wv_alpha_raw_init: float = -10.0`.
- **Compute**: per block, `α = sigmoid(α_raw)`. `W_V_eff = W_V + α · W_V_A @ W_V_B`. `V = x @ W_V_eff`. (Apply **before** reshape into heads so the correction is per-block, not per-head.)
- **Bit-identical at step 0**: sigmoid(-10) ≈ 4.5e-5 ⇒ `α · W_V_A @ W_V_B ≈ 0` ⇒ `W_V_eff ≈ W_V` to fp32 precision. The cap on `|α · W_V_A @ W_V_B|` is well below 1e-6 at the standard W_V init scale (σ ≈ 1/√64 ≈ 0.125).
- **Params**: 2 × (d_model · r + r · d_model) × 12 blocks = 2 × (64·8 + 8·64) × 12 = 12,288 params (+1.3% of 0.94M); plus 12 α scalars. Silent at step 0.
- **Why W_V and not W_O**: W_O is owned by **207-wo-lowrank-bottleneck** (in-repo, same mechanism, needs-taste r1) — a separate A/B on the same sub-block is the duplicate we explicitly want to avoid. W_V is the natural counterpart of W_O in the attention path (V is the "value" projection that gets attended-to; W_O is the "output" projection that writes to the residual stream). V binds at 0.94M (021-vres WIN), so a low-rank axis on V is the highest-prior test.
- **Why not W_Q / W_K**: 016-qk_norm (joint, WIN) shows pre-softmax QK magnitude binds, but 162 (Q-only norm, null) and 165 (K-only norm, null) show single-side norm doesn't — V is the only single-side attention projection that *positively* binds at 0.94M, so V is the cleanest placement for the rank axis.

## Scale evidence
LoRA at 7B–65B (residual low-rank, Hu et al. 2021); transformer weight-rank analyses at 7B+ (LLM.int8(), Frantar et al. 2022 — Q/K/V/O matrices show effective rank 30–60% of nominal at 7B). No published *training-from-scratch* W_V low-rank correction win at <100M that I'm aware of. Transfer-risk: med (lever is well-defined, novel placement, identity/zero-init-able).

## Why it's worth a slot (RE-PITCHED, round 2)
The r1 FFN-side pitch was killed because the FFN axis is closed-out at 0.94M (6+ nulls: 146, 153, 157, 158, 170 + 117/118/145 MoE sub-axis). The mechanism (rank-r residual correction, init at α=0, bit-identical step 0) is correct and the engineering is clean — **the placement is the problem**.

**This re-pitch moves the same lever to W_V**, where:
- 021-vres (WIN) establishes V as special at 0.94M (Q-side analog 164-q-carry nulls — V-bind is not symmetric).
- 184-v-carry-block (rejected 3x) shows V *cross-block* is structurally hostile — but a *within-block W_V rank correction* is a different shape (parameter-space, not graph-space).
- 016-qk_norm (WIN) shows pre-softmax attention input binds; 160-rms-gain-per-head (null) shows post-AV output doesn't — **the W_V rank axis is orthogonal to both** (within-block parameter rank, not pre-softmax magnitude, not post-AV magnitude).
- 207-wo-lowrank-bottleneck owns W_O; 194-r2 on W_V is the *complementary* axis on the *other* major d_model × d_model attention sub-block.

**The bet, sharp.** At 0.94M/12L/4H, the W_V ∈ R^{64,64} projection maps d_model → d_model (one head's V contribution). At d_k=16, H=4, the per-head V is 16-dim; aggregate V is 64-dim. If head-V outputs are redundant (which they often are at d_k=16, H=4), W_V has effective rank < 32. A rank-8 correction (init α=0) gives the optimizer a knob to exploit that structure without disturbing the main projection.

**Pre-registered test (falsification framing).** Run a fresh ctrl to end of training, compute `effective_rank(W_V)` (ratio of L1 to L∞ singular values, or sum-of-singular-values²). Then:
- If `effective_rank(W_V) < 32` (>50% of singular values are near-zero) → the rank-8 correction should win (Δval < −0.005); optimizer activates the rank-r path.
- If `effective_rank(W_V) ≥ 56` (W_V is full-rank) → the rank-8 correction should null; a clean null **closes the entire low-rank-residual sub-block family at 0.94M** (FFN tested, W_O tested in 207, W_V tested here — all three sub-blocks).
- A win (Δ < −0.01) means V has exploitable rank structure that the FFN doesn't, distinct from the closed 016/160/021/184/151/176 V-adjacent nulls.

**Leverage in one sentence.** W_V at 0.94M is the natural counterpart of W_down in FFN (a d_model × d_model projection), and V is the only single-side attention projection that *positively* binds at this tier (021-vres WIN), so a rank-r residual on W_V is the highest-prior untested rank axis — and the only one whose null is *informative* (closes the family) rather than *redundant* (already tested on W_O by 207).

A null result here is the **terminal axis-closure test for low-rank-residual sub-block corrections at 0.94M**. A win opens a new axis orthogonal to the closed V-side levers.
