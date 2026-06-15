---
id: 200-rope-phase-offset-per-layer
status: planning
round: 1
updated: 2026-06-15T12:11:39Z
transfer-risk: med
plain: Give each layer × each RoPE pair its own learnable static K-rotation angle (init 0 so step-0 is byte-identical, K-only application so QK^T can bind) — a depth-axis twin of 185-static-per-head-k-rotation, now with per-pair × per-layer granularity.
---

# 200 — Per-Layer × Per-Pair Static K-Rotation (RoPE-Parametrized Depth-Axis Rebase of K)

## Source
- 185-static-per-head-k-rotation (closed procedurally, r4 build-smoke failure — see "Engaging the priors" below) — per-head × per-pair static K-rotation, parameterized via RoPE-style 2D plane angles. 200 is the **per-layer** variant of the same lever family; 185 was never empirically tested.
- 154-rebased-attn (in-repo WIN Δ=−3.48) — fixed *random shared* rebase of K and V. Validates the orthogonal-rebase axis at 0.94M. The lever form "static orthogonal rebase of K" is real; the open question is whether the **learned per-head** (185) or **learned per-layer × per-pair** (200) axis binds.
- 172-per-head-rope-base (in-repo closed null Δ=+0.0109 wrong-sign tiny) — per-head *frequency* (multiplicative scale on RoPE's position-dependent angle). 200 is per-layer *phase* (constant offset, position-independent). Different mechanism — see engagement below.
- 175-alibi-slopes (in-repo WIN Δ=−0.1585) — per-head additive bias on attention scores. Establishes that *positional* shape per head binds at 0.94M; speaks to neither per-layer × per-pair (200) nor per-head frequency (172) directly.
- Su et al., "RoFormer / RoPE" (Neurocomputing 2024, arXiv:2104.09864) — the per-pair 2D rotation that parameterizes 200's per-plane angles.

## Mechanism (revised — fix for r1's no-op finding)

**R1 problem (acknowledged and fixed)**: the r1 mechanism applied a per-layer phase `φ_l` to RoPE's angle `θ_i(m) = m·base^(-2i/d_k) + φ_l` for **both** Q and K. Because `R(θ+φ) = R(φ)·R(θ)` and orthogonal `R` preserves the inner product on each pair, the same `φ` on both sides **cancels in QK^T exactly** — every pair's per-position dot product is unchanged from baseline. Per-pair `φ_l_i` (different per pair) is no better: same cancellation applies within each pair, and the cross-pair terms are absent because RoPE's pairs are disjoint. Net: the r1 lever was a guaranteed null, regardless of how many scalars were used.

**R2 mechanism**: apply the phase as a **K-only static rotation, parameterized per-layer × per-pair**, init 0. Q is left untouched; K is rotated by a block-diagonal `R_l ∈ R^{d_k × d_k}` whose 8 planes are 2D rotations with learnable angles `φ_{l,i}`:

```python
# Per pair i in d_k=16 (8 planes), per layer l:
K_pair_i → R(φ_{l,i}) @ K_pair_i            # 2D rotation on the (2i, 2i+1) plane
                                           # φ_{l,i} learnable scalar, init 0

# Q unchanged: standard RoPE on Q, K = R_l @ standard_RoPE(K).
# At init φ_{l,i}=0 ⇒ R_l = I ⇒ K unchanged ⇒ QK^T unchanged ⇒ loss unchanged.
```

Because the rotation is **applied to K only**, QK^T is no longer inner-product-preserved for that side: `<R(φ) K, Q>` ≠ `<K, Q>` in general, so the lever has a real axis to pull on. The orthogonal constraint is preserved (each 2D rotation is orthogonal; the block-diagonal product is orthogonal), so K's norm and the softmax temperature are unchanged.

**Why this isn't 185 (structurally)**: 185 is per-head × per-plane (H × d_k/2 = 4 × 8 = **32 angles per block**, 384 total — every head has its own 8-plane rotation in every layer). 200 is per-layer × per-plane (L × d_k/2 = 12 × 8 = **96 angles total, shared across heads** — one set of 8 plane angles per layer). The parameterization axes are orthogonal: 185 varies across heads within a layer; 200 varies across layers (depth axis). At 12L × 4H = 48 (block, head) cells, the two levers live in different cells of the (depth × head) grid.

**Per-pair vs per-scalar**: 200's per-pair (8 planes per layer) breaks the cross-pair orthogonal symmetry — the same K passed through different per-layer rotations per pair gives a K whose pairs have been rotated by *different* angles. W_O (a per-channel projection, applied post-attention) **cannot absorb** this: W_O mixes channels but doesn't know which pair was rotated by how much, and the rotation is applied pre-softmax where W_O has no reach. The rotational lever has a structurally different optimization surface from the scalar depth-axis nulls (see below).

**Cost**: 96 angles × 1 fp32 = 384 bytes params; +0.001% of 0.94M. Negligible compute: ~524K flops per block per forward (same shape as 185's). ~50 LoC across `models/layers.py`, `configs/llm_config.py`, `models/llm.py`.

## Design sketch
- **File**: `models/layers.py:MultiHeadAttention.__init__` — add `use_per_layer_k_rotation: bool = False` kwarg. Allocate `self.per_layer_k_rotation_angles = nn.Parameter(torch.zeros(d_k // 2))` (8 angles, init 0 ⇒ identity rotation; **shared across heads**, varying per-layer).
- **Forward**: in `MultiHeadAttention.forward`, after RoPE on K, apply the per-plane 2D rotation block:
  ```python
  if self.use_per_layer_k_rotation:
      cos_a = self.per_layer_k_rotation_angles.cos()  # [d_k//2]
      sin_a = self.per_layer_k_rotation_angles.sin()
      K_pairs = K.reshape(B, T, H, d_k//2, 2)
      K_a, K_b = K_pairs[..., 0], K_pairs[..., 1]
      K_pairs = torch.stack([K_a*cos_a - K_b*sin_a,
                              K_a*sin_a + K_b*cos_a], dim=-1)
      K = K_pairs.reshape(B, T, H, d_k)
  ```
- **Config**: `use_per_layer_k_rotation: bool = False` on `LLMConfig`; new `Tiny1M3MPerLayerKRotationConfig(Tiny1M3MConfig)` with flag on.
- **Step-0 byte-identity**: `θ=0` ⇒ `cos(0)=1.0`, `sin(0)=0.0` in fp32 (bit-exact) ⇒ `K_a_new = K_a`, `K_b_new = K_b` exactly ⇒ K unchanged ⇒ QK^T unchanged ⇒ loss unchanged. The implementer must verify `max_abs_diff(MinimalLLM(Tiny1M3MConfig())(ids), MinimalLLM(Tiny1M3MPerLayerKRotationConfig())(ids)) == 0.0` under seed 42.
- **Param count**: `d_k/2 × 1 = 8` per block × 12 blocks = **96 params total** (+0.001% of 0.94M — negligible).

## Engaging the priors (round 1 findings closed)

The r1 taste review identified four blockers; this revision addresses each directly.

**(1) CRITICAL — the r1 default sketch (1 φ per layer, 12 scalars, applied to both Q and K) was a no-op.** Fixed. The r2 mechanism applies the rotation to **K only**, which breaks QK^T symmetry and has a real axis to bind on. The 12-scalar version is **explicitly dropped** — even with K-only, 12 scalars (1 per layer, no per-pair breakdown) gives only a constant per-layer rotation that is then absorbed by W_O in the same way a per-channel gain is. The committed sketch is **per-layer × per-pair (8 angles per layer × 12 layers = 96 angles, K-only)**; this is the smallest version of the lever that breaks cross-pair W_O symmetry.

**(2) CRITICAL — the per-pair sketch "overlaps 185-static-per-head-k-rotation" (closed procedurally after r4 build-smoke failures).** Two points:

- **185's closure is procedural, not empirical.** Per `ideas/185-static-per-head-k-rotation/log.jsonl`, 185 hit `MAX_RECODE_ROUNDS=3` (auto-closed by `flip.sh`) because the box's `git pull` couldn't see the local un-committed model code (`Tiny1M3MStaticKRotationConfig` class at `configs/llm_config.py:6539`). The class was verified locally (commit `fa1ed31`): SMOKE_OK, +384 params, step-0 `max_abs_diff = 0.0` under seed 42. The lever itself **never reached the GPU** — no `evidence.md` exists, no val was measured. Calling 185 "structurally less-expressive than the per-head variant 185 just abandoned" (paraphrasing the r1 review) is empirically incorrect: 185 has *no measured outcome* to be less-expressive than.

- **200 lives in a different (depth × pair) cell than 185 (head × pair).** The parameterizations are orthogonal axes of the same lever family. If the family binds, 185 vs 200 measures *which axis binds* — per-head specialization (185) or per-layer × per-pair specialization (200). The 200 lever is small enough (96 params) that an empirical test is feasible; the r1 reviewer's worry that "if 185 with 32 scalars per block couldn't bind" is moot because 185 was never measured at the GPU. A 200 result (WIN/NULL/DRIFT) provides the first empirical datapoint for the static-K-rotation lever family at 0.94M, and informs which axis (head vs depth) to escalate to Phase-2 ≥135M where more capacity is available.

**(3) Prior evidence — 172 closed the position-axis at 0.94M for the per-head-frequency variant (Δ=+0.0109 wrong-sign tiny).** 172 is **per-head frequency** (multiplicative scale on RoPE's position-dependent angle `θ_i(m)`): the rotation changes *with position*. 200 is **per-layer phase** (constant offset, position-independent): the rotation is *fixed across positions*. These are mechanically orthogonal:
- 172 modulates the **slope** of the angle-vs-position curve per head; same shape, different steepness.
- 200 shifts the **angle** by a constant per layer (K-only), independent of position.
- 172's prior says "per-head scalar modulations of RoPE angles don't bind at 0.94M". 200's lever isn't a per-head scalar modulation of the angle — it's a *rotational* reparameterization of K post-RoPE, applied per-layer × per-pair. Different tensor (K post-RoPE, not Q/K angles pre-rotation), different operation (rotation, not angle scaling), different axis (per-layer, not per-head).

**(4) Prior evidence — depth-axis null cluster (161/142/130/111/116) at 0.94M.** The cluster: `161-dyt-temp` (per-layer τ, scalar), `142-layerscale` (per-channel diagonal gain), `130-rezero` (per-layer scalar α, identity init), `111-drop-path` (per-layer stochastic depth), `116-hyper-connections` (per-layer cross-block gate). All null at 0.94M/12L. The r1 review clusters these with 200 because all are "per-layer scalars at L=12".

200 is structurally different from this cluster:
- The cluster is **scalar per-layer parameters** — a single number per layer (or per channel) that scales a path magnitude.
- 200 is **rotational per-layer parameters** — 8 angles per layer that rotate K by an *orthogonal* block-diagonal matrix.
- A scalar per-layer param is **absorbable by the adjacent W_O** (W_O is per-channel; a uniform scalar gain in front of W_O's input is degenerate with W_O's row-scaling). All five depth-axis nulls were scalars → absorbable → redundant with W_O at 0.94M where W_O has 64×16 params to absorb a 1-scalar-per-layer drift.
- A rotational per-layer param **cannot be absorbed by W_O** because the rotation is per-pair (within d_k) and W_O is per-channel (along d_model → d_k mapping); the rotation's per-pair structure breaks the cross-pair symmetry in a way W_O's per-channel projection has no equivalent operation for. The lever has a structurally fresh axis at this tier.
- `175-alibi-slopes WIN` (Δ=−0.1585) is the **counter-example the r1 review noted**: a *per-head* scalar (not per-layer) that wins because per-head bias is **not absorbed by W_O** (W_O is per-channel; per-head bias lives on attention scores, post-W_Q/W_K, pre-softmax, where W_O doesn't reach). 200's lever is in the same "outside W_O's absorbable set" category as 175 (post-QK-projection, pre-softmax modulation that W_O can't undo).

## Scale evidence
- **175-alibi-slopes** (per-head additive bias, WIN Δ=−0.1585 at tiny1m3m) — closest in-repo validation of the orthogonal-rebase axis. 175 is *bias on attention scores*; 200 is *rotation of K* before the score is computed. Both are pre-softmax, post-Q/K-projection, W_O-non-absorbable.
- **154-rebased-attn** (WIN Δ=−3.48 at tiny1m3m) — fixed random orthogonal rebase of K and V. Validates the rebase axis at 0.94M; 200 is the learned per-layer × per-pair variant of the same axis (154 was *fixed shared on both K and V*; 200 is *learned per-layer × per-pair on K only*).
- **172-per-head-rope-base** (closed null) — establishes that *per-head scalar frequency* modulations of RoPE angles don't bind. Doesn't directly speak to *per-layer × per-pair rotational* modulations of K post-RoPE.
- **185-static-per-head-k-rotation** (closed procedurally, no GPU outcome) — same lever family on the per-head axis. Empirical question open.
- **RoPE at LLaMA-1/2/3, Mistral, Qwen (7B-405B)** — establishes that per-pair rotational parameterizations are a stable lever family at scale; speaks to transferability of the parameterization, not to learnability of depth-specific phases.
- **Transfer-risk: med** — lever form is novel at ≥100M (no published paper tests per-layer static K-rotation), but the underlying orthogonal-rebase mechanism is well-validated by 154 (WIN at 0.94M) and the broader RoPE family at 7B+.

## Why it's worth a slot (r2 bet, sharpened)

**Pattern across the closed list**: scalar per-layer parameters don't bind at 0.94M (depth-axis null cluster). Per-head scalar modulations of RoPE angles don't bind (172). The depth × RoPE-modulation axis has only been tested with scalars, never with rotations. **200 is the first test of "depth × pair × rotation" axis at 0.94M**, parameter-efficient enough (96 params) that an empirical test is cheap.

The bet, in one sharp sentence: **at 0.94M the depth × pair rotational axis is empirically open (185 never ran; 172 tested a different lever; depth-axis cluster tested scalars); 200 tests whether the optimizer can find a per-layer × per-pair static K-rotation that lowers val loss, with 96 params and a step-0 byte-identity guarantee.**

- **WIN**: depth × pair rotational axis binds at 0.94M → unlocks a per-layer × per-pair lever for Phase-2 ≥135M where the depth × head × pair grid is 24L × 12H × 8P = 2,304 cells (vs 12L × 4H × 8P = 384 at tiny1m3m) and the optimizer has richer structure to find.
- **NULL**: closes the per-layer × per-pair static-K-rotation axis at 0.94M. Combined with 172's per-head-frequency null and 185's procedural closure, the entire RoPE-static-rotation family has now been empirically nulled at this tier — informs Phase-2 to look at *non-RoPE* position-encoding extensions.
- **DRIFT**: learned rotation is *worse* than no rotation (the optimizer finds a basis that hurts QK^T alignment) → strong negative on the lever family at this tier; closes the static-K-rotation axis until ≥135M.

A WIN is informative for Phase-2 design; a NULL is also informative (bounds the lever family at this tier); a DRIFT is informative (closes the family). All three outcomes carry information; the lever is worth a slot.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: `autoresearch/baseline-cache.json` box `5b8a7fea8963` (RTX 3060), `val_mean = 6.3988`, `noise_band = 0.04`, `n_measurements = 3`. Re-pull on run day.
- **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule. Δ = −0.005 is a modest but real win for a 96-param lever; a Δ ≤ −0.01 would be a strong confirmation of the depth × pair axis.
- **NULL**: `|trt_val − ctrl_val_mean| < 0.01`. Likely outcome if the depth × pair axis doesn't bind at 0.94M (per the depth-axis null cluster pattern).
- **DRIFT**: `trt_val > ctrl_val_mean + 0.01`. Strong negative — closes the lever family until ≥135M.
- Sub-noise is **inconclusive** per one-seed-only rule; do not propose multi-seed.

## Distinct from closed axes (defensive)
- **154-rebased-attn** (WIN, fixed *random shared* rebase of K and V) — 200 is *learned, per-layer × per-pair, K-only*. Different lever: 154 tests the *axis exists*; 200 tests whether a *learned per-layer × per-pair* axis binds.
- **172-per-head-rope-base** (closed null, per-head *position-dependent* RoPE frequency) — 200 is *position-independent* per-layer × per-pair rotation on K. Different tensor (K post-RoPE, not Q/K angles pre-rotation), different operation (rotation, not frequency scaling), different axis (per-layer, not per-head).
- **175-alibi-slopes** (WIN, per-head additive bias on scores) — 200 is per-layer × per-pair *rotational rebase of K* (pre-softmax, but on K, not scores). Both pre-softmax, both W_O-non-absorbable, but different operation and different parameterization.
- **185-static-per-head-k-rotation** (closed procedurally, no GPU outcome) — 200 is the per-layer × per-pair variant of the same lever family. Same family, different axis (depth × pair vs head × pair); 185's closure is procedural, the family is empirically open.
- **176-v-pre-av-norm** (closed null) — different tensor (V, not K), different op (norm, not rotation).
- **180-qk-logit-conv** (rejected) — pre-softmax QK^T smoothing; different op.
- **152/155/160/166 per-head scalar family** (closed null) — 200 is per-layer × per-pair *rotational* on K, not per-head scalars on scores. Different lever shape.
- **161/142/130/111/116 depth-axis null cluster** (closed null) — all scalar per-layer params, all absorbable by W_O. 200 is rotational per-layer × per-pair, NOT absorbable by W_O. See "Engaging the priors" point (4) above for the structural argument.
- **009-fire-pe** (WIN Δ=−0.064) — fixed RoPE variant with continuous position integration. 200 is a *learned* twist on the K side of RoPE, not a RoPE replacement.
- **021-value-residual** (WIN Δ=−0.034) — cross-block V residual; 200 is intra-block K rotation. Different placement.
- **023-canon-conv** (WIN Δ=−0.06) — gated depthwise causal Conv1d on residual stream; 200 is K rotation, not convolution.
- **024-gated-attention** (WIN Δ=−0.095) — per-head sigmoid output gate post-AV; 200 is K rotation pre-softmax. Different placement and op.
