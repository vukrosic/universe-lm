---
id: 185-static-per-head-k-rotation
status: tasting
round: 1
updated: 2026-06-15T07:16:00Z
transfer-risk: med
plain: Give each head its own small learned rotation matrix for its keys (no position information, just a static re-orientation), starting at the identity so step-0 is byte-identical — a per-head generalization of the rebased-attention idea.
---

# 185 — Static Per-Head K-Rotation (Learned Per-Head Orthogonal Rebase of K)

## Source
- 154-rebased-attn (Shi et al. 2024, arXiv:2407.06641) — applies a *fixed* (random, frozen) rebase matrix to K and V before softmax; WIN at tiny1m3m (Δ=−3.48, record break). Validated at 1B+ scale.
- 172-per-head-rope-base (closed null) — applied a per-head *base frequency* in RoPE, position-dependent. Different shape: 172 modulates the *position* rotation (angle scales with m·θ_base), 185 modulates a *static* per-head rotation (angle independent of m).
- Su et al., "RoFormer" (RoPE, 2021) — pairs of 2D rotations in head-dim planes, position-dependent. 185's static rotation is structurally identical to RoPE's per-pair rotation matrices, but with the *position-dependent* angle replaced by a *learned* angle. This is essentially "per-head RoPE without the position input".
- 176-v-pre-av-norm (closed null) — V-normalization pre-AV; different placement (V, not K). 180-qk-logit-conv (rejected) — pre-softmax conv on QK^T; different mechanism (smoothing, not rotation).
- In-repo context: 154-rebased-attn WIN establishes the orthogonal-rebase axis as a strong lever at 0.94M. 185 is the *learned* version of 154 (per-head), with the K-rebase and V-rebase decoupled (185 only rebases K; 154 rebases both K and V with the same fixed matrix). The per-head axis is fresh at this tier.

## Mechanism
Standard attention:
```
K = W_K @ x                                     # [B, H, T, d_k]
scores = Q @ K^T / sqrt(d_k)                    # [B, H, T, T]
attn = softmax(scores)                          # [B, H, T, T]
out = attn @ V                                  # [B, H, T, d_k]
```
With static per-head K-rotation:
```
K = W_K @ x                                     # [B, H, T, d_k]
K = R_h @ K                                     # R_h: [H, d_k, d_k], applied per head
scores = Q @ K^T / sqrt(d_k)                    # [B, H, T, T]
attn = softmax(scores)                          # [B, H, T, T]
out = attn @ V                                  # [B, H, T, d_k]
```
`R_h ∈ R^{d_k × d_k}` is a per-head *orthogonal* matrix (so the rotation preserves the norm of K and the softmax's "temperature" — no re-scaling of the QK^T magnitudes, just a basis change). Parameterize as a product of 2D rotations in `d_k/2 = 8` planes:
```
R_h = R_h^0 · R_h^1 · ... · R_h^{d_k/2-1}      # 8 rotation matrices, each [2, 2]
R_h^{i} = [[cos θ_{h,i}, -sin θ_{h,i}], [sin θ_{h,i}, cos θ_{h,i}]]   # applied to the (2i, 2i+1) pair
```
The 8 angles `θ_{h,0..7}` per head (one per plane) are the learnable parameters. Init `θ_{h,i} = 0` for all h, i ⇒ `R_h = I_{d_k}` exactly ⇒ `K = R_h @ K = K` ⇒ **byte-identical to baseline at step 0** (the rotation is the identity at init, the QK^T is unchanged, the softmax is unchanged).

**Why orthogonal, not arbitrary**: an arbitrary linear map `M_h` would re-scale the K vectors and shift the softmax temperature. The orthogonal constraint keeps `||M_h v|| = ||v||` for all `v`, so the QK^T magnitudes are preserved (modulo the basis change, which doesn't change the QK^T magnitudes between the same pair — `<M_h q, M_h k> = <q, k>` for orthogonal `M_h` since the inner product is preserved). This is the same "preserve the dot product" property that RoPE has for its position rotation, and it's what made 154's fixed orthogonal rebase work cleanly.

**Step-0 byte-identity**: `θ_{h,i} = 0` for all h, i ⇒ `R_h = I` exactly (in fp32, `cos(0) = 1.0` and `sin(0) = 0.0` are bit-exact in IEEE 754) ⇒ `K = I @ K = K` exactly ⇒ QK^T unchanged ⇒ softmax unchanged ⇒ loss is bit-identical to baseline. The implementer should verify with `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0`.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_static_k_rotation: bool = False` to `MultiHeadAttention.__init__`. Allocate `self.k_rotation_angles = nn.Parameter(torch.zeros(n_heads, d_k // 2))` (init 0 ⇒ identity rotation). In `forward`, after computing K (post-W_K, pre-softmax), build the orthogonal matrix `R_h` for each head from the angles, and apply `K = einsum("hij,bhtj->bhti", R_h, K)` (broadcasting `R_h` over batch and time). The R_h construction can use a small helper function or inline: for each (h, i), `R_h^i = [[cos θ, -sin θ], [sin θ, cos θ]]` on the (2i, 2i+1) plane; the full `R_h` is the product of the 8 `R_h^i` block-diagonal.
  - `configs/llm_config.py` — add `use_static_k_rotation: bool = False`. Add `Tiny1M3MStaticKRotationConfig(Tiny1M3MConfig)` with `use_static_k_rotation: bool = True`.
  - `models/llm.py` — thread into both `TransformerBlock` sites.
- **Config flag**: `use_static_k_rotation: bool = False`.
- **Param count**: H=4, d_k=16, planes=8. Per block: H × d_k/2 = 4 × 8 = 32 angles. Total: 32 × 12 = **384 params (+0.041% of 0.94M)**. Negligible.
- **Intuition (why it might lower val loss)**: 154 showed that a *fixed* random orthogonal rebase of K and V gives a Δ=−3.48 record break at 0.94M. The mechanism: a random orthogonal rebase changes the basis in which Q and K are compared, which can break unfavorable "default" alignments in the random-init W_Q, W_K basis. 185 is a *learned* version, with the rebase *per-head* and applied to K only (V is left alone). The hope: the optimizer can find a per-head rebase that's better than the random one, and the per-head axis lets heads specialize (some heads want one rotation, others another). 154's WIN tells us the rebase axis is real at 0.94M; 185 tests whether the *learned* axis binds at this tier.
- **Why it might bind where 172 (per-head RoPE base) nulled**: 172 was *position-dependent* (the rotation angle scales with `m · θ_base`). 185 is *position-independent* (a static rotation per head). The position-dependent lever requires the model to learn *per-position* angular structure, which is a stronger signal than a single per-head basis change. 185's static rotation is a weaker prior (just a basis change) and may bind more easily.

## Scale evidence
- 154-rebased-attn (the closest analog): WIN at tiny1m3m (Δ=−3.48); source paper tested at 1B+.
- RoPE at 1B-405B (LLaMA 1/2/3, Mistral, Yi, Qwen, Gemma, Falcon) — the *position-dependent* version of per-head rotation.
- **Transfer-risk: med** — the static-rotation lever form is novel at ≥100M (no published paper tests exactly this); the *underlying* orthogonal-rebase mechanism is well-validated by 154's WIN at 0.94M. The bet is that the rebase axis is real, and a *learned* per-head version is a stricter test of the axis.

## Why it's worth a slot
The bet, in one sharp sentence: **154-rebased-attn's WIN (Δ=−3.48) established the orthogonal-rebase axis as a strong lever at 0.94M, and 185 is the natural learned generalization (per-head, K-only, position-independent) — if the rebase axis is real, the optimizer should be able to find a *better* basis than the random one 154 used, and the per-head axis lets heads specialize in their basis choice**. A null at 0.94M would tell us that the *random* rebase was the binding part of 154's WIN (the noise injection aspect, not the reorientation), and a learned version can't improve on the random one. A win would unlock a per-head static-rotation axis for Phase-2 ≥135M where the per-head gradient signal is larger and the optimizer can find a richer mix of per-head bases.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: `autoresearch/baseline-cache.json` box `5b8a7fea8963` (RTX 3060), `val_mean = 6.3988`, `noise_band = 0.04`, `n_measurements = 3`. Re-pull on run day.
- **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule. With 154's WIN being Δ=−3.48 (record break), a Δ=−0.005 would be a modest but real win. A WIN at this magnitude would be informative; a *larger* WIN (e.g., Δ ≤ −0.01) would be a strong confirmation of the rebase axis.
- **NULL**: `|trt_val − ctrl_val_mean| < 0.01`. Likely outcome if the random rebase was the binding part of 154's WIN.
- **DRIFT**: `trt_val > ctrl_val_mean + 0.01`. DRIFT would mean the learned rebase is *worse* than no rebase (perhaps the optimizer finds a basis that hurts QK^T alignment) — a strong negative result that closes the axis.
- **Sub-noise is inconclusive** per one-seed-only rule.

## Distinct from closed axes (defensive)
- 154-rebased-attn (WIN) — fixed rebase of K, V (same matrix, not per-head, not learned). 185 is *learned per-head* on K only. Different lever: 154 tests the *axis exists*; 185 tests whether *per-head learned* rebase can improve on fixed.
- 172-per-head-rope-base (closed null) — per-head position-dependent RoPE base. 185 is *position-independent*. Different mechanism.
- 176-v-pre-av-norm (closed null) — V normalization pre-AV. 185 is K re-rotation pre-softmax, on a different tensor (K, not V) and a different operation (rotation, not normalization).
- 180-qk-logit-conv (rejected) — pre-softmax conv on QK^T (smoothing, not rotation). Different operation.
- Closed "per-head scalar" family (152, 155, 160, 166) — all per-head scalars on score magnitudes. 185 is per-head *matrices* on K, not per-head scalars on scores. Different lever shape.
- 181-cross-head-rmsnorm (rejected) — cross-head normalization post-AV. 185 is K rotation pre-softmax. Different placement and operation.
