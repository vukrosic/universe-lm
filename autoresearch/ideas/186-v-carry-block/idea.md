---
id: 186-v-carry-block
status: needs-run
round: 1
updated: 2026-06-15T07:56:19Z
transfer-risk: med
plain: Let each attention head carry a small fraction of its own previous value vector forward in time (a learned per-head scalar, starting at 0 so step-0 is byte-identical), like a tiny recurrent filter on V.
---

# 186 — Within-Block V-Carry (Per-Head Learnable Recurrence on V)

## Source
- Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Inference with Linear Attention" (ICML 2020, arXiv:2006.16236) — the original linear-attention derivation shows that the softmax(QK^T) V can be rewritten as a recurrence when Q, K are feature-mapped, and the recurrence is `S_t = S_{t-1} + k_t ⊗ v_t`. The carry concept (carry `S_{t-1}` forward) is the foundational mechanism.
- 154-rebased-attn (Shi et al. 2024, arXiv:2407.06641) — rebases K and V before softmax; WIN at tiny1m3m. The rebase is a *static* transformation; 186 is a *learned* dynamic carry.
- 168-av-output-carry (closed null) — carries the *attention output* (AV) across blocks, not V within a block. Different placement (cross-block vs within-block) and different tensor (AV vs V).
- 164-q-carry (closed null) — carries Q across blocks; null at 0.94M. V-side carries are the orthogonal axis (021-value-residual was a WIN for V-side cross-block).
- 021-value-residual (WIN, Δ=−0.034) — carries V *across blocks*; 186 carries V *within* a block (recurrence along the time axis). Different placement.
- In-repo context: 154 established the orthogonal-rebase axis as a strong lever at 0.94M. 186 is a *learned, dynamic* (time-dependent) version of the V rebase — instead of a fixed static rotation, the carry lets V integrate information from previous time steps. The mechanism is a 1D leaky integrator on V (a learned EMA over the time axis).

## Mechanism
Standard attention V is computed as `V = W_V @ x` and used in `out = softmax(QK^T) @ V`. The V vectors at different time steps are independent — `V_t` doesn't depend on `V_{t-1}`.

With within-block V-carry, for each head h, learn a scalar `α_h ∈ [−1, 1]` that controls how much of `V_{h, t-1}` is mixed into `V_{h, t}` before the attention product:
```
V_h_t = V_h_t + α_h · V_h_{t-1}            # recurrent mix
```
The recurrence is run causally (left-to-right) within the attention computation: `V_h_0` is the standard V, `V_h_1 = V_h_1 + α_h · V_h_0`, `V_h_2 = V_h_2 + α_h · V_h_1 = V_h_2 + α_h · V_h_1_orig + α_h² · V_h_0_orig`, etc. After T steps, `V_h_T` is a geometric-weighted sum of all prior `V_h_t_orig`, with weights `(1, α, α², ..., α^{T-1})` (approximately, ignoring the time-varying W_V @ x contribution).

**Parameterization**: `α_h = tanh(α_raw_h)` with `α_raw_h = 0` init ⇒ `α_h = tanh(0) = 0` exactly ⇒ `V_h_t = V_h_t + 0 · V_h_{t-1} = V_h_t` ⇒ **byte-identical to baseline at step 0**. The tanh parameterization keeps `|α_h| ≤ 1`, preventing exponential explosion (the geometric sum `1 + α + α² + ...` is finite iff `|α| < 1`, and the tanh output is bounded by 1).

**Step-0 byte-identity**: `α_h = 0` for all h ⇒ `V_h_t = V_h_t` exactly (no carry) ⇒ attention product unchanged ⇒ QK^T unchanged ⇒ softmax unchanged ⇒ loss and gradient bit-identical to baseline. The implementer should verify with `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0`.

**Training dynamics**: when the optimizer pushes `α_h` away from 0, the head starts integrating V over time. For `α_h = 0.5` and T=2048, the effective window is ~2 time steps; for `α_h = 0.99`, the effective window is ~100 time steps. The optimizer can find a per-head α_h that matches the head's natural temporal range (some heads want long-range V integration, others want short-range).

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_v_carry_block: bool = False` to `MultiHeadAttention.__init__`. Allocate `self.v_carry_alphas = nn.Parameter(torch.zeros(n_heads))` (init 0 ⇒ α_h = 0). In `forward`, after computing V (post-W_V, shape `[B, H, T, d_k]`) and before the attention product, apply the per-head recurrent mix: `α = tanh(self.v_carry_alphas).view(1, H, 1, 1)`; then `V_new = torch.empty_like(V)`; `V_new[:, :, 0, :] = V[:, :, 0, :]`; for t in 1..T-1: `V_new[:, :, t, :] = V[:, :, t, :] + α * V_new[:, :, t-1, :]`. This is a sequential scan over T — for T=2048 it's ~2000 sequential ops per head, but the inner work is just a scalar broadcast + add per step. Can be parallelized via a `torch.cumsum`-style trick: `cumV_t = Σ_{s≤t} α^{t-s} · V_s` (a single cumsum-like op). Or use the standard linear-attention scan kernel from fla-org.
  - `configs/llm_config.py` — add `use_v_carry_block: bool = False`. Add `Tiny1M3MVCarryBlockConfig(Tiny1M3MConfig)` with `use_v_carry_block: bool = True`.
  - `models/llm.py` — thread into both `TransformerBlock` sites.
- **Config flag**: `use_v_carry_block: bool = False`.
- **Param count**: H=4, n_layers=12. Per block: 4 α scalars. Total: 48 params (+0.005% of 0.94M). Negligible.
- **Intuition (why it might lower val loss)**: 154 showed that *re-orienting* K and V before softmax (a static transformation) gives a Δ=−3.48 record break at 0.94M. The mechanism: random orthogonal rebase changes the basis in which Q, K, V interact, breaking default unfavorable alignments in the random-init weights. 186 is a *learned dynamic* version: instead of a static basis change, V integrates information from previous time steps. The carry gives each head a free "memory" of V over the time axis, similar to the linear-attention recurrence. The hope: the V-axis integration captures long-range V dependencies that pure attention (where V at time t is independent of V at time t−1) can't.
- **Why it might bind where 168-av-output-carry nulled**: 168 was *cross-block* (carries AV across blocks); 186 is *within-block* (carries V along time within a single block). Cross-block carries require 12 sequential blocks to develop a useful signal; within-block carries develop within a single block. 168 also had a fixed α=0.999 (not learned, not per-head); 186 is per-head learned. The lever shape is structurally different — 186 is more like a soft recurrence than a block-to-block skip.

## Scale evidence
- Linear attention / Katharopoulos 2020 — recurrence on the *KV* outer product, not on V alone. Validated at 1B+ (Performer, Linear Transformers, RWKV).
- 154-rebased-attn (closest analog): WIN at tiny1m3m (Δ=−3.48).
- 021-value-residual (WIN, cross-block V carry): V-side carries are validated at 0.94M.
- **Transfer-risk: med** — the within-block V-carry lever form is novel at ≥100M (no published paper tests exactly this), but the underlying *recurrence-on-V* mechanism is well-validated by linear attention at 1B+ and by 021's cross-block V carry at 0.94M. The bet is that within-block V carry can be a useful lever in addition to (or instead of) cross-block V carry.

## Why it's worth a slot
The bet, in one sharp sentence: **154's WIN established the K/V rebase axis at 0.94M, and 186 is the *learned dynamic* version of V rebase (a recurrent carry along the time axis) — if the V-axis reorientation is the binding part of 154's WIN, the optimizer should be able to find a *better* V rebase by learning a per-head time-dependent carry, and the per-head axis lets heads specialize in their temporal integration**. A null at 0.94M would close the within-block V-carry axis and tell us the static rebase (154) is the binding part of 154's WIN, not the dynamic integration. A win would unlock a per-head time-recurrent V axis for Phase-2 ≥135M where the per-head gradient signal is larger and the optimizer can find a richer mix of per-head temporal ranges.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: `autoresearch/baseline-cache.json` box `5b8a7fea8963` (RTX 3060), `val_mean = 6.3988`, `noise_band = 0.04`, `n_measurements = 3`. Re-pull on run day.
- **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule. A WIN at this magnitude would be informative; given 154's Δ=−3.48 win, a Δ ≤ −0.005 from 186 is plausible.
- **NULL**: `|trt_val − ctrl_val_mean| < 0.01`. Most likely outcome per the 168-av-output-carry null pattern (recurrence on V doesn't bind at 0.94M in the cross-block setting); the within-block setting may or may not differ.
- **DRIFT**: `trt_val > ctrl_val_mean + 0.01`. Could occur if α_h drifts to a magnitude that the recurrence amplifies (e.g., α_h close to ±1 with T=2048 amplifies the V magnitudes by 100× and the softmax saturates).
- **Sub-noise is inconclusive** per one-seed-only rule.

## Distinct from closed axes (defensive)
- 154-rebased-attn (WIN) — fixed orthogonal rebase of K, V. 186 is a *learned dynamic* (time-dependent) carry on V only. Different lever: 154 tests static rebase; 186 tests dynamic carry.
- 168-av-output-carry (null) — cross-block AV carry with fixed α=0.999. 186 is within-block V carry with per-head learned α_h. Different placement and parameterization.
- 164-q-carry (null) — cross-block Q carry. 186 is within-block V carry. Different tensor and placement.
- 021-value-residual (WIN) — cross-block V carry. 186 is within-block V carry. Different placement (cross-block vs within-block).
- 012-gated-deltanet / 008-gated-deltanet (closed) — full linear attention with delta rule. 186 is a single learned scalar α_h, not a full delta rule. Different lever shape.
- 004-retnet-retention (null) — RetNet's full retention mechanism. 186 is a single learned scalar α_h, not the full retention. Different lever shape.
