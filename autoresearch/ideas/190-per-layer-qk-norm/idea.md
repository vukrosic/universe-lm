---
id: 190-per-layer-qk-norm
status: needs-review
round: 2
updated: 2026-06-15T12:02:07Z
transfer-risk: low
plain: Replace 016's per-channel RMS gain (16 scalars per head-dim) on Q and K inside attention with a single scalar gain per block per side — a strictly coarser parameterization that tests whether 016's WIN was driven by per-channel resolution or by simple block-level scale.
---
# 190 — Per-Layer QK Norm (Scalar γ Per Block Per Side, Not Per-Channel)

## Source
- 016-qk-norm (in-repo WIN Δ=−0.0138 at tiny1m3m, val 6.3906) — symmetric RMSNorm on Q and K pre-softmax. **Implementation is `make_norm(self.d_k, "rmsnorm", _qk_use_ln)` at `models/layers.py:1503-1504`, which returns `RMSNorm(d_k)` with `weight` of shape `(d_k=16,)`** (`models/layers.py:473`). The gain is **per-channel** (along the d_head axis), broadcast across the H axis because heads are stacked along H and RMSNorm only reduces the last dim. **016 has no per-head γ axis.** Per block: 16 (Q) + 16 (K) = 32 params. Per run: 12 blocks × 32 = **384** params.
- 162-q-only-norm (closed null Δ=−0.0043 inside band) — Q-only RMSNorm with per-channel γ on d_head.
- 165-k-only-norm (closed null Δ=−0.0293 inside band) — K-only RMSNorm with per-channel γ on d_head.
- 169-qk-norm-depth (closed null Δ=−0.020 inside band) — per-depth scalar `qk_norm_scale` shared across Q and K; closed by 016's WIN.
- Zhang & Sennrich, "RMSNorm" (arXiv:1910.07467, 2019) — base RMSNorm formulation.

## Mechanism
016's QK norm is `Q_norm = RMSNorm(Q) · γ_c`, `K_norm = RMSNorm(K) · γ_c` where `γ_c ∈ R^{d_head}` (per-channel, length 16). 190 keeps the symmetric QK-norm structure but **collapses the gain from a length-d_head vector to a length-1 scalar per block per side**: one `γ_bQ` and one `γ_bK` per block, applied AFTER the existing RMSNorm's per-channel weight, BEFORE the QK dot product.

The bet: at `d_k=16`, the per-channel γ axis (16 scalars/block/side, 384 total) may be over-parameterized for the binding signal. 016's WIN at 0.94M might be driven by *block-level scalar magnitude* (a single scale that bounds the QK logit range), not by *per-channel resolution* (which channel gets how much gain). If true, collapsing to a scalar γ per block per side gives **16× fewer γ params for the same block-level conditioning benefit**, with no information loss on the binding axis.

**Distinct from closed levers**: 169 added a single scalar shared across Q and K (collapses the Q/K symmetry); 190 keeps Q and K separate scalars (preserves the symmetry 162+165 closed, 016 attributed WIN to). 162/165 closed the Q-side and K-side of the 016 attribution with **per-channel** γ; 190 closes the *granularity* axis (per-channel vs scalar) while preserving QK symmetry. **None of the closed levers cover this axis.**

## Design sketch
- **File**: `models/layers.py` — add two new MHA flags, both default off (no module registered, no branch taken at init → existing configs byte-identical):
  - `qk_norm_scalar_per_block: bool = False` — when True, after `Q_norm = self.q_norm(Q)` and `K_norm = self.k_norm(K)` and after the per-channel γ of the existing `RMSNorm(d_k)` has been applied, multiply by a per-block scalar:
    - `Q_norm = Q_norm * self.qk_norm_scalar_q` (scalar `nn.Parameter(torch.ones(()))`)
    - `K_norm = K_norm * self.qk_norm_scalar_k` (scalar `nn.Parameter(torch.ones(()))`)
  - `qk_norm_scalar_qk_shared: bool = False` — independent sub-flag; if True, replace the two scalars with **one** shared scalar `qk_norm_scalar` applied to both Q and K (the 169-style collapse, kept as a knob but not the default).
- **Defaults**: `qk_norm_scalar_per_block = False`, `qk_norm_scalar_qk_shared = False`. Init `1.0` for both.
- **Q/K split preserved by default** (separate scalars for Q and K) so 190 does NOT collapse to the 162/165 closed axes and preserves the QK symmetry 016's WIN was attributed to. The Q/K-shared variant is gated behind a second flag and is a *different* lever (would re-collapse to 162/169 split); default off.
- **Step-0 bit-identity**: with `qk_norm_scalar_per_block=True` and `qk_norm_scalar_qk_shared=False`, γ=1 ⇒ `Q_norm·1 = Q_norm`, `K_norm·1 = K_norm` — bit-identical to the 016 forward path at step 0 (fp32 max-abs-diff = 0.0). **Corrected framing**: the existing 016 RMSNorm has `weight ∈ R^{d_head}` (per-channel) init to all ones, not per-head. The new scalar γ sits downstream of that per-channel weight and is its own parameter.
- **Params** (corrected math): 016 baseline = 12 × 2 × 16 = **384** γ params. 190 separate-scalar (default) = 12 × 2 × 1 = **24** γ params. 190 Q/K-shared (sub-flag on) = 12 × 1 = **12** γ params. Δ vs 016 = −360 params (−0.038% of 0.94M) for the default separate-scalar variant. Negligible — the lever is the *granularity change*, not the budget change.
- **A/B baseline** (corrected): the trt is `Tiny1M3MQKNormConfig` (016 ctrl, val 6.3906) + `qk_norm_scalar_per_block=True`; the ctrl is `Tiny1M3MQKNormConfig` (016 ctrl) alone. **The A/B must isolate the granularity axis (per-channel γ → scalar γ), not the existence axis (RMSNorm vs no-RMSNorm).** Comparing 190 to `Tiny1M3MConfig` (no QK norm) would conflate 016's WIN with 190's lever and tell us nothing about the per-channel-vs-scalar attribution question.
- **Tier**: tiny1m3m only, seed 42 only, single config, single tier — ONE-TIER-ONLY rule.

## Pass bar (re-anchored)
With the corrected mechanism the pass bar should be:
- **PASS**: trt val_loss ≤ 016-ctrl (6.3906) − 0.005, i.e. trt ≤ 6.391 — matches 016's WIN magnitude under scalar γ ⇒ scalar γ is sufficient, per-channel resolution was over-parameterized.
- **NULL band |Δ| < 0.005** vs 016-ctrl, i.e. trt ∈ (6.3856, 6.3956) — inconclusive; the granularities are statistically tied at this tier and we cannot distinguish them.
- **LOSS**: trt val_loss > 016-ctrl + 0.005, i.e. trt > 6.3956 — the per-channel resolution is binding; collapsing to scalar γ throws away capacity and is a regression.
- **DRIFT**: trt val_loss > 016-ctrl + 0.05 (sanity, indicates a broken lever, not a real axis).
- One seed (42), tiny1m3m only. 0.005 is the realistic single-seed threshold at this tier (matches the 016 plan's own bar; the taste-reviewer's prior −0.01 suggestion was too loose).

## Attribution insight (reframed)
The closed.md lineage (162-Q-only NULL with per-channel γ, 165-K-only NULL with per-channel γ, 169-depth NULL with shared scalar, 016-QK per-channel WIN Δ=−0.0138) leaves an open question on the γ granularity axis: was 016's WIN driven by the per-channel resolution (16 scalars/block/side), or by the simpler block-level scalar magnitude (1 scalar/block/side)? 190 directly tests this:
- **190-WIN** (matches 016's WIN magnitude with a single scalar γ per block per side): scalar γ suffices, per-channel resolution was over-parameterized at 0.94M; the binding axis is *block-level QK magnitude*, not *per-channel specialization*.
- **190-NULL** (loses to 016-ctrl by > 0.005): per-channel resolution IS binding, scalar γ throws away capacity; 016's WIN was carried by the 16-channel resolution per side.
- **190 within |Δ| < 0.005 of 016-ctrl**: inconclusive — the granularities are statistically tied at this tier; we cannot distinguish them without a larger model or longer horizon.

Either branch is informative for the QK-norm attribution puzzle and forward-useful for the 135M recipe.

## Scale evidence
016-qk-norm WIN at tiny1m3m (Δ=−0.0138, val 6.3906); QK-norm literature validated at LLaMA / Gemma-2 / Qwen-2.5 (≥7B). Transfer-risk: low (the lever is a strictly coarser parameterization than 016's per-channel γ, not a niche one).

## Why it's worth a slot
- **Sharp attribution bet**: directly tests the granularity axis (per-channel vs scalar γ) on the 016 WIN, orthogonal to the closed Q-side / K-side / per-depth axes.
- **Mechanism-shaped (not HP)**: the lever is one structural choice (scalar vs per-channel γ), not a knob to sweep.
- **Both branches informative**: WIN confirms scalar γ suffices (over-param result); NULL confirms per-channel resolution binds (informative for Phase-2 135M where capacity matters more).
- **Identity-able**: step-0 byte-identical to 016 ctrl when scalar γ=1.
- **<200 LoC, no scope creep**: ~15 LoC branch in `MHA.__init__`/`forward` + two config flags + a thin `Tiny1M3MQKNormScalarConfig` subclass. No conflicts on `models/layers.py` or `configs/llm_config.py` (git diff clean against the other AI's work).
