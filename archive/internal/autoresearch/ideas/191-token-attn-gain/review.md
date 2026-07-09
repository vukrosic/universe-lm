# Review log — 191 per-token-attn-gain

## r1 — 2026-06-15 — verdict: approve
- **Source real + current.** NormFormer (Shleifer et al. 2021, arXiv:2110.09423, BERT-base ~110M); CaiT/ResNet-vd (Touvron et al. 2021, ImageNet). Both well-known; the per-token output-gain primitive is well-validated at ≥100M. transfer-risk: low tag justified by source scale.
- **Mechanism is a mechanism, not an HP.** Per-token scalar gain `γ_t` (init 0) on attention output `out * (1+γ_t)` before W_O is a *granularity* lever (T scalars/block vs H=4 for 160 vs d_model=64 for 142). The 512× DOF increase over the closed 160-axis is a real structural change, not a constant tweak. Step-0 bit-identity holds: γ_t=0 ⇒ (1+0)=1 ⇒ exact baseline at step 0.
- **Axis-distinct from closed levers.** 142-layer-scale (per-channel diagonal on residual), 160-rms-gain-per-head (H scalars post-AV), 176-v-pre-av-norm (V-side pre-attention). 191 is the per-token post-AV axis, none of which are closed. 168-av-output-carry closed NULL on a different mechanism (cross-block carry, not per-token per-block gain). The bet is crisp: per-token granularity escapes the W_O absorption that absorbed the per-head 160 axis. Falsifiable.
- **tiny1m3m-only.** Sketch references T=2048·12 blocks = 24,576 params (+2.6% of 0.94M); no screen20m, no ladder, no scale-up. Bit-identity verified algebraically.
- **< 200 LoC.** Slots in cleanly alongside the existing `use_attn_output_gate` / `use_head_gain` hooks in `MultiHeadAttention.forward` (post-AV, pre-merge-reshape, pre-W_O). Estimated ~5-10 LoC flag decl + init + 3-line forward hook + ~3-line assert for mutex. Far under budget.
- **Falsifiable pass-bar.** Cache-mean baseline 6.4394±0.04 at tiny1m3m seed 42; expected Δ within |Δ|<0.01 noise band is predicted NULL (W_O absorption extends to per-token); |Δ|≥0.02 right-sign is WIN; |Δ|≥0.04 wrong-sign is DRIFT. Two-ctrl bracket rule applies (PIPELINE.md §2). One seed only.
- **Concurrency check.** `git diff HEAD models/layers.py` shows parallel-AI is staging 188 (use_cross_block_kv_share). 191's flag name (`use_token_attn_gain`) does not conflict. Code gate can land 191 independently or alongside 188.

### Findings for the plan/code gate (non-blocking)
- **Pick one variant.** Design sketch mentions both `[T]` broadcast-across-batch and `[B,T]` per-batch learnable; the plan must commit to one. Recommend `[T]` shared-across-batch (same shape as the closed 142-channel `layer_scale` and the open 160-head `head_gain` patterns) — fewer params, simpler LoC, and the bet is per-token per-block, not per-example.
- **Assert mutex with closed gates.** Plan should add the standard mutual-exclusion asserts alongside the existing `use_attn_output_gate` / `use_head_gain` / `use_attn_output_channel_gate` asserts so the build-smoke catches double-on configs (e.g. `use_token_attn_gain ∧ use_head_gain`). Both-on would restructure the lever (post-AV axis composed twice).
- **Baseline control.** Use the 175-alibi champion ctrl shape (use_fire_pe=True, use_alibi_slopes=True — current cache) per `autoresearch/LEADERBOARD.md`. Note 175 is the current cache-authoritative baseline.

### Verdict
`approve` — sound, axis-distinct, bit-identical at step 0, falsifiable, transfer-risk tag justified, ≤200 LoC. Routes to code gate (`needs-plan`).
