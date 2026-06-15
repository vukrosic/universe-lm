---
id: 202-grouped-value-projection
status: needs-taste
round: 1
updated: 2026-06-15T09:00:00Z
transfer-risk: low
plain: Group attention heads into clusters that share a single Value projection (init each cluster's projection to match the per-head baseline so step-0 is byte-identical), like partial value-sharing across heads — between full MHA and full MQA.
---

# 202 — Grouped Value Projection (Share W_V Across Head Groups, K and Q Stay Per-Head)

## Source
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (arXiv:2305.13245, 2023) — interpolates MQA↔MHA via group size; validated at Llama-2-7B/13B/70B.
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (MQA, arXiv:1911.02150, 2019) — single shared K, V across all heads.
- 178-mqa-gated (closed null Δ=+0.1647) — gated MQA probe with per-head α blend between head-local and shared K, V. Closed at 0.94M.
- closed.md line "MHA vs GQA, MLA, Tied QK" — closed fixed-group-size variants. 202 is a different lever: group size is *fixed* but the per-head α blend from 178 doesn't apply.
- **Key novelty**: GQA shares K, V across heads within a group; 202 shares **V only** (K stays per-head). This is an asymmetric variant of GQA — V-sharing is well-motivated (V is "what to read from", can be shared) while K-sharing is "what to attend to" (more head-specific).

## Mechanism
Standard attention: each head h has its own `W_V_h ∈ R^{d_model × d_k}`. Total: H × d_model × d_k = 4 × 64 × 16 = 4096 V params per block.

Grouped V: heads are partitioned into G groups of size h_per_group = H/G. Within each group, all heads share the same `W_V_group`:
```
heads in group g → W_V_g ∈ R^{d_model × d_k}
```
With G=2 groups (h_per_group=2), there are 2 W_V matrices per block × 12 blocks = 24 W_V matrices instead of 48. Saves 24 × 64 × 16 = 24,576 params (-2.6% of 0.94M).

For bit-identity at step 0: initialize each group's W_V to match the *mean* of the per-head W_V in that group (so each group's W_V is the average of the 2 head's W_V_init). This is *not* byte-identical to per-head V projection, but is a reasonable init that preserves the V-projection statistics.

**Or use the soft-blend formulation from 178**: `W_V_h_eff = (1 − α_h) · W_V_h + α_h · W_V_group`. At α=0, per-head W_V (bit-identical). At α=1, group-shared W_V.

## Design sketch
- **File**: `models/layers.py` — modify the W_V projection to optionally be group-shared with α blend.
- **Config flag**: `use_grouped_v_projection: bool = False`, `v_group_size: int = 2`, `v_group_alpha_init: float = -10.0` (sigmoid ≈ 0).
- **Compute**: per block, compute W_V_group for each group (mean of in-group W_V's, or learned). Apply soft blend: `W_V_h_eff = (1 − sigmoid(α_h)) · W_V_h + sigmoid(α_h) · W_V_group`.
- **Bit-identical at step 0**: α_h_raw = -10 ⇒ sigmoid ≈ 0 ⇒ `W_V_h_eff = W_V_h` exactly.
- **Params**: H α scalars + G group W_V's per block × 12 blocks = 4 × 12 + 2 × 4096 × 12 = 48 + 98,304 ≈ 98k params (replaces 48 × 4096 = 196,608; net −98k, −10.4% of 0.94M).
- **Intuition**: V is "what to read from", which is largely content-based and can be shared across heads without losing expressive power. K is "what to attend to", which is more head-specific. Asymmetric GQA (V-only sharing) preserves the head-specific attention pattern (K) while reducing V redundancy.

## Scale evidence
GQA validated at Llama-2 7B/13B/70B. The asymmetric V-only variant is novel (no published paper I'm aware of). 178-mqa-gated closed null at 0.94M (probe-style blend, full K and V sharing). Transfer-risk: low (lever is a GQA variant; GQA itself is well-validated).

## Why it's worth a slot
**Pattern**: 178-mqa-gated closed null at 0.94M. 202 is the *V-only* version of GQA (asymmetric) — different from 178's symmetric K, V gating. The bet: at 0.94M, V redundancy across heads is real (V projections can be shared without losing accuracy); K redundancy is *not* real (K is head-specific). A 202 WIN would mean V-only GQA binds at 0.94M (where symmetric MQA-gated doesn't); a 202 NULL would confirm the V redundancy is also not a binding axis at 0.94M.
