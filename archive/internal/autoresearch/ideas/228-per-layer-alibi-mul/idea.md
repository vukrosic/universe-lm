---
id: 228-per-layer-alibi-mul
status: needs-taste
round: 1
updated: 2026-06-16T01:05:00Z
transfer-risk: low
plain: Multiply the ALiBi slope (the existing per-head distance bias) by a per-block learnable scalar. Different from the closed ALiBi variants — this keeps the original per-head slopes fixed but lets each layer scale the whole ALiBi effect up or down.
---

# 228 — Per-Layer ALiBi Slope Multiplier

## Source
ALiBi (Press et al. 2022, arXiv:2108.12409) adds a per-head additive distance bias `m_h * (i - j)` to QK scores before softmax, where `m_h` is a per-head slope. Closed 175-alibi-slopes WIN at 0.94M (Δ-0.1585) is the baseline win.

**228 keeps the per-head ALiBi slopes fixed but adds a per-block multiplicative scalar `α_l`** so the effective bias becomes `α_l * m_h * (i - j)`. This is a "depth-dependent ALiBi strength" lever: early layers can use `α_l ≈ 1` (full ALiBi effect), late layers can use `α_l < 1` (weaker ALiBi, since late layers have learned content-based attention anyway).

Related closed ideas:
- 213, 214, 215, 216 are ALiBi challengers (different mechanisms). 228 is a *multiplier* on top of 175.
- 184-logit-scale null was a global scalar logit scale (multiplicative on all scores); 228 is a per-block scalar on the ALiBi portion only.

## Mechanism
```
# baseline ALiBi (175):
alibi = m_h * (i - j).unsqueeze(0)             # [1, H, 1, T]
scores = Q @ K^T / sqrt(d_k) + alibi

# 228:
alpha_l = self.alibi_per_layer_mul[l]            # [n_layers], init 1.0
scores = Q @ K^T / sqrt(d_k) + alpha_l * alibi  # per-block ALiBi strength
```

Init: `alpha_l = 1.0` for all layers ⇒ step-0 bit-identical to closed ALiBi (175) baseline. The optimizer can shrink any layer's ALiBi contribution toward 0 or amplify it.

## Design sketch
- **Files**: `models/layers.py` — locate the ALiBi bias application. Add an `nn.Parameter(n_layers)` initialized to 1.0. Multiply the ALiBi bias by `alpha_l[block_idx]` before adding to scores.
- **Config flag**: `use_per_layer_alibi_mul: bool = False`, `alibi_mul_init: float = 1.0`.
- **Cost**: 12 scalars per model. +12 params, +0.0013%. Free.
- **Why it should help at tiny1m3m**: at 175-alibi-slopes WIN, the ALiBi slope is fixed per-head and constant across layers. In practice, late layers in deep LMs often have *weaker* positional bias (they attend more content-based) — but the closed 175 doesn't allow this. 228 lets the model learn a depth-dependent schedule: e.g., `[1.0, 1.0, ..., 0.7, 0.5]` (weaker ALiBi in late layers). At 0.94M/12L this might help, especially since the WIN baseline (175) is already a strong starting point.
- **Why it might be null**: the closed ALiBi challengers (208, 213, 214, 215, 216) all tried different ALiBi variants and 4 of them nulled; the lever may already be saturated at 0.94M. Per-block multiplier on top of the WIN may not have headroom.

## Scale evidence
ALiBi (Press et al. 2022) validated at 80M-1.5B; closed 175-alibi-slopes WIN at 0.94M is direct prior. Per-layer ALiBi scaling is novel but motivated by the well-known observation that early/late layers have different positional-bias needs. Transfer-risk **low** (extends a known WIN, source scale above 0.94M).

## Why it's worth a slot
A win would say the depth-dependent ALiBi schedule helps on top of the existing 175-alibi-slopes WIN. A null would close the per-layer-multiplier-on-existing-WIN axis at 0.94M. The lever is essentially free (+12 params, ~10 LoC), bit-identical at init=1.0, and composes cleanly with 175 (a known winner) — the only slot-style idea that *adds to* an existing winner rather than replacing or duplicating it.
