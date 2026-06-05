# Query-tweaks manifest (29-experiment implementation checklist)

This is the **implementation mirror** of [plan.md](plan.md). Each row
here is a single experiment that needs:

1. A flag in `configs/llm_config.py` (defaults to off / identity-init).
2. A mechanism in `models/layers.py` `MultiHeadAttention`.
3. A `Screen10M20M<Name>Config` recipe in `configs/llm_config.py`.

After all 29 are wired, the breadth screens (Batches 4–6) run on
`tiny1m` first; top ~2 per batch promote to `screen20m` 3-seed.

---

## Status legend

- [ ] not started
- [~] in progress
- [x] wired (flag + mechanism + recipe)
- [✓] run complete, verdict logged
- [-] closed (closed with explanation)

---

## Batch 1 — high-signal levers (Q1–Q4)

| # | Flag | Mechanism | Recipe | Status |
|---|---|---|---|---|
| Q1 | `use_alibi_bias` | per-head learnable slope on `(i−j)` | `Screen10M20MAlibiBiasConfig` | [ ] |
| Q2 | `use_q_temp_token` | per-head `Q *= (1+tanh(x·w_h))` | `Screen10M20MQTempTokenConfig` | [ ] |
| Q3 | `use_cosine_attn` | L2-norm Q,K + per-head learnable τ | `Screen10M20MCosineAttnConfig` | [ ] |
| Q4 | `use_qk_bilinear` | per-channel `diag(d_h)` on score | `Screen10M20MQKBilinearConfig` | [ ] |

## Batch 2 — flagship + positional (Q5–Q7)

| # | Flag | Mechanism | Recipe | Status |
|---|---|---|---|---|
| Q5 | `use_talking_heads_q` | learned `n_h × n_h` M on logits pre-softmax | `Screen10M20MTalkingHeadsQConfig` | [ ] |
| Q6 | `use_per_head_rope_base` | per-head learnable rotary base | `Screen10M20MPerHeadRopeBaseConfig` | [ ] |
| Q7 | `partial_rotary_p` | fraction p of Q/K dims rotated | `Screen10M20MPartialRotaryConfig` | [ ] |

## Batch 3 — exotic (Q8–Q10)

| # | Flag | Mechanism | Recipe | Status |
|---|---|---|---|---|
| Q8 | `use_q_expansion` | 2× Q heads, 2 reads, mean | `Screen10M20MQExpansionConfig` | [ ] |
| Q9 | `use_decoupled_content_pos` | two score streams summed | `Screen10M20MDecoupledContentPosConfig` | [ ] |
| Q10 | `use_antisym_qk` | learnable skew S on Q^T S K | `Screen10M20MAntisymQKConfig` | [ ] |

## Batch 4 — query-norm zoo (Q11–Q16)

**Prereq wire:** add `q_norm_type` flag (defaults to `qk_norm_type`).

| # | Config (sets `q_norm_type=...`) | Mechanism | Status |
|---|---|---|---|
| Q11 | `Screen10M20MNormPNormConfig` | "pnorm1.5" | [ ] |
| Q12 | `Screen10M20MNormClipConfig` | "clipnorm3" | [ ] |
| Q13 | `Screen10M20MNormChannelScaleConfig` | "channelscale" | [ ] |
| Q14 | `Screen10M20MNormManhattanConfig` | "manhattan" | [ ] |
| Q15 | `Screen10M20MNormCenterConfig` | "center" | [ ] |
| Q16 | `Screen10M20MNormNoneConfig` | "none" (skip Q norm) | [ ] |

## Batch 5 — learnable-param zoo (Q17–Q23)

| # | Flag | Mechanism | Recipe | Status |
|---|---|---|---|---|
| Q17 | `use_q_per_head_bias` | `Q += b_h` after RoPE | `Screen10M20MQPerHeadBiasConfig` | [ ] |
| Q18 | `use_q_per_channel_gain` | `Q *= g_d` per-channel | `Screen10M20MQPerChannelGainConfig` | [ ] |
| Q19 | `use_q_hd_gain` | `Q *= g_hd` head×channel | `Screen10M20MQHDGainConfig` | [ ] |
| Q20 | `use_q_norm_gate` | `Q *= σ(a_h·‖x‖+b_h)` | `Screen10M20MQNormGateConfig` | [ ] |
| Q21 | `use_q_lowrank_refine` | `Q += W1·x·W2` (zero-init) | `Screen10M20MQLowRankRefineConfig` | [ ] |
| Q22 | `use_q_layerscale` | `Q *= (1 + ls_d)` per-channel | `Screen10M20MQLayerScaleConfig` | [ ] |
| Q23 | `use_q_softplus_gain` | `Q *= softplus(g_h)` per-head | `Screen10M20MQSoftplusGainConfig` | [ ] |

## Batch 6 — architecture / mixing (Q24–Q29)

| # | Flag | Mechanism | Recipe | Status |
|---|---|---|---|---|
| Q24 | `use_q_head_mix` | `Q ← Q + Q·M` (M=I init) pre-attn | `Screen10M20MQHeadMixConfig` | [ ] |
| Q25 | `use_q_time_conv` | `Q += conv1d(Q, k=3)` zero-init | `Screen10M20MQTimeConvConfig` | [ ] |
| Q26 | `use_q_ema_smooth` | `Q ← α·Q + (1−α)·Q_prev` | `Screen10M20MQEMASmoothConfig` | [ ] |
| Q27 | `use_q_feature_map` | **NOT identity-init**, own control | `Screen10M20MQFeatureMapConfig` | [ ] |
| Q28 | `use_q_per_token_rope` | per-token θ via small MLP | `Screen10M20MQPerTokenRopeConfig` | [ ] |
| Q29 | `use_q_noise_reg` | `Q += N(0, σ²)` training only | `Screen10M20MQNoiseRegConfig` | [ ] |

---

## Counts

- **Total experiments:** 29
- **Full mechanism implementations:** 23 (batches 1, 2, 3, 5, 6)
- **Routing-only (config sets a flag):** 6 (batch 4)
- **Net per-experiment work:** 1 flag + 1 forward-pass branch +
  1 Screen10M20M<Name>Config class

## Batches 4–6 breadth-screen policy

Batches 4, 5, 6 are explicitly **breadth screens**:

- Run on `tiny1m` first (each run ~2 min on RTX 3050).
- Promote top ~2 per batch to a 3-seed `screen20m` run.
- **Net cost: 23 tiny runs + 6 screen20m × 3 seeds = 41 runs.**
- vs full-cost (29 × 3 seeds = 87 screen20m runs).
- **Saves ~47 runs** at the cost of one extra tiny-batch round.
