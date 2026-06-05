# Query ablations — experiment manifest

Run status for [../plan.md](../plan.md). Implementing AI: flip status as you go.
`status` ∈ {TODO, wired, tiny-done, screen-running, screen-done, dropped}.

**Code state (2026-06-05): all Q1–Q29 levers are `wired`** — every flag is guarded
in `MultiHeadAttention.__init__` ([models/layers.py:245](../../../../models/layers.py))
and every `Screen10M20M<Name>Config` exists in
[configs/llm_config.py](../../../../configs/llm_config.py). **No runs yet** — every row
below is implemented and launchable but unrun, so [results.md](results.md) stays
`pending`. Launch a lever with:

```bash
python train_llm.py --config_class configs.llm_config.Screen10M20M<Name>Config --seed 42
```

## Batch 1

| # | Name | Config class | Flag added | step-0==base | status |
|---|---|---|---|---|---|
| — | control | `Screen10M20MConfig` | — | — | done (4.7984, `s_ctrl_full`) |
| Q1 | AlibiBias | `Screen10M20MAlibiBiasConfig` | `use_alibi_bias` | yes (m=0) | wired |
| Q2 | QTempToken | `Screen10M20MQTempTokenConfig` | `use_q_temp_token` | yes (w=0) | wired |
| Q3 | CosineAttn | `Screen10M20MCosineAttnConfig` | `use_cosine_attn` | ~yes | wired |
| Q4 | QKBilinear | `Screen10M20MQKBilinearConfig` | `use_qk_bilinear` | yes (d=1) | wired |

A/B to record: Q1 vs shipped `attn_sink`.

## Batch 2

| # | Name | Config class | Flag | step-0==base | status |
|---|---|---|---|---|---|
| Q5 | TalkingHeadsQ | `Screen10M20MTalkingHeadsQConfig` | `use_talking_heads_q` | yes (M=I) | wired |
| Q6 | PerHeadRopeBase | `Screen10M20MPerHeadRopeBaseConfig` | `use_per_head_rope_base` | yes | wired |
| Q7 | PartialRotary | `Screen10M20MPartialRotaryConfig` | `partial_rotary_p` | yes (p=1) | wired |

## Batch 3 (gated)

| # | Name | Config class | Flag | status |
|---|---|---|---|---|
| Q8 | QExpansion | `Screen10M20MQExpansionConfig` | `use_q_expansion` | wired |
| Q9 | DecoupledContentPos | `Screen10M20MDecoupledContentPosConfig` | `use_decoupled_content_pos` | wired |
| Q10 | AntisymQK | `Screen10M20MAntisymQKConfig` | `use_antisym_qk` | wired |

## Batch 4 — query-norm zoo (`q_norm_type` flag now wired)

| # | Name | Config class | Query norm | status |
|---|---|---|---|---|
| Q11 | QNormPnorm15 | `Screen10M20MNormPNormConfig` | `pnorm1.5` | wired |
| Q12 | QNormClip | `Screen10M20MNormClipConfig` | `clipnorm3` | wired |
| Q13 | QNormChannelScale | `Screen10M20MNormChannelScaleConfig` | `channelscale` | wired |
| Q14 | QNormManhattan | `Screen10M20MNormManhattanConfig` | `manhattan` | wired |
| Q15 | QNormCenter | `Screen10M20MNormCenterConfig` | `center` | wired |
| Q16 | QNormNone | `Screen10M20MNormNoneConfig` | identity | wired |

## Batch 5 — learnable-parameter zoo

| # | Name | Config class | Flag | step-0==base | status |
|---|---|---|---|---|---|
| Q17 | QBiasHead | `Screen10M20MQPerHeadBiasConfig` | `use_q_per_head_bias` | yes | wired |
| Q18 | QGainChannel | `Screen10M20MQPerChannelGainConfig` | `use_q_per_channel_gain` | yes (g=0) | wired |
| Q19 | QGainHeadChannel | `Screen10M20MQHDGainConfig` | `use_q_hd_gain` | yes (G=0) | wired |
| Q20 | QGateNorm | `Screen10M20MQNormGateConfig` | `use_q_norm_gate` | yes (gate≈1) | wired |
| Q21 | QResidualLowRank | `Screen10M20MQLowRankRefineConfig` | `use_q_lowrank_refine` | yes (U=0) | wired |
| Q22 | QLayerScale | `Screen10M20MQLayerScaleConfig` | `use_q_layerscale` | yes | wired |
| Q23 | QSoftplusGain | `Screen10M20MQSoftplusGainConfig` | `use_q_softplus_gain` | yes | wired |

## Batch 6 — query architecture / mixing

| # | Name | Config class | Flag | step-0==base | status |
|---|---|---|---|---|---|
| Q24 | QHeadMix | `Screen10M20MQHeadMixConfig` | `use_q_head_mix` | yes (M=I) | wired |
| Q25 | QTimeConv | `Screen10M20MQTimeConvConfig` | `use_q_time_conv` | yes (identity tap) | wired |
| Q26 | QEMASmooth | `Screen10M20MQEMASmoothConfig` | `use_q_ema_smooth` | yes (α=0) | wired |
| Q27 | QFeatureMap | `Screen10M20MQFeatureMapConfig` | `use_q_feature_map` | **no** — baseline shifts | wired |
| Q28 | QPerTokenRope | `Screen10M20MQPerTokenRopeConfig` | `use_q_per_token_rope` | yes (0 init) | wired |
| Q29 | QNoiseReg | `Screen10M20MQNoiseRegConfig` | `use_q_noise_reg` | yes (eval clean) | wired |

## Per-experiment checklist (the AI ticks these before marking screen-done)

- [ ] flag guarded with `if self.use_<x>:`, baseline path untouched
- [ ] step-0 val_loss matches clean control (identity/zero-init verified)
- [ ] new params confirmed receiving gradient under the right optimizer
- [ ] tiny tier run (kill if clearly washing)
- [ ] screen 3-seed (42/43/44) run, mean + std recorded in results.md
- [ ] `metrics.json` committed, evidence index regenerated
