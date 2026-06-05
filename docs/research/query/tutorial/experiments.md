# Query ablations — experiment manifest

Run status for [../plan.md](../plan.md). Implementing AI: flip status as you go.
`status` ∈ {TODO, wired, tiny-done, screen-running, screen-done, dropped}.

## Batch 1

| # | Name | Config class | `--config` key | Flag(s) added | step-0==base | status |
|---|---|---|---|---|---|---|
| — | control | `Screen10M20MConfig` | (existing) | — | — | done (4.7984, `s_ctrl_full`) |
| Q1 | AlibiBias | `Screen10M20MAlibiBiasConfig` | TBD | `use_alibi_bias` | yes (m=0) | TODO |
| Q2 | QTempToken | `Screen10M20MQTempTokenConfig` | TBD | `use_q_temp_token` | yes (w=0) | TODO |
| Q3 | CosineAttn | `Screen10M20MCosineAttnConfig` | TBD | `use_cosine_attn` | ~yes | TODO |
| Q4 | QKBilinear | `Screen10M20MQKBilinearConfig` | TBD | `use_qk_bilinear` | yes (d=1) | TODO |

A/B to record: Q1 vs shipped `attn_sink`.

## Batch 2

| # | Name | Config class | Flag(s) | step-0==base | status |
|---|---|---|---|---|---|
| Q5 | TalkingHeadsQ | `Screen10M20MTalkingHeadsQConfig` | `use_talking_heads_q` | yes (M=I) | TODO |
| Q6 | PerHeadRopeBase | `Screen10M20MPerHeadRopeBaseConfig` | `use_per_head_rope_base` | yes | TODO |
| Q7 | PartialRotary | `Screen10M20MPartialRotaryConfig` | `rotary_fraction` | yes (p=1) | TODO |

## Batch 3 (gated)

| # | Name | status |
|---|---|---|
| Q8 | QExpansion | TODO |
| Q9 | DecoupledContentPos | TODO |
| Q10 | AntisymQK | TODO |

## Batch 4 — query-norm zoo (needs `q_norm_type` flag first)

| # | Name | Query norm | status |
|---|---|---|---|
| Q11 | QNormPnorm15 | `pnorm1.5` | TODO |
| Q12 | QNormClip | `clipnorm3` | TODO |
| Q13 | QNormChannelScale | `channelscale` | TODO |
| Q14 | QNormManhattan | `manhattan` | TODO |
| Q15 | QNormCenter | `center` | TODO |
| Q16 | QNormNone | identity | TODO |

## Batch 5 — learnable-parameter zoo

| # | Name | Flag | step-0==base | status |
|---|---|---|---|---|
| Q17 | QBiasHead | `use_q_bias_head` | yes | TODO |
| Q18 | QGainChannel | `use_q_gain_channel` | yes (g=0) | TODO |
| Q19 | QGainHeadChannel | `use_q_gain_hc` | yes (G=0) | TODO |
| Q20 | QGateNorm | `use_q_gate_norm` | yes (gate≈1) | TODO |
| Q21 | QResidualLowRank | `use_q_residual_lr` | yes (U=0) | TODO |
| Q22 | QLayerScale | `use_q_layerscale` | yes | TODO |
| Q23 | QSoftplusGain | `q_gain_param=softplus` | yes | TODO |

## Batch 6 — query architecture / mixing

| # | Name | Flag | step-0==base | status |
|---|---|---|---|---|
| Q24 | QHeadMix | `use_q_head_mix` | yes (M=I) | TODO |
| Q25 | QTimeConv | `use_q_time_conv` | yes (identity tap) | TODO |
| Q26 | QEMASmooth | `use_q_ema` | yes (α=0) | TODO |
| Q27 | QFeatureMap | `q_feature_map` | **no** — baseline shifts | TODO |
| Q28 | QPerTokenRope | `use_q_per_token_rope` | yes (0 init) | TODO |
| Q29 | QNoiseReg | `q_noise_std` | yes (eval clean) | TODO |

## Per-experiment checklist (the AI ticks these before marking screen-done)

- [ ] flag guarded with `if self.use_<x>:`, baseline path untouched
- [ ] step-0 val_loss matches clean control (identity/zero-init verified)
- [ ] new params confirmed receiving gradient under the right optimizer
- [ ] tiny tier run (kill if clearly washing)
- [ ] screen 3-seed (42/43/44) run, mean + std recorded in results.md
- [ ] `metrics.json` committed, evidence index regenerated
