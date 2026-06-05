# RMSNorm ablations — experiment manifest

Run status for [../plan.md](../plan.md). `status` ∈ {TODO, wired, tiny-done,
screen-running, screen-done, dropped}. Control = `Screen10M20MConfig` plain RMSNorm
(4.7984, `s_ctrl_full`). **3-seed mandatory** — norm wins are seed-noisy.

## Batch 1 — free / 1-param (original bank)

| # | Name | Config class | norm_type / knob | step-0==base | status |
|---|---|---|---|---|---|
| N1 | ReparamGain | `Screen10M20MReparamGainConfig` | `reparam_gain` | yes (g₀=0) | TODO |
| N2 | RMSBias | `Screen10M20MRMSBiasConfig` | `rms_bias` | yes (b=0) | TODO |
| N3 | GlobalTemp | `Screen10M20MGlobalTempConfig` | `norm_temp` | yes (τ=1) | TODO |
| N4 | PartialNormMix | `Screen10M20MPartialNormMixConfig` | `partial_norm_lambda` | yes (λ=0) | TODO |
| N5 | LearnableFloor | `Screen10M20MLearnableFloorConfig` | `learnable_floor` | ~yes (c=0) | TODO |
| N6 | ScaledGainInit | `Screen10M20MScaledGainInitConfig` | `gain_init=0.5` | no — own ctrl | TODO |
| N7 | SoftplusGain | `Screen10M20MSoftplusGainConfig` | `softplus_gain` | yes | TODO |

## Batch 2 — NEW structural

| # | Name | Config class | knob | step-0==base | status |
|---|---|---|---|---|---|
| N8 | PartialNormVector | `Screen10M20MPartialNormVectorConfig` | `partial_norm_vector` | yes (λ=0) | TODO |
| N9 | GroupRMS | `Screen10M20MGroupRMSConfig` | `rms_groups=G` | ~yes | TODO |
| N10 | StopGradRMS | `Screen10M20MStopGradRMSConfig` | `stopgrad_rms` | yes | TODO |
| N11 | AsymGain | `Screen10M20MAsymGainConfig` | `asym_gain` | yes | TODO |
| N12 | GainClamp | `Screen10M20MGainClampConfig` | `gain_clamp_a` | yes | TODO |
| N13 | DepthScaledGainInit | `Screen10M20MDepthScaledGainInitConfig` | `depth_scaled_gain` | no — own ctrl | TODO |
| N14 | LearnableEps | `Screen10M20MLearnableEpsConfig` | `learnable_eps` | yes | TODO |

## Batch 3 — NEW replacement probes (gated)

| # | Name | Config class | knob | step-0==base | status |
|---|---|---|---|---|---|
| N15 | DynTanh | `Screen10M20MDynTanhConfig` | `norm_type=dyntanh` | no — own ctrl | TODO |
| N16 | DoubleNorm | `Screen10M20MDoubleNormConfig` | `double_norm` | ~yes | TODO |
| N17 | CenterMix | `Screen10M20MCenterMixConfig` | `center_mix_mu` | yes (μ=0) | TODO |

## Batch 4 — existing norm zoo sweep (no new code)

| norm_type | status |
|---|---|
| pnorm1.5 / pnorm1.75 | TODO |
| clipnorm3 | TODO |
| channelscale | TODO |
| manhattan / center / centeredl1 | TODO |
| manifold / median / peak / squash | TODO |
| layernorm | TODO |

## Per-experiment checklist (tick before screen-done)

- [ ] norm applied at norm1 + norm2 + final self.norm (consistent)
- [ ] step-0 val_loss matches control (or own control if non-identity)
- [ ] new params getting gradient (AdamW)
- [ ] tiny run (kill if washing)
- [ ] screen **3-seed** (42/43/44), mean + std in results.md
- [ ] metrics.json committed, evidence index regenerated
