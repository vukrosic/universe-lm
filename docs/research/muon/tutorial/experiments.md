# Muon ablations — experiment manifest

Run status for [../plan.md](../plan.md). `status` ∈ {TODO, wired, tiny-done,
screen-running, screen-done, dropped}. Control = `Screen10M20MConfig` default Muon
(`muon_lr=0.024, momentum=0.95, ns_steps=5`) → 4.7984 (`s_ctrl_full`).

⚠️ LR-sensitive levers (marked) need a `muon_lr` sweep — report best LR per arm.

## Batch 1 — orthogonalization

| # | Name | Config class | knob | LR sweep | status |
|---|---|---|---|---|---|
| M1 | NSStepsSweep | `Screen10M20MNSSteps{1,2,3}Config` | `muon_ns_steps` | no | TODO |
| M2 | NoOrtho | `Screen10M20MNoOrthoConfig` | `muon_orthogonalize=False` | **yes** | TODO |
| M3 | NSCoeffs | `Screen10M20MNSCoeffsConfig` | `muon_coeffs=ns_quintic` | **yes** | TODO |
| M4 | OrthoDtypeFp32 | `Screen10M20MOrthoFp32Config` | `muon_ortho_dtype=fp32` | no | TODO |

## Batch 2 — shape-scaling

| # | Name | Config class | knob | LR sweep | status |
|---|---|---|---|---|---|
| M5 | NoShapeScale | `Screen10M20MNoShapeScaleConfig` | `muon_shape_scale=none` | **yes** | TODO |
| M6 | SpectralScale | `Screen10M20MSpectralScaleConfig` | `muon_shape_scale=spectral` | **yes** | TODO |
| M7 | RMSMatchScale | `Screen10M20MRMSMatchScaleConfig` | `muon_shape_scale=rms_match` | **yes** | TODO |

## Batch 3 — momentum / nesterov

| # | Name | Config class | knob | status |
|---|---|---|---|---|
| M8 | MomentumSweep | `Screen10M20MMuonMom{90,99}Config` | `muon_momentum` | TODO |
| M9 | NesterovOff | `Screen10M20MNesterovOffConfig` | `muon_nesterov=False` | TODO |

## Batch 4 — routing

| # | Name | Config class | knob | status |
|---|---|---|---|---|
| M10 | EmbedToMuon | `Screen10M20MEmbedToMuonConfig` | `muon_route_embedding=True` | TODO |
| M11 | PerGroupMuonLR | `Screen10M20MPerGroupMuonLRConfig` | `muon_lr_attn / muon_lr_ffn` | TODO |
| M12 | LRRatioSweep | (use `--muon_lr/--adamw_lr`) | LR ratio | TODO |

## Batch 5 — compute savings

| # | Name | Config class | knob | status |
|---|---|---|---|---|
| M13 | LazyOrtho | `Screen10M20MLazyOrthoConfig` | `muon_ortho_every=N` | TODO |
| M14 | Bf16Buffer | `Screen10M20MBf16BufferConfig` | `muon_buffer_dtype=bf16` | TODO |

## Per-experiment checklist (tick before screen-done)

- [ ] change guarded by a flag; default Muon path untouched when off
- [ ] LR-sensitive levers: muon_lr swept (≥3), best-LR arm reported
- [ ] screen 3-seed (42/43/44) at chosen LR, mean + std in results.md
- [ ] **wall-clock recorded** (`total_time_minutes`) — speed levers live or die here
- [ ] metrics.json committed, evidence index regenerated
