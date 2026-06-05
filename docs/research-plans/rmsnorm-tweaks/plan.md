# RMSNorm micro-tweaks (plan)

Base op: `y = g · x / rms(x)`. Each idea is one extra param / a free trick. Most start as exact baseline. Wire in [models/layers.py](../../../models/layers.py) (`make_norm`) + a `Screen10M20M<Name>Config`.

| Idea | What | Extra params | Step-0 == base? | Conf | Why |
|---|---|---|---|---|---|
| Reparam gain | `g = 1 + g₀`, learn g₀ (g₀=0 init) | 0 | yes | med | Symmetric gradient around 0; matches zero-init pattern. Learn deviation from identity, cleaner landscape. Free. |
| RMSNorm + bias | `y = g·x_norm + b` (b=0 init) | +d_model/block | yes | med-high | The bias classic RMSNorm deliberately dropped. Per-block learnable residual offset. Tests if it was load-bearing at depth 24. |
| Global temperature | `y = g·x / (rms(x)·τ)`, scalar τ=1 | +1 | yes | med | One scalar magnitude knob decoupled from per-channel gain. 1-param cousin of LayerScale. |
| Partial-norm mix | `y = g·x_norm + λ·x`, scalar λ=0 | +1 | yes | med-high | Keeps a fraction of un-normalized signal. Additive interp between norm and identity (simpler than ManifoldNorm fractional power). |
| Learnable floor | `rms = √(mean(x²) + c²)`, scalar c≈0 | +1 | ~yes | low-med | Soft dead-zone: low-energy tokens not blown up to unit norm. Stabilizes rare near-zero residuals. |
| Scaled gain init | init `g = 0.5` not 1.0 | 0 | no | low-med | Manual LayerScale via init only. Smaller starting branch magnitude → cleaner deep signal prop. One line. |
| Softplus gain | `g = softplus(w)`, w set so g≈1 | 0 | yes | low | Strictly-positive scales, no sign flips. Marginal optimization smoothing. |

Cheapest high-signal: Reparam gain (free), Partial-norm mix (1 param, real expressivity), RMSNorm+bias (the dropped classic).

## Existing norm variants (already in repo)
PeakNorm, ManhattanNorm, SquashNorm, CenterNorm, ManifoldNorm, PNorm, CenteredL1Norm, ClipNorm, ChannelScaleNorm, MedianNorm — see `make_norm` in [models/layers.py](../../../models/layers.py).

## Status / results
(add per-idea notes, branches, A/B numbers here)
