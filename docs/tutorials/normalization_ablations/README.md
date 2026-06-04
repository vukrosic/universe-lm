# Normalization ablations: what works, what breaks

This is the short companion to [`docs/norm-findings-0604.md`](/Users/vukrosic/my-life/llm-research-kit-scaling/docs/norm-findings-0604.md).

For the teaching version that explains how to add and evaluate a new norm, see [`docs/tutorials/normalization_research/README.md`](/Users/vukrosic/my-life/llm-research-kit-scaling/docs/tutorials/normalization_research/README.md).

Research question:

**Can we replace RMSNorm with a norm that is slightly more robust to outlier channels, without losing scale?**

## Full-stack result

On the tiny1m stack we already verified:

| ablation | val_loss | read |
|---|---:|---|
| pnorm1.5 | 6.3063 | best result so far |
| pnorm1 / L1 | 6.3156 | strong |
| channelscale | 6.3253 | strong |
| RMSNorm / pnorm2 | 6.3563 | baseline |
| layernorm | 6.3487 | about baseline |
| clipnorm3 | 6.3563 | no real gain |
| peak / L-infty | 6.4266 | too peaky |
| median | 6.8597 | too robust |
| squash / DyT | 7.6278 | diverged |
| center only | NaN | diverged |

The short version from the full stack:

- **A mild robust norm wins.**
- **Too much robustness loses information.**
- **No division at all is unstable.**

## Clean-baseline result

`norm4` finished on the GPU. This sweep removed the architecture helpers:

- no SWA
- no value embedding
- no q-gain
- plain full attention

That makes it a cleaner test of normalization itself.

| ablation | val_loss | read |
|---|---:|---|
| layernorm | 6.3628 | best clean-baseline run |
| body + QK pnorm1.5 | 6.3922 | attention placement helps |
| body + V pnorm1.5 | 6.4025 | value placement helps |
| pnorm1.75 | 6.4088 | best plain p-norm |
| pnorm1.375 | 6.4091 | close |
| pnorm1.25 | 6.4125 | close |
| pnorm1.0 | 6.4259 | beats RMSNorm |
| pnorm1.625 | 6.4328 | beats RMSNorm |
| pnorm1.5 | 6.4387 | beats RMSNorm, but not best here |
| RMSNorm | 6.4516 | clean baseline |

Read:

- The robust-p-norm idea still survives: every tested p-norm beat RMSNorm.
- The exact optimum moved: pnorm1.5 was best in the richer stack, but pnorm1.75 was best in the plain baseline.
- Adding pnorm1.5 inside QK or V helped much more than body-only pnorm1.5 on the clean baseline.
- LayerNorm winning here says centering may matter when the architecture is stripped down.

## Why pnorm1.5 works

Transformer activations are not nicely Gaussian. A few channels become very large and behave like learned biases.

Under RMSNorm:

```text
denom = sqrt(mean(x^2))
```

those large channels dominate the denominator, which shrinks the rest of the vector too much.

pnorm1.5 softens that:

```text
denom = mean(|x|^1.5)^(1/1.5)
```

It still divides by a smooth global magnitude, but it is less hostage to a few extreme channels.

On the full stack, that seems to be the sweet spot:

- p = 2 is a bit too sensitive to outliers.
- p = 1 is better, but slightly underfits the scale.
- p = 1.5 lands between them and is the best full-stack norm we have seen.

## Failed ablations

### Peak / median / clip

These try to be "more robust," but robustness is not free.

- **peak** only looks at one channel, so it throws away too much of the vector.
- **median** is so robust that it forgets useful scale information.
- **clipnorm3** trims outliers, but the outliers appear to be functional, not noise.

So the lesson is not "remove outliers completely."
It is "down-weight them gently."

### Centering or squashing without division

These break the basic normalization contract.

- **center only** can go unstable.
- **squash / DyT** also diverges in this setup.

The model needs a stable divide-by-magnitude step.

## Practical takeaway

**RMSNorm is a decent default, but the best norm depends on the stack: mildly robust p-norms help in the richer recipe, while LayerNorm is the stable winner in the clean baseline. Attention-side placement is still worth taking seriously.**

## Paired-seed check

The follow-up sweep on seeds 43/44 sharpened the story:

| norm | seed 43 | seed 44 | read |
|---|---:|---:|---|
| layernorm | 6.3594 | 6.3644 | stable best |
| rmsnorm | 6.3931 | 6.3953 | baseline |
| pnorm1.75 | 6.4019 | 6.4013 | stable, but behind LayerNorm |
| pnorm1.5 | 6.3963 | 6.5822 | seed 44 was shorter and weaker; rerun needed |

The important update is that the clean-baseline winner is still not pnorm1.5.
LayerNorm stayed best on both seeds, and pnorm1.75 stayed competitive without
overtaking it.

The placement follow-up still looks interesting:

| ablation | val_loss | read |
|---|---:|---|
| body + QK pnorm1.5 | 6.3928 | promising |
| body + V pnorm1.5 | 6.4287 | weaker |

That makes the practical rule a little more specific:

- Mild robustness helps.
- The best p can change when the architecture is stripped down.
- Placement can matter as much as the norm itself.
- You still need the second seed before promoting a winner.

## Latest partial sweep

The GPU was turned off before the next clean-baseline candidate list could
finish, so this is only a single-seed follow-up on seed 45.

| ablation | val_loss | val_acc | read |
|---|---:|---:|---|
| channelscale | 6.3725 | 0.1447 | best single-seed result in the halted sweep |
| layernorm | 6.4000 | 0.1427 | still strong |
| rmsnorm | 6.4131 | 0.1430 | baseline |
| pnorm1.6 | 6.4278 | 0.1385 | worse than LayerNorm and RMSNorm here |
| manhattan | 6.4400 | 0.1417 | no gain |
| centeredl1 | 6.4472 | 0.1420 | no gain |

This is the new lead:

- `channelscale` is the best seed-45 result we pulled back.
- It beats both `LayerNorm` and `RMSNorm` on that seed.
- It still needs a second seed before we claim it is generally better.
