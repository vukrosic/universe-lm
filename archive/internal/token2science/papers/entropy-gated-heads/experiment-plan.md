# Entropy-Gated Attention Heads - experiment plan

For the implementing AI. This is the full runnable experiment suite for the
paper on entropy-gated attention heads.

## Setup

- Tier for the first pass: `Tiny1M3MConfig`, seed `42`, `3M` tokens, about `4`
  minutes on a V100.
- Tiny baseline reference: validation loss `6.4216`.
- Core mechanism: `gate_h = 1 - alpha_h * normalized_attention_entropy_h`,
  where `alpha_h` is a per-head learnable scalar with zero init.
- Config convention: one subclass per variant, one flag per subclass, baseline
  path untouched when the flag is off.
- Claim rule: tiny runs are for screening only. A result is live only if it
  clears the control bracket by at least `0.01` validation loss and the sign
  survives the next tier.

## 8-Class teaching arc

Each class answers one question, isolates one axis, and feeds the next class.
Run them in order.

### Class 1 - Core probe + variance

| Field | Plan |
| --- | --- |
| Question | Does entropy gating beat the normal run-to-run noise, or is the tiny win just drift? |
| Variants / configs | `Tiny1M3MConfig` control A, `Tiny1M3MEntropyGateConfig`, `Tiny1M3MConfig` control B. Same seed `42` for all three. |
| Measure | Final validation loss, mean of the two controls, control spread, train loss curve, learned alpha mean and std, gate histogram, throughput. |
| Expected signal | Controls should stay near `6.4216`. The gated run should move below the control bracket if the mechanism is real. If it lands inside the bracket, treat the effect as unresolved noise. |

### Class 2 - Gate-function ablation

| Field | Plan |
| --- | --- |
| Question | What gate shape is actually useful: linear decay, sigmoid, exponential decay, or a centered reparam? |
| Variants / configs | `Tiny1M3MEntropyGateLinearConfig` with `gate = 1 - alpha_h * h`; `Tiny1M3MEntropyGateSigmoidConfig` with `gate = 2 * sigmoid(z_h)` so step 0 is identity; `Tiny1M3MEntropyGateExpConfig` with `gate = exp(-lambda_h * h)`; `Tiny1M3MEntropyGateAttenuateOnlyConfig` with `gate = clamp(1 - softplus(alpha_h) * h, 0, 1)`; `Tiny1M3MEntropyGateAmpAttenConfig` with `gate = 1 + beta_h * (1 - h)`; `Tiny1M3MEntropyGateCenteredEntropyConfig` with centered entropy `h - mean(h)` before the gate. |
| Measure | Final validation loss, alpha or lambda distribution, gate range, fraction of heads clamped or saturated, training stability, whether the gate ever leaves the identity basin. |
| Expected signal | Linear and exponential should be the main contenders. Sigmoid may be smoother but can be too blunt. Centering should help if different layers live on different entropy scales. Attenuate-only is the conservative control; amplify-and-attenuate should help only if some heads need to be boosted relative to the layer mean. |

### Class 3 - Granularity

| Field | Plan |
| --- | --- |
| Question | Is the right level of control global, per-layer, per-head, or per-head-per-layer? |
| Variants / configs | `Tiny1M3MEntropyGateGlobalAlphaConfig` with one alpha for the whole model; `Tiny1M3MEntropyGatePerLayerAlphaConfig` with one alpha per layer; `Tiny1M3MEntropyGatePerHeadAlphaConfig` with one alpha per head, shared across layers; `Tiny1M3MEntropyGatePerHeadPerLayerAlphaConfig` with one alpha per head per layer. |
| Measure | Final validation loss, parameter count, alpha sparsity, alpha rank by layer/head, whether any variant collapses to near-zero gating. |
| Expected signal | If the phenomenon is head-specific, per-head should beat global and per-layer. Per-head-per-layer may win only if the signal is sharply localized enough to justify the extra parameters. |

### Class 4 - Placement

| Field | Plan |
| --- | --- |
| Question | Where should the gate act: on head context output, on the residual add, or on pre-softmax logits? |
| Variants / configs | `Tiny1M3MEntropyGateHeadOutConfig` gating the per-head context output before head merge; `Tiny1M3MEntropyGateResidualAddConfig` gating after the output projection at the residual add; `Tiny1M3MEntropyGatePreSoftmaxConfig` applying the same entropy signal to the score branch before softmax. |
| Measure | Final validation loss, gradient norms, train stability, gate distribution, entropy shift before and after gating, throughput cost. |
| Expected signal | Head-output gating should be the default sweet spot because it is close to the mechanism and still local to the head. Residual-add gating may be weaker because the signal is diluted after merge. Pre-softmax gating could help if entropy is really a routing signal, but it is the most coupled and likely the most fragile. |

### Class 5 - Mechanistic analysis I

| Field | Plan |
| --- | --- |
| Question | Which heads learn large alpha, and does alpha track measured head entropy? |
| Variants / configs | Use the best winner from Classes 2 to 4, plus the plain `Tiny1M3MConfig` control. If the linear per-head head-output gate wins, this is `Tiny1M3MEntropyGateConfig`. Otherwise, substitute the actual winning gate variant verbatim. |
| Measure | Per-head alpha histogram, per-layer alpha histogram, Pearson and Spearman correlation between learned alpha and measured ungated head entropy, top-k gated heads, bottom-k gated heads, alpha drift over training. |
| Expected signal | Alpha should be largest on diffuse heads with high entropy and smallest on sharp heads. If the correlation is near zero or negative, the gate is not learning the intended ranking. |

### Class 6 - Mechanistic analysis II

| Field | Plan |
| --- | --- |
| Question | Do the gated heads look positional, diffuse, sinky, or broadly unstructured? |
| Variants / configs | Same analysis config as Class 5, again with the plain control for comparison. Dump validation attention maps for the top-gated heads, middle-gated heads, and low-gated heads. |
| Measure | Attention map grids, diagonal mass, average attention distance, sink mass, peakiness, entropy overlays, and side-by-side head visualizations for high-alpha versus low-alpha heads. |
| Expected signal | The strongest-gated heads should tend to be diffuse, positional, or sink-like rather than crisp content heads. Sharp content heads should usually stay near the baseline gate. |

### Class 7 - Interaction and composition

| Field | Plan |
| --- | --- |
| Question | Does entropy gating add to FIRE and Canon, or does it mostly duplicate them? |
| Variants / configs | FIRE control: `Tiny1M3MConfig` with `use_fire_pe=True`. Stack: `Tiny1M3MEntropyGateConfig` with `use_fire_pe=True`. Canon control: `Tiny1M3MConfig` with `use_canon_conv=True`. Stack: `Tiny1M3MEntropyGateConfig` with `use_canon_conv=True`. |
| Measure | Standalone delta for FIRE, standalone delta for Canon, stacked delta, interaction terms `I_fire = L(FIRE+Entropy) - L(FIRE) - L(Entropy) + L(base)` and `I_canon = L(Canon+Entropy) - L(Canon) - L(Entropy) + L(base)`, train stability, and any throughput penalty from the extra logging. |
| Expected signal | The gate should be at least partly additive with Canon because Canon changes the residual path while entropy gating is attention-local. FIRE is the more likely source of overlap because both live in the attention pathway and may compete to explain the same sharpness signal. |

### Class 8 - Scale transfer

| Field | Plan |
| --- | --- |
| Question | Does the tiny-tier gain survive the `20M` token screen and then the larger ladder? |
| Variants / configs | Mirror the tiny winner into `Screen10M20MEntropyGateConfig` and `Full10M200MEntropyGateConfig`, plus the matching `Screen10M20MConfig` and `Full10M200MConfig` controls. If a different gate variant wins in Classes 2 to 4, mirror that exact variant name instead. |
| Measure | Validation loss delta at each tier, sign consistency from tiny to screen to full, extra parameter count, estimated FLOP cost, wall-clock throughput, memory use, and whether the gain survives the next scale without re-tuning. |
| Expected signal | A real mechanism should keep its sign as scale increases. If the improvement shrinks away or flips at `20M` tokens, the tiny win was likely a local artifact. If the sign survives screen and full, the mechanism is worth paper-level attention. |

## Metrics and rigor note

| Rule | Why |
| --- | --- |
| Use a two-control variance bracket around every candidate | This keeps the tiny result honest and makes the local noise visible. |
| Keep seed `42` fixed for the tiny tier | The first pass is for ranking mechanisms, not for averaging away uncertainty. |
| Use the win-bar `control_mean - 0.01` as the screen threshold | This matches the repo convention that a live lever must beat control by a clear margin, not just by a fraction of a point. |
| Record parameter and FLOP cost for every nontrivial variant | Some granularities buy signal by spending more parameters. The paper needs to say whether the gain is worth that cost. |
| Treat analysis runs as logging passes, not new claims | The mechanistic plots explain the best gate, but they do not upgrade a null into a win. |
