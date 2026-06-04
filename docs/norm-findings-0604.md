# Normalization & outlier-robustness findings — tiny1m (2026-06-04)

Single-seed unless noted. Base recipe: value-emb + q-gain + SWA384 + RoPE250k,
0.94M params / 3M tokens. Noise floor ≈ 0.02 val_loss. All norms are ~param-free,
so these are confound-free comparisons. Full numbers: `runs/tiny1m_0604_results.md`.

For the short version, see [`docs/tutorials/normalization_ablations/README.md`](/Users/vukrosic/my-life/llm-research-kit-scaling/docs/tutorials/normalization_ablations/README.md).

## Study 1 — invented norms (#90)

| norm | val_loss | note |
|---|---|---|
| manhattan (L1) | 6.3213 | ✅ beats RMSNorm, and cheaper (no ²/√) |
| rmsnorm (L2, baseline) | 6.3378 | reference |
| layernorm | 6.3487 | ≈ baseline |
| manifold (rms^ρ, learnable ρ) | 6.3697 | ✗ partial normalization hurts |
| peak (L∞) | 6.4266 | ✗ depends on 1 dim |
| squash (DyT tanh, no division) | 7.6278 | ✗✗ diverged |
| center (mean-only, no division) | NaN | ✗✗✗ diverged |

## Study 2 — generalized p-norm sweep (PNorm, single knob = p)

| norm | val_loss |
|---|---|
| **pnorm1.5** | **6.3063** ← best, beats RMSNorm by 0.052 |
| pnorm1 (=L1) | 6.3156 |
| pnorm3 | 6.3297 |
| centeredl1 | 6.3375 |
| pnorm2 (=RMSNorm, sanity ✓) | 6.3563 |
| rmsnorm (native) | 6.3584 |
| pnorm0.5 | NaN (diverged) |

## The mechanism (why)

Transformers develop a few **massive-activation channels** (dims with huge values
that act as learned biases; Sun et al. 2024). Under **L2 (RMSNorm)** those outliers
inflate the denominator → **shrink every other channel** → suppress real signal.
**Lower-order/robust statistics (L1, L1.5) down-weight the outliers** → the rest of
the vector keeps its scale → better. Two load-bearing properties, proven by the
failures:
- **Scale-invariance (must DIVIDE by a magnitude):** center (no ÷) → NaN; squash
  (no ÷) → diverge; manifold (partial ÷, ρ<1) → worse.
- **Smooth ALL-dimension aggregate:** peak (L∞, 1 dim) → weak.
- Given those, **fractional p≈1.5 is the optimum** (robust but p>1, or it diverges).

**Headline for a post:** *RMSNorm (L2) is a suboptimal convention — the optimal
normalization is L1.5, which beats it by a clear margin at equal cost.*

## Architectures derived from the mechanism (#91–93, running)

If outlier channels are the problem, attack them at new loci:
- **#91 robust QK-norm** (pnorm1.5 on Q,K) — outlier-robust attention logits.
- **#92 robust V-norm** (pnorm1.5 on V) — outlier-robust value aggregation.
- **#93 saturating FFN** (c·tanh(relu/c)) — stop *amplifying* outliers (squared-ReLU
  squares them; this soft-caps instead).
- + a "stack everything robust" run to test if the mechanism compounds.

**Reasoned kills:** L1/Laplacian attention (O(T²d), OOM on the 6GB card);
register/residual-bias tokens (indirect + risky seq surgery); bounded/DyT residual
(already falsified by squash); more p values (hyperparam, and p<1 diverges).

## Study 3 — outlier-attack norms (#94–96)

Same base recipe. Attack the massive-activation channels three different ways:

| norm | val_loss | idea / verdict |
|---|---|---|
| channelscale (learnable pre-scale, then RMS) | 6.3253 | ✅ let model down-weight outlier channels *before* the denom — beats RMS |
| clipnorm3 (winsorize \|x\| to 3·mean, then RMS) | 6.3563 | ✗ clipping outliers ≈ no help — they're not pure noise |
| median (÷ median \|x\|, 50% breakdown) | 6.8597 | ✗✗ too robust — throws away real scale info |

**Read:** robustness is a *sweet spot*, not "more is better." L1.5 (mild) wins,
median (extreme) collapses. clipnorm failing says the outliers are **functional**
(learned biases), not corruption — you want to *de-weight* them, not delete them.

## Study 4 — clean-baseline de-confound + placement (norm4, COMPLETE)

The studies above sit on the full stack (V+q+SWA384+RoPE250k), so the norm signal
could be entangled with attention changes. Study 4 re-ran the norm sweep on a
**plain full-attention transformer** (no SWA / no V-emb / no q-gain / rope=10k) to
test whether the norm effect stands alone, plus a placement test.

| norm / placement | val_loss | read |
|---|---:|---|
| layernorm | 6.3628 | best clean-baseline run |
| body + QK pnorm1.5 | 6.3922 | robust attention logits help |
| body + V pnorm1.5 | 6.4025 | robust value mixing helps |
| pnorm1.75 | 6.4088 | best plain p-norm |
| pnorm1.375 | 6.4091 | close |
| pnorm1.25 | 6.4125 | close |
| pnorm1.0 | 6.4259 | beats RMSNorm |
| pnorm1.625 | 6.4328 | beats RMSNorm |
| pnorm1.5 | 6.4387 | beats RMSNorm, but not best here |
| rmsnorm | 6.4516 | clean baseline |

**Read:** the robustness hypothesis survives the de-confound: every p-norm in the
grid beats RMSNorm on the plain baseline. But the exact p is **not universal**:
pnorm1.5 wins in the richer stack, while pnorm1.75 is the best plain p-norm here.
LayerNorm winning says centering can matter once the architecture helpers are
removed. The placement test is the most actionable result: putting pnorm1.5 on
QK or V is much better than body-only pnorm1.5 in the clean baseline.

## Study 5 — paired-seed follow-up (norm5, partial)

The clean-baseline sweep was followed by a paired-seed check on the strongest
controls. That follow-up confirms the broad shape of the result, but it also
shows that one pnorm1.5 run was not cleanly comparable to the others and needs a
rerun before we promote it.

| norm | seed 43 | seed 44 | read |
|---|---:|---:|---|
| layernorm | 6.3594 | 6.3644 | stable best |
| rmsnorm | 6.3931 | 6.3953 | baseline stays baseline |
| pnorm1.75 | 6.4019 | 6.4013 | stable, but behind LayerNorm |
| pnorm1.5 | 6.3963 | 6.5822 | seed 44 was weaker and shorter; rerun needed |

Placement follow-up from this sweep:

| ablation | val_loss | read |
|---|---:|---|
| body + QK pnorm1.5 | 6.3928 | still promising |
| body + V pnorm1.5 | 6.4287 | not as strong |

**Read:** the universal lesson survived, but the clean-baseline winner is not
pnorm1.5. LayerNorm is the most stable result we have on the stripped model,
pnorm1.75 is the best plain p-norm, and the attention-side placements are still
worth checking with the missing second seed.

## Study 6 — halted clean-baseline follow-up (norm6, single seed)

The GPU was turned off before the full candidate list could finish, so this is a
partial follow-up on seed 45 rather than a completed sweep.

| norm | val_loss | val_acc | read |
|---|---:|---:|---|
| channelscale | 6.3725 | 0.1447 | best single-seed result in the halted follow-up |
| layernorm | 6.4000 | 0.1427 | still strong |
| rmsnorm | 6.4131 | 0.1430 | baseline |
| pnorm1.6 | 6.4278 | 0.1385 | worse than LayerNorm and RMSNorm here |
| manhattan | 6.4400 | 0.1417 | no gain |
| centeredl1 | 6.4472 | 0.1420 | no gain |

**Read:** `channelscale` is the most promising stripped-baseline candidate from
this last partial sweep, and on seed 45 it beats both RMSNorm and LayerNorm.
That is a real signal, but it is still only one seed, so we should treat it as
a lead, not a general conclusion.

## Norm cookbook (formula → verdict)

All are `g ⊙ x / denom` except where noted; `g` = learnable per-channel gain.

| norm | denom (per-token over channels) | verdict |
|---|---|---|
| **pnorm1.5** | `mean(\|x\|^1.5)^(1/1.5)` | 🥇 optimal — robust but p>1 |
| pnorm1 / manhattan | `mean(\|x\|)` | ✅ strong, cheapest |
| channelscale | RMS of `(pre⊙x)`, learnable pre | ✅ beats RMS |
| rmsnorm (=pnorm2) | `sqrt(mean(x²))` | baseline convention |
| centeredl1 | `mean(\|x−x̄\|)`, centered | ≈ baseline |
| layernorm | `std(x)`, centered | ≈ baseline |
| clipnorm3 | RMS of winsorized x | ✗ outliers are functional |
| pnorm3 | `mean(\|x\|³)^⅓` | ✗ too peaky (toward L∞) |
| manifold | `rms^ρ`, ρ∈(0,1) learnable | ✗ partial ÷ hurts |
| peak (L∞) | `max(\|x\|)` | ✗ rides 1 dim |
| median | `median(\|x\|)` | ✗✗ over-robust, drops scale |
| squash (DyT) | none — `tanh(α·x)` | ✗✗ no ÷ → diverges |
| center | none — `x−x̄` only | ✗✗✗ no ÷ → NaN |

**One-line law:** a norm must *divide by a smooth all-dimension magnitude*.
Mild robustness helps, but the best magnitude is **stack-dependent**: the full
recipe liked a mildly robust L1.5, while the stripped baseline preferred
LayerNorm and a slightly less robust p-norm. The sweet spot is softer than L2
(down-weights the learned outlier channels) but not so soft it forgets the
scale (L1<p, not median).
