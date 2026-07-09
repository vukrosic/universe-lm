---
id: 019-dyt
status: rejected
round: 1
updated: 2026-06-09T12:38:35Z
---

# 019 — Dynamic Tanh (DyT) replacing LayerNorm

## Source
"Transformers without Normalization" (Zhu, Chen, Liu, Darrell, He —
Meta + NYU + MIT + Princeton; March 2025 — arXiv:2503.10622). Reports
that replacing every LN / RMSNorm in a transformer with a parameter-only
`tanh(α·x)` (no statistics, no per-sample reduction) matches or beats
the LN baseline across ViT, DiT, LLaMA-style LLM, and wav2vec setups.

## Mechanism
A `LayerNorm(d)` computes per-token statistics
`y = γ · (x - μ(x)) / σ(x) + β`, requiring a mean and variance reduction
across `d_model` per token (and the same in the backward). `DyT(d)` drops
the reduction entirely and replaces the norm-and-affine with a scaled
non-linear squash:

```
DyT(x) = γ ⊙ tanh(α · x) + β       # α is a single learnable scalar
                                    # γ, β are length-d_model learnable vectors
```

`α` is a *scalar* (one parameter per DyT module), shared across all
features and tokens; `γ, β` are the usual per-feature affine
parameters. Replaces every `nn.LayerNorm` (or `RMSNorm`) call site:
the two pre-norms inside each block, the final pre-out-proj norm, and
(if present) the embedding norm. < 50 LoC: one new `nn.Module`, a
sed-style replacement at ~6 call sites in `models/layers.py`.

Init follows the paper: `α₀ = 0.5` for LLM-style models (paper Table 2),
`γ₀ = 1`, `β₀ = 0`. At step 0, `tanh(0.5·x) ≈ 0.5x - x³/24 …` for
small `x` — *not* bit-identical to LN, but for the residual stream's
init RMS ≈ 1 the gain is close to 1; the output stays within ~10% of
LN's at the start, and the model trains from the same input/output
distribution. (Strict identity-init is impossible here because the
mechanism is *by construction* a non-norm; the bet is that the model
recovers within a few hundred steps.)

## Why it's worth a slot
The bet: the *closed* norm-zoo (pnorm, manhattan, center, squash, clip,
channelscale) all kept LN's reduce-then-normalize template and varied
which *statistic* was subtracted — they all showed null. DyT is a
*categorical* change: there is no reduction at all, no mean, no
variance — just a learnable squash. It tests whether LN's stabilization
value comes from *bounding activations* (which `tanh` does
parameter-only) or from *centering and scaling* (which the closed
norm-zoo tested). If DyT matches LN at tiny1m3m, we get a small free
speedup (no all-reduce across features → ~5-8% step-time at our small
`d_model`) *and* a positive signal to test the same at screen20m later;
if it loses, we confirm the closed norm-zoo result that the *reduce*
matters, not the *shape* of the post-affine. Strictly distinct from
017-sub-ln-sandwich (which adds a *second* LN per sublayer — same
function, different placement) and from 016-qk-norm (which adds LN on
Q/K). DyT *replaces* LN at all sites with a non-norm. < 50 LoC, one
boolean flag.
