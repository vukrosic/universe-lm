---
id: 069-focal-loss
status: needs-plan
round: 1
updated: 2026-06-11T01:21:40Z
transfer-risk: med
---

# 069 — Frequency-conditioned focal loss

## Source
Mukhoti et al., "Calibrating Deep Neural Networks using Focal Loss" (arXiv:2002.09437, Feb 2020) — base form. The frequency-conditioned variant pitched here is LM-specific and not in Mukhoti; closest prior art is class-balanced re-weighting (Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", arXiv:1901.05555, Jan 2019), which our γ-schedule generalizes.

## Mechanism
Per-token CE is wrapped as `(1 - p_t)^γ(token) · CE_t`, where **γ depends on the target token's corpus frequency**: `γ(token) = γ_max * (1 - rank_norm(token))`, with `rank_norm` the normalized log-frequency rank over the tokenizer (top-1 token like ` ` gets γ≈γ_max, rare-tail tokens get γ≈0). γ_max=2 from Mukhoti default; the per-token γ table is precomputed once from the training corpus and stored as a `(vocab_size,)` buffer. This is *not* a global wrapper like polyloss/label-smoothing/confidence-penalty/unlikelihood — those reshape the loss uniformly across the vocab; this reshapes it **per-class** with a non-uniform γ schedule. Implementable in ~80 LoC: one buffer + one elementwise multiply in the loss path, model graph unchanged.

## Scale evidence
Mukhoti 2020 shows base focal loss is competitive accuracy + better calibration on vision and NLP at moderate scale. Cui 2019 shows class-balanced re-weighting helps long-tailed CIFAR/ImageNet-LT at ResNet-50 scale. `transfer-risk: med` because both mechanisms are real and cross-domain, but the **composition** (per-token γ schedule, not 1/freq weight) has no direct sub-200M LM-pretraining citation — the bet is mechanistic, not evidentiary. A win is a genuine find; a null is genuinely informative about whether long-tail re-weighting transfers to tiny LMs at all.

## Why it's worth a slot
**Bet:** at tiny1m3m the top-200 vocab tokens (whitespace, punctuation, common subwords) saturate `p_t > 0.5` within the first ~10 steps even at this scale — these *are* the easy-token population whose existence the taste round-1 review questioned. Frequency-conditioned focal redirects gradient from those onto the long tail, which is exactly the population val CE measures most volatility on. **Why this differs from polyloss-NULL:** polyloss is a global Taylor-expansion wrapper — uniform γ across vocab — and went null because mean (1-p_t)^γ ≈ 1 at underfit. This lever sidesteps the underfit-mean argument by *only* activating where p_t is high (top-of-vocab); the rest of the loss is untouched CE. **Information value either way:** a win means LM long-tail re-weighting works at this scale (directly portable to the 10M→135M ladder where vocab-imbalance gets worse not better). A null means the model isn't actually saturating on common tokens even at this scale, which falsifies the "easy-token gradient waste" thesis for tiny1m3m and tells the next loss-family screen to stop looking here. Both outcomes close a question the 4 already-queued loss bets cannot — they are uniform wrappers and cannot distinguish per-class saturation from global underfit.

