---
id: 051-scalenorm
status: needs-run
round: 1
updated: 2026-06-11T01:38:25Z
transfer-risk: med
---

# 051 — ScaleNorm (RMSNorm with a single scalar gain)

## Source
Nguyen & Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention" (arXiv:1910.05895). Used recently in modded-nanogpt-adjacent recipes as the cheapest possible residual-stream norm.

## Mechanism
Replace each `nn.RMSNorm(d_model)` site (current baseline: `models/layers.py:246` `RMSNorm`, used at both `norm1` and `norm2` of every block via `make_norm(..., "rmsnorm")`) with a norm whose **gain is a single learned scalar `g ∈ ℝ`** instead of a per-channel vector `g ∈ ℝᵈ`. Operation: `y = g · x / √(mean(x²) + eps)`. New `norm_type` string `"rmsnorm_scalar"` registered in `make_norm`; identity-init `g = 1.0` so step-0 ≈ baseline up to a constant. <50 LoC.

This is **not** PNorm(p=2) (closed): PNorm keeps the per-channel gain vector and only changes the denominator from RMS to L₂; ScaleNorm keeps the RMS denominator and shrinks the gain. It is also **not** N1 `reparam_gain` / N6 `scaled_init` / N7 `softplus_gain`: those three all keep `g ∈ ℝᵈ` and only reparameterize the per-channel vector. ScaleNorm is the one axis the RMSNorm lever family does not touch — **gain shape (vector → scalar)** rather than gain init or parameterization.

## Scale evidence
Source's strongest evidence is at NMT (Transformer-base, ~60M params): +1.1 BLEU average over five low-resource pairs, competitive on WMT14 EN-DE. No published LM-pretraining ablation at the d=192 regime. transfer-risk: **med** — the *direction* of the bet (drop residual-stream gain expressivity) is independent of task, but the magnitude and even sign at tiny1m3m's d=192 is genuinely unknown.

## Why it's worth a slot
**The bet (adversarial to 016-qk-norm).** 016-qk-norm WIN (`closed.md:36`) said *adding* targeted Q/K norm structure helps at tiny1m3m — the model wants more norm machinery on attention. ScaleNorm bets the opposite shape: the **residual-stream** per-channel gain is doing little or no work; replacing `g ∈ ℝᵈ` with `g ∈ ℝ` removes the model's ability to silently scale individual channels up/down via norm, forcing channel selection to live in the W matrices where it's visible. The two bets are compatible (different sites) but the *direction* is opposite: ScaleNorm is "less norm expressivity is better, on residual."

**The leverage is the null direction, not param count.** Param savings are ~0.24% of 0.94M (rounding). The actual edge is *information*: a clean ScaleNorm loss is the cleanest evidence we'll get that per-channel residual-stream gain is load-bearing — which directly says "don't try cheaper-gain residual norms; do double down on per-channel structure." A win is the same diagnostic in reverse: residual gain is vestigial, so it becomes worth running follow-ups that freeze the gain to 1 entirely (no params at all on residual norms) and possibly migrate the norm budget to attention sites where 016-qk-norm already showed it pays.

**The prior.** I expect ScaleNorm to **lose narrowly** at tiny1m3m (per-channel gain probably is mildly load-bearing in the d=192/12-layer regime, where embedding tables already dominate the param count), but to lose by less than the noise gap — making the loss informative in the "per-channel gain is doing ~something" direction without being a blowout. The taste call is whether that null is interesting enough to spend a slot on; the bet is that it is, because every later residual-norm idea references which side of this line we ended up on.
