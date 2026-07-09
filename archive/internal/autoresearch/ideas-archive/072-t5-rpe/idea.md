---
id: 072-t5-rpe
status: needs-plan
round: 2
updated: 2026-06-11T01:22:23Z
transfer-risk: low
---

# 072 — T5 Relative Position Bias

## Source
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al., JMLR 2020 / arXiv:1910.10683). T5 has been the canonical learned-bucketed distance bias since 2019/2020; no successor has dethroned it as a mainstream pretrained form, and it is the reference point every modern PE (ALiBi, RoPE, FIRE, CoPE) is benchmarked against — source-date is old, the mechanism is still the reference.

## Mechanism
Replace the current positional prior with a learned bucketed relative-distance bias table added directly to attention logits. **T5 default — shared across heads, `n_buckets=32`**, log-spaced past a small linear region (the published bucketing function). Zero-init the table so step 0 is bit-identical to the baseline.

## Scale evidence
T5 ships the bias as part of a family that scales to large pretrained models, including the 11B checkpoint in the paper. transfer-risk: low — the mechanism is mainstream at scale and the step-0 identity is exact.

## Why it's worth a slot
This isolates a learned bucketed distance prior from FIRE's fixed smooth γ-decay kernel; if the bucketed form wins on top of FIRE, the useful part is quantized relative distance, not a smooth kernel.

## Tier
tiny1m3m (0.94M · 3M tokens · ctx 1024), seed 42. The lever fires at any context length, so tiny1m3m is sufficient.

## Bar (primary A/B)
- **Control:** the current PE WIN — V+q+SWA+HighRoPE+**FIRE** tiny1m3m baseline, **val 6.3234** (per `closed.md:50`, the 009-fire-pe winner).
- **Treatment:** same baseline **+ T5 bucketed bias** (zero-init, shared across heads, `n_buckets=32`) added on top of the FIRE kernel — bias is purely additive on attention logits, so step 0 is identical to control.
- **Expected Δ:** in `[−0.005, −0.020]` val loss (the bucket table can pick up periodic content structure the smooth γ-kernel cannot resolve; the upper end is the bound of "real but small" at this tier).
- **Pass / fail / noise** (one-seed, box noise ≈ ±0.01 val):
  - **pass (WIN):** Δ ≤ −0.005 (treatment val ≤ 6.3184) **and** outside ctrl-pair gap.
  - **fail (NULL / drift):** Δ ≥ +0.005 (treatment val ≥ 6.3284) → bucketed bias does not stack on FIRE.
  - **inconclusive:** `|Δ| < 0.005` → log inconclusive; do not add seeds.
- **Run shape:** standard two-control bracket (ctrl + ctrl + trt, seed 42) per the closed-007/closed-009 protocol so the ctrl-pair gap can be measured against ±0.01 box noise.

## Knobs (pinned, not free)
- `n_buckets=32` — T5 paper default; at T=1024 this gives 4–8 distances per bucket near the far end, coarse enough to test the "quantized relative distance" hypothesis.
- **Shared across heads** — T5 default; per-head is a tiny extension worth a follow-up idea, not shipped here (no extra knob to sweep).
- **Additive on RoPE+FIRE** — bias adds to logits *after* RoPE rotation, so it stacks rather than replaces; this is the clean test of "is the bucket form doing something the smooth γ doesn't?"
- A **NoPE + T5-bias** counterfactual (T5-bias substitutes for RoPE rather than augments) is **deferred** to a follow-up if the primary A/B wins — not staged in this idea.

## Failure modes
- **Bucket table never moves off zero** → null result by construction; report `‖B‖` at end of training in evidence.
- **Stacks destructively with FIRE's smooth kernel** (cf. 013-cope drifted +0.069/+0.077 when stacked on FIRE per `closed.md:37`) → fail bar fires, close as additive-with-FIRE NULL; the NoPE counterfactual is then the natural follow-up.
- **Drift inside ±0.005** → log inconclusive per the one-seed rule; do not "re-run with more seeds."

## Reviser note (r2)
- The bar is now numeric and tied to the FIRE control (6.3234), with pass/fail/noise bands.
- `n_buckets=32` is pinned and shared-across-heads is explicit.
- The additive-on-RoPE+FIRE choice is now the committed primary A/B.
