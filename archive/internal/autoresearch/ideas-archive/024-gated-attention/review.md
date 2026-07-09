# Review log — 024 gated-attention

## r2 — 2026-06-10 — verdict: approve

All four blocking findings from r1 are now cleanly pinned. Spec is sound and ready to plan.

- **Qwen disambiguation (r1 #1) — closed.** Line 11 now states the Qwen *vs* Qiu contrast explicitly: Qwen gates **Q** pre-softmax; Qiu gates `o_h` post-AV — *different site, different axis*, this idea is on the o_h site. Cites the actual paper, no fabricated contrast. Good.
- **Identity-init form (r1 #2) — closed.** Line 18 pins form (b) `2·σ(W·x + b)` with `W=0, b=0` verbatim, with the reason (b) over (a)/(c) — constant exactly 1, `o_proj` init untouched, step-0 ≡ baseline to floating-point. Plan will inherit without improvisation.
- **Per-head scalar shape (r1 #3) — closed.** Line 20 pins `nn.Linear(d_model, H)` with the per-layer/whole-model param math (12,288 params = 1.3% of model). Vector-form cost (393,216 = 42%) is named and rejected as a parameter lever in disguise. Correct tier-appropriate choice.
- **Numeric pass bar (r1 #4) — closed.** Line 26 pins `Δ := trt_val − ctrl_val; pass iff Δ ≤ −0.01`, tied to the box-noise floor, with the no-add-seeds guardrail inline. Sub-noise deltas are explicitly *inconclusive → null*, not "run more seeds." Inheritable by the plan gate.
- **Gate input site (r1 #5, follow-on) — closed.** Line 16 explicitly states the gate is computed from the *sublayer input residual `x`* (pre-LN/attn), not from `o_h` — avoids circularity, matches Qiu and modded-nanogpt. The `MHA.forward` site is unambiguous.
- **Site distinctness (re-verified).** Re-checked `closed.md` and the active attention-side set (020 FoX → A-prob decay, 021 V-residual → cross-layer V, 022 softpick → softmax swap, 023 canon-conv → pre-attn conv, 025 SSMax → logit temperature). No closed or active lever on the post-AV head-output site. 024 stands alone on this axis.
- **LoC budget (re-confirmed).** ~30 LoC: one flag-gated `nn.Linear(d_model, H)`, one sigmoid, one elementwise multiply, and `use_gated_attn: bool = False` plumbed through MHA → TransformerBlock → LLMConfig (mirroring the `use_fire_pe` pattern at `models/layers.py:428`, `models/llm.py:215`, `configs/llm_config.py:150`). Under 200-LoC budget; safe.
- **Tier (confirmed tiny1m3m-only).** No screen20m, no multi-tier, no full ladder. Scalar gate math is parameter-budgeted at this tier. The d_k=128 Qiu result is explicitly flagged as not transferring directly — the cheap scalar is the right ablation.
- **Trt config class.** Spec should pair with the FIRE-equipped baseline (per taste convention) so the A/B partitions the *orthogonal* head-output axis from 009's additive position bias — implementer's call on the exact baseline mirror, but the orthogonal-A/B principle is the right one.

No new findings. Approve → `needs-plan`. Round reset to 1 for the code gate.

## r1 — 2026-06-10 — verdict: revise

- **Source (verified).** arXiv:2505.06708 (Qiu et al., May 2025) is a real, current paper on the post-AV head-output gate — citing it is fine. The modded-nanogpt speedrun "head gate" variant is the same site, also real.
  - **But the "Qwen-style gated attention" line in idea.md:11 is wrong and must be removed/clarified.** Qwen's "gated attention" gates **Q** *before* softmax (`softmax(Q·K^T) ⊙ σ(Q_g·x)`), not `o_h` *after* AV. These are **different sites** — Qiu's mechanism is post-AV, Qwen's is pre-softmax. The active set has nothing on the Q-gate site; this idea is on the o_h site. Reviser: delete the Qwen mention or replace it with a one-line clarification that Qwen's gate is on a *different* axis (and is therefore *not* a duplicate of this lever).

- **Pin the identity-init form. The idea lists three options (a, b, c) but does not pick one — that is a definition-gate call, not a taste call.** The three are:
  - (a) `sigmoid(W·x + b)` with W=0, b=0 → constant 0.5 at step 0; requires 2× `o_proj` init when `use_gated_attn=True`.
  - (b) `2 · sigmoid(W·x + b)` with W=0, b=0 → constant 1.0 at step 0; o_proj init unchanged. **Cleanest step-0 ≡ baseline.**
  - (c) `1 + sigmoid(W·x + b)` with W=0, b=0 → constant 1.5 at step 0; requires `(2/3)×` o_proj init.
  - Recommend **(b)**: a single 2× multiplier and W=0, b=0 makes step-0 numerically equivalent to baseline to floating-point precision, with no o_proj init gymnastics. The plan must pin this form, *and* the W=0, b=0 init, verbatim — implementer must not improvise.

- **Pin the gate shape: per-head *scalar*, not per-head vector.** The idea mentions both `nn.Linear(d_model, H·d_k)` (vector, d_k gates per head) and `nn.Linear(d_model, H)` (scalar, 1 gate per head). At tiny1m3m (d_k=32, H=8, 6L):
  - vector: 256·32·8 = 65,536 params/layer × 6L = **393,216 params → 42% of the 0.94M model**. A *parameter* lever disguised as a structural one.
  - scalar: 256·8 = 2,048 params/layer × 6L = 12,288 params → **1.3% of the model**. Negligible.
  - The Qiu paper primarily tests the vector form at d_k=128; at d_k=32 the parameter cost is prohibitive and the lever stops being "cheap." Reviser: pin **scalar** in the spec, with the one-line justification that vector form blows the parameter budget at this tier. (Paper's d_k=128 result does not transfer directly — the cheap variant is the right ablation here.)

- **Falsifiable pass bar is missing.** The idea says "expect a val-loss drop" with no number. Box noise at tiny1m3m is ~±0.01 val loss; smaller deltas are unresolvable at seed 42. Reviser: add a concrete bar in the form `Δ := trt_val − ctrl_val; pass iff Δ ≤ −0.01` (or whatever number you can defend from the Qiu paper's per-tier scaling, but a number is required). The plan gate will inherit it — but the spec must own it. **Do not** punt to "add seeds" if Δ is sub-noise — log it null and close (per the seed-42 rule).

- **Where the gate value comes from (clarify).** Idea says "from the *current token's* residual `x_t`" — i.e. input to the *current* sublayer, not the attention output. That's the right site (Qiu uses pre-attention residual; modded-nanogpt uses the same). Reviser: state explicitly in the spec that the gate input is the sublayer input residual `x` (pre-LN/attn), not `o_h` itself, to avoid circularity. The plan's `MHA.forward` will read `x` from the parent block.

- **Site distinctness (confirmed against closed axes).** Cross-checked `autoresearch/closed.md`. No "head-output value gate" lever filed. The two "gated" closes (008, 012 — gated-deltanet) are Yang et al. *linear-RNN* recurrence gating, an entirely different site and family. The active attention-side set (020 FoX → A-prob decay, 021 V-residual → cross-layer V, 022 softpick → softmax swap, 023 canon-conv → pre-attn conv, 025 SSMax → logit temperature) does not include a post-AV head gate. **024 is the only lever in flight on this site.** Distinct.

- **LoC budget (verified).** Taste estimate of ~30 LoC checks out: one `nn.Linear(d_model, H)` flag-gated, one `sigmoid`, one elementwise multiply, and `use_gated_attn: bool = False` plumbed through `MHA`, `TransformerBlock`, and `LLMConfig` (mirroring the `use_fire_pe` flag pattern at `models/layers.py:428`, `models/llm.py:215`, `configs/llm_config.py:150`). Implementable in the current harness. Trt config class should add the flag to the FIRE-equipped baseline (so the A/B partitions the *orthogonal* head-output axis from 009's additive position bias — same convention 020/022 use).

- **Niche fit (confirmed).** Mechanism, not HP. Identity-init-able (modulo the pin above). Runs at tiny1m3m, no data/infra dependency, no tier-mismatch (Qiu validates ≥1.5B, but the *cheap* scalar variant is in scope and informative even if it nulls). No schedule, no EMA trap (unlike 018 AdEMAMix).

**Summary for reviser:** salvageable, four blocking items, none of them fatal. (1) Remove the Qwen-style mis-attribution. (2) Pin identity-init form (b) `2·σ(W·x + b)`, W=0, b=0. (3) Pin per-head *scalar*. (4) Add a numeric pass bar (recommend Δ ≤ −0.01). Apply, re-pitch, back to this gate for re-review.
