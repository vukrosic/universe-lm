## r2 — 2026-06-15 — verdict: accept

**The r2 pitch earned the slot.** All 4 r1 findings closed concretely, and the new framing exposes a sharper bet than I gave credit for in r1.

**Sharp mechanistic sentence (delivered).** "The lever is on the discrete operator change, not on a smooth correction." At α_h>1 the bottom ~70% of K positions receive p=0 exactly, ∂L/∂V_i=0 exactly, and a single bit of α_h movement crosses a discontinuity in ∂L/∂V. This is non-perturbative in a way the 8 prior smooth siblings (152/155/160/162/165/166) genuinely are not. The bit-identical-at-step-0 framing is real (`α_h = 1 + 0.5·(1+tanh(0)) = 1`, the bisection degenerates to the standard softmax projection in the α=1 limit, bisection_tol=1e-7 keeps max-abs-diff < 1e-7 well below the 1e-5 fp32 noise floor). The lever is **isolated to α_h** by construction — no other parameter can absorb the change. Sharp.

**Tight bar (delivered).** Δ ≤ -0.015 WIN OR Δ ≥ +0.05 DRIFT as the bar for "lever binds at this tier"; anything inside the null band is a clean close, not a WIN. With 8 prior family nulls, the prior is genuinely 70% in-band null — the r2 honest Δ prior is right and the bar is tight. This is a *commitment* on the win condition, not a vibe.

**3-family differentiation (delivered).** Concretely separated:
- Family 1 (operator perturbation): 152/155/160/162/165/166 — smooth, small-Lipschitz, absorbed by Q/K gradient updates.
- Family 2 (operator replacement, non-attention): 148 — focal modulation replaces the attention block. Different architecture.
- Family 3 (capacity injection): 156/117/118/146 — MoE/router/expert levers that *add* parameters.

Entmax-1.5 is **none of these three**: bit-identical at step 0 AND non-perturbative as α_h moves. The lever is isolated to one axis. A null from a non-smooth, isolated test is a *stronger* null than 8 prior smooth siblings — if a non-perturbative operator change can't bind at this tier, the soft-perturbation nulls are independently confirmed. WIN unlocks Phase-2 family. DRIFT saves a Phase-2 slot. All three outcomes informative.

**Field veto reframed (delivered).** 6+ years of softmax dominance in production LMs is a real soft negative. The r2 pitch correctly identifies that the close-line will read exactly "softmax-replacement axis closed at 0.94M" — and that is a known outcome worth logging with a stronger test, not a reason to skip the test. Agreed.

**Milder variant deferral (acceptable).** The r1 reviewer floated entmax-1.2 as a "small dose" play. The r2 defers to a follow-up if r2 entmax-1.5 nulls. Reasonable: splitting the slot into 1.5+1.2 doubles run budget and dilutes signal. Clean r2 first, follow-up is cheap.

**Why this clears the taste bar.**

- *Leverage*: Δ bar of -0.015 matches the 154-rebased-attn WIN family (Δ=-3.48 against a buggy ctrl but the plan bar was tighter), and the r2 honest prior says 20% mild WIN / 10% strong WIN-or-DRIFT / 70% null. Modest expected magnitude but a real lever.
- *Information value*: high — three informative outcomes, all of which log something useful about the softmax-replacement axis.
- *Non-obviousness*: the per-head-learnable α_h as a *single-axis, non-perturbative, bit-identical-at-step-0 lever* is a fresh framing for a well-known mechanism. Distinct from the closed "NSA/diff-attn/hybrid heads" axis (diff-attn is smooth post-QK, Family 1).
- *Portfolio fit*: yes, 9 prior family nulls are crowded. But r2 is structurally different from each of the 9 (only non-perturbative operator replacement in the family), not the 10th smooth perturbation.
- *Niche fit*: mechanism (not HP), identity/zero-init-able, tiny1m3m-showable. ✓
- *Crisp bet*: "Δ ≤ -0.015 OR Δ ≥ +0.05 as the bar for lever-binding; otherwise clean null." Single sentence, falsifiable.
- *Transfer*: med-risk tag, with the null outcome pointed at Phase-2 re-evaluation. Mechanism is scale-free. Acknowledged.

**Engineering**: sound. LoC ~80, bisection budget 32 is realistic, the tanh parameterization is correct, the YOCO/standard-MHA pass-through is straightforward, the closed-axis / step-0-byte-identical claims are documented. The `entmax_15` helper follows Peters et al.'s well-known recipe and is implementable from the `entmax` PyPI reference. No concerns.

**Resetting round to 1 for the definition gate's budget.**

## r1 — 2026-06-15 — verdict: revise

**Leverage is soft.** Paper gain is modest (+0.5 BLEU on WMT'14, +0.5–1.0 GLUE on BERT-base at ≥100M) and the miner's own predicted band Δval ∈ [-0.005, -0.020] has the lower end sitting inside the |Δ|<0.01 null band — 60% of the expected range is null. Not a high-leverage lever at our tier.

**The bet is a vibe.** "Sparse attention is a strong inductive bias that may shorten the optimization horizon" is hand-wavy. A revise needs ONE sharp mechanistic sentence — e.g., "at H=4 d_k=16 T=2048, the dense AV matmul accumulates gradient noise on the bottom ~70% of K positions; forcing α_h>1.5 collapses that mass into the top ~30%, so per-step gradient SNR on the surviving K rows rises by ~3×." Predict something concrete.

**Softmax-replacement axis is weakly-answered here.** Three siblings have already nulled in this family:
- 148-focal-mod null — "non-softmax-attention axis at 0.94M" (closed)
- 156-moa null — parallel-attention-experts + router (closed)
- 166-t5-rpe / 152 / 155 / 160 / 162 / 165 — every per-head attention-shape lever null at 0.94M/12L/4H
- closed.md axes line: "NSA / diff-attn / hybrid heads" closed

Diff-attn is the closest cousin (smooth differentiable post-QK operator) and is explicitly on the closed axis. A null from entmax-1.5 would re-confirm what 148 + 156 + 152 + 155 + 160 + 162 + 165 already weakly-answered — the softmax-replacement / per-head-shape family does not bind at 0.94M. The miner acknowledges 148 but argues entmax-1.5 is a different operator (true sparsity, not gated-additive context, and not focal modulation). That distinction is real but the queue has 9 prior soft signals in the family.

**Field-veto signal.** 7 years after Peters et al. (2019), LLaMA 1/2/3, Mistral, Qwen 1/2/2.5, Gemma 1/2, OLMo, Falcon all use softmax. The mechanism has had 6+ years to be adopted; it wasn't. That is not proof it's bad, but it IS soft evidence the lever is in the right tier, not ours.

**Transfer case is plausible but unvalidated for causal LM.** Miner's "transfer-risk: med" is honest. Closest 135M re-evaluation would be in Phase-2; the lever needs to clear tiny1m3m to get there.

**Engineering is good.** Bit-identical step-0 init via `α_h = 1 + 0.5·(1+tanh(α_raw_h))` is correct (the three-way init comparison is well done), bisection budget is realistic, LoC ~80 is fine, helper is well-known. None of this is the problem.

### What the r2 pitch must add
1. **One sharp mechanistic sentence** that names the binding constraint at 0.94M/12L/4H/3M tokens and predicts which axis entmax-1.5 will move. Vibe: not enough.
2. **A stronger expected Δ** — at least Δ ≤ -0.015 committed (the bar —current best 154-rebased-attn WIN was Δ=-3.48 against a buggy ctrl, plan bar is -0.005/-0.01). If the miner's honest read is "this is a null-confirmation play", say so explicitly with a stated info-value.
3. **Explicit differentiation from 148/156/152/155/160/162/165/166**: argue WHY entmax-1.5 (true sparsity, operator replacement) is a different family than the eight prior soft siblings. The "we are softmax at step 0" framing is the strongest differentiation; pin it down.
4. **Optional**: consider a milder-sparsity variant (entmax-1.2 — close to softmax, only mildly sparse) as a "small dose" of the lever; if the r2 pitch goes that way, the bet becomes "a small dose of sparsity helps" and the Δ floor can be -0.005 with confidence.
