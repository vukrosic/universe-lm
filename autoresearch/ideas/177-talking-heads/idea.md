---
id: 177-talking-heads
status: running
round: 1
updated: 2026-06-15T05:07:19Z
transfer-risk: med
plain: Let attention heads talk to each other: each head's pre-softmax scores are mixed through a learned H×H matrix (and similarly post-softmax on the value output), starting at identity so step-0 is byte-identical.
---

# 177 — Talking-Heads Attention (Learnable Cross-Head Linear Mix on Pre-Softmax Scores and Post-Softmax Outputs)

## Source
- Shazeer, Lan, Cheng, Mao, Le, "Talking-Heads Attention" (arXiv:2003.02436, March 2020). The paper inserts two learnable H×H linear projections: one on the **pre-softmax logits** (cross-head logit mixing) and one on the **post-softmax outputs** (cross-head value mixing). Both projections are initialized to identity (no-op at step 0). Validated on Transformer-Big WMT'14 En-De/En-Fr (~220M params): +1.0 BLEU on En-De vs baseline Transformer-Big, +0.5 on En-Fr. Adopted in some recent architectures (e.g. GLaM, Primer) as a cheap parameter-shared mixing layer.
- In-repo context: the mechanism is **fully implemented** in `models/layers.py` (pre-softmax M init at line 1956, post-softmax M_out init at line 2089, application at lines 3187-3191 and 3217-3219), with config flags `use_talking_heads_q` / `use_talking_heads_out` already plumbed through `MultiHeadAttention.__init__` (lines 1065 / 1092) and `TransformerBlock.__init__` (line 3616). Identity init ⇒ bit-identical at step 0. **The mechanism is built but has never been A/B'd at tiny1m3m** — no `177-talking-heads` idea file, no entry in `closed.md`.
- Closest in-repo analogs (all closed null at tiny1m3m):
  - **152-attn-logit-bias** (per-head QK^T additive bias, Δ=+0.0131)
  - **155-per-head-temp** (per-head QK^T multiplicative temperature, Δ=-0.0063)
  - **160-rms-gain-per-head** (per-head post-AV multiplicative gain, Δ=-0.0023)
  - **166-t5-rpe** (per-head additive bucketed logit bias, Δ=+0.0106)
  - All four are **per-head scalar/bias levers** (H params, per-head). 177 is a **per-head × per-head cross-mix** lever (H² params, cross-head). The basis is qualitatively different — see "Why this breaks the prior" below.

## Mechanism
Standard attention (per batch, position):
1. `scores[b, h, t, s] = Q[b, h, t, :] · K[b, h, s, :] / √d_k` — shape `[B, H, T, T]`.
2. `attn_w[b, h, t, s] = softmax(scores[b, h, t, :], dim=-1)` — same shape.
3. `out[b, h, t, d] = attn_w[b, h, t, :] @ V[b, h, :, d]` — shape `[B, H, T, d_k]`.
4. concat heads → `[B, T, d_model]` → `W_O`.

With talking heads (pre-softmax — Q side):
- After step 1, mix across heads:
  `scores[b, h_new, t, s] = Σ_h M[h_new, h] · scores[b, h, t, s]` — `[H, H]` mix.

With talking heads out (post-softmax — O side):
- After step 3, mix across heads:
  `out[b, h_new, t, d] = Σ_h M_out[h_new, h] · out[b, h, t, d]`.

**Step-0 bit-identical**: both M and M_out are initialized to identity. `M @ x = x` for any `x` ⇒ scores, softmax, attn_w, out unchanged ⇒ **byte-identical to baseline at step 0 (max-abs-diff = 0.0)**.

The two levers (talking-heads-Q pre-softmax, talking-heads-out post-softmax) are independent. **Treatment**: both True (paper's full form). **Ablations**: Q-only and Out-only if first run shows signal.

## Design sketch
- **Files**: mechanism **already implemented** in `models/layers.py` (pre-softmax `M` at line 1956, post-softmax `M_out` at line 2089, application at lines 3187-3191 and 3217-3219, `_apply_logit_op` / `_apply_output_op` already aware of both flags). Config flags `use_talking_heads_q` / `use_talking_heads_out` already plumbed from `LLMConfig` to `MultiHeadAttention` to `TransformerBlock` (lines 1065, 1092, 3616). The remaining work is purely **a config subclass + runner wiring**:
  - `configs/llm_config.py` — add `Tiny1M3MTalkingHeadsConfig(LLMConfig)` with `use_talking_heads_q=True, use_talking_heads_out=True` (or a `talking_heads: bool = True` umbrella flag).
  - `autoresearch/bin/run-*.sh` or runner script — wire the new config into the A/B harness.
- **Config flags**:
  - `use_talking_heads_q: bool = False` (default off, pre-softmax mix).
  - `use_talking_heads_out: bool = False` (default off, post-softmax mix).
  - **Treatment**: both True (paper's full form).
- **LoC**: ~5-15 lines of new code (a config subclass, a runner entry). Among the smallest costs in the queue.
- **Coordination check**: `git diff` confirms the parallel AI is also editing `models/layers.py` and `configs/llm_config.py` but the diff hunks touching talking-heads code (lines 3187-3219) are non-overlapping with the `talking_heads_M` / `talking_heads_out_M` parameter definitions; the A/B wiring is a clean new entry. No conflict.

## Scale evidence
- Talking-Heads Attention validated on Transformer-Big WMT'14 (~220M params, encoder-decoder translation), +1.0 BLEU En-De. **Direct validation at ≥100M.**
- In-repo at 0.94M: four per-head-attention-shape levers have closed null (152, 155, 160, 166), all within the |Δ| < 0.014 cache band. 177 is the **cross-head** analog — a structurally different axis from any of the four (see "Why this breaks the prior" below).
- **Transfer risk: med** (validated at ≥100M in translation; not directly validated at GPT-style causal LM at ≥100M. The mechanism is scale-free so the bet is plausible; cross-head mixing is also adopted in some recent causal-LM architectures.)

## Why this breaks the per-head-scalar prior
The four closed nulls (152, 155, 160, 166) all share a **specific failure mode** that 177 explicitly avoids:

- **Failure mode of 152/155/160/166**: each is a **per-head scalar/bias** — the lever only lets each head shift or scale *its own* logit / post-AV output by an H-independent constant. The function class on the `[B, H, T, T]` attention tensor is `scores[b, h, t, s] ← α_h · scores[b, h, t, s] + β_h` (or post-AV gain α_h). Parametrically, this spans an H-dimensional affine subspace of the full `[H, T, T]` score tensor. **The Q/K/W_O gradient updates can absorb this exactly** — Q[b, h, :] and K[b, h, :] can each be rescaled by α_h^(1/2) post-hoc without changing the score magnitudes; the per-head bias β_h can be re-implemented as a per-head shift in Q's channel direction. At 0.94M/12L/4H with 92 update steps, the model has no incentive to develop specialization that the Q/K gradient couldn't already provide, so the lever collapses to identity in the loss surface.
- **What 177 does differently**: the H×H matrix `M` is a **strictly richer basis** on the `[H, T, T]` score tensor. It spans an H²-dimensional subspace of the full H×T×T function space. Critically, the H×H matrix **mixes across heads** — `scores[b, h_new, t, s] = Σ_h M[h_new, h] · scores[b, h, t, s]`. This is not a reparameterization of Q/K: there is no Q[b, h, :] or K[b, h, :] rescaling that produces cross-head coupling of score values. The only way to reproduce a non-identity M is to learn a *correlated* change across two heads' Q/K projections simultaneously, which costs more gradient signal at 0.94M than just letting M absorb the coupling directly. At 0.94M/12L/4H the H×M = 4×4 = 16 params per layer is tiny in absolute terms, but the *basis richness* per param is strictly higher than per-head scalars.
- **Why the cross-head axis plausibly binds where per-head scalars did not**: at 0.94M the binding bottleneck is **cross-head communication**, not per-head tuning. With H=4 heads and d_model=64, each head has 16 dims of Q/K/V and the residual stream mixes all four head outputs into a 64-dim vector for the next block. A per-head scalar lever has no cross-head axis to exploit — it can only be re-absorbed. A cross-head mix has an axis that Q/K/W_O gradients cannot easily reparameterize (the cross-head coupling), so the optimizer has a *strictly new* loss direction to move along.
- **Plausibility prior**: talking-heads is well-validated at ≥100M in translation (arXiv:2003.02436, WMT'14, +1.0 BLEU). The mechanism is scale-free (no size-dependent parameterization), and the per-head × per-head cross-mix is the first lever in our queue to operate on a tensor axis that per-head scalars don't span.

## Why it's worth a slot
The bet, in one sentence: **cross-head mix is a strictly richer function class on the attention tensor than per-head scalars, and is the first lever in our queue that operates on an axis Q/K/W_O gradients cannot easily reparameterize, so it has a positive probability of binding at 0.94M where the per-head-scalar family (152, 155, 160, 166) failed.**

- **Expected Δval** (tightened per r1 finding 3): we expect **Δval ∈ [-0.015, -0.035]** if the lever binds. Lower bound: -0.015 is the magnitude of the largest per-head-scalar null (155 Δ=-0.0063, 160 Δ=-0.0023, 166 Δ=+0.0106 wrong-sign). Upper bound: -0.035 is in the WIN zone per 016-qk_norm (Δ=-0.0138/-0.0185) and 154-rebased-attn (Δ=-3.48 record break); we don't expect talking-heads to break the record, but the basis-richness argument supports a clean PASS. **Pass bar**: Δ ≤ -0.025 (clean PASS, outside the cache band and above the largest per-head-scalar null magnitude). **Null band**: |Δ| < 0.015. **Drift**: |Δ| > 0.04 wrong-sign ⇒ reject the cross-head-mix family at 0.94M (re-evaluate at Phase-2 ≥135M).
- **Info value of a null**: a clean null at 0.94M would close the *cross-head-mix* sub-axis and complete the per-head-attention-shape family closure (152, 155, 160, 166, 177 all null). This is informative because it tells us the binding bottleneck at 0.94M is not "head specialization" in any form (per-head scalar, per-head × per-head cross-mix) — it's something more fundamental, likely "each head has too few gradient updates to specialize at all." That hypothesis re-directs Phase-2 attention levers to richer architectures (multi-query, GQA, MLA) rather than more head-shape tuning.
- **Info value of a win**: a clean PASS at 0.94M would unlock the cross-head-mix family for Phase-2 (≥135M), where the H grows (e.g. H=12+ for 135M, ~3× the per-layer parameter count for talking-heads mix) and each head has more gradient signal. Talking-heads is well-validated at 220M+; a 0.94M win would be a strong prior for Phase-2 screening.
- **Cost**: ~5-15 LoC of new code (config subclass + runner wiring). Smallest cost in the queue.
- **Queue decision (per r1 finding 2)**: 176 (V-pre-AV-norm) is already at `needs-review` and 177 is the one being repitched. The queue has already made the slot-pick — this repitch accepts it and sharpens 177's bet rather than relitigating the order.

## What a runner should do
1. Implement `Tiny1M3MTalkingHeadsConfig(LLMConfig)` with `use_talking_heads_q=True, use_talking_heads_out=True` in `configs/llm_config.py`.
2. Add a runner entry that points the A/B harness at the new config (vs the cached baseline 6.4394±0.04 / 6.4447±0.0488).
3. Verify step-0 byte-identical (max-abs-diff = 0.0 vs baseline, no tolerance needed — the operation is a literal identity matrix multiplication).
4. Run at tiny1m3m, seed 42 (the protocol — see PIPELINE.md).
5. Cache verdict: WIN if Δ ≤ -0.025, NULL if |Δ| < 0.015, DRIFT otherwise.
