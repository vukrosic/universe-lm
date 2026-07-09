---
id: 046-multi-token
status: needs-plan
round: 1
updated: 2026-06-11T01:34:07Z
transfer-risk: med
---

# 046 — Multi-Token Attention (Q/K time-axis, stacked on Canon)

## Source
Golovneva et al., "Multi-Token Attention" (arXiv:2504.00927), 2025.
The paper conditions attention scores on small neighborhoods of
query/key tokens by depthwise convolving over projected Q, K, and
(in the full paper) the heads axis. For tiny1m3m we adopt the
**Q/K time-axis** half of the mechanism and explicitly drop the
heads-axis conv (see `## Mechanism` and `## Scale evidence`).

## Mechanism
Insert a single **causal depthwise Conv1d, kernel size 3, along the
time axis only**, on the projected Q stream and on the projected K
stream, **once per block, immediately after the QKV projection and
before the score reshape/dot-product** (i.e. before any `(B, H, T, D)`
view). Each stream has its own per-block scalar output gate `g_q`,
`g_k` init to 0 so step-0 ≡ vanilla attention. Depthwise, k=3,
left-padded with 2 zeros along time (same causality treatment as
023-canon-conv — not `padding=2`, which would leak future tokens).
**Heads-axis conv from the paper is deliberately dropped** because
at tiny1m3m `n_heads ≈ 6–8` a depthwise conv over the heads dim is
approximately identity (LoC cost without a math change). The
remaining lever is the Q/K time-axis half of the paper, which is
the half with the cleanest mechanistic argument (n-gram-conditioned
pairwise scores). One new module `MultiTokenQK` (DWConv on Q +
DWConv on K + two scalar gates) plus a `use_mta_qk` flag in
`LLMConfig` / `TransformerBlock` and a new
`Tiny1M3MMTAonCanonFireConfig` ctrl/trt. **Stacks on the 023 WIN
ctrl** (`Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True +
use_canon_conv=True`); trt adds `use_mta_qk=True`. ~40 LoC: well
under the 200 ceiling.

## Scale evidence
The paper reports gains on language modeling and on long-context
retrieval/search benchmarks from richer local context inside the
score. It does not advertise one giant pretrain run — evidence is
broad but mid-scale, so `transfer-risk: med`. Heads-axis conv is
**deliberately de-scoped for the tier** (n_heads ≈ 6–8 makes a
depthwise heads-axis conv approximately identity at tiny1m3m);
the mechanism filed here is the time-axis Q/K half, which is the
half with the most direct score-side argument. The Q/K time-axis
conv is identity-init (zero scalar gate) and parameter-cheap
(`2 × d_model × k = 2 × d_model × 3` weights per block, plus two
scalar gates), so the LoC and param costs are both well bounded.

## Why it's worth a slot
**Bet (one sentence):** MTA on Q and K (depthwise k=3 time-axis,
delta-init, **stacked on Canon**) lowers val loss by ≈ −0.01 or
more because QK scores are still computed from **single-token**
projections, and an n-gram-conditioned score is a **new degree
of freedom** that residual-side local mixing (Canon) cannot
express — Canon mixes the residual stream, but the attention
*score* is still a single-token dot product.

**Two informative outcomes (so the null still teaches us):**
- **WIN** (`trt < ctrl − 0.01`): score-side local mixing is a
  **separate lever axis** from residual-side local mixing; the
  recipe takes both into the 10M → 135M ladder. Orthogonality
  with the 023 winner is established, and 046 ships.
- **NULL** (`|trt − ctrl| ≤ 0.01`): Canon's residual conv already
  captured the local-mixing degree of freedom at this scale and
  the QK conv is redundant — closes the orthogonality question
  for this cluster and frees 10M+ budget for other axes. Still
  informative: the 10M → 135M ladder does not need a separate
  score-side mixer, and we do not re-test it.
- **FAIL** (`trt > ctrl + 0.01`): the QK conv actively hurts on
  top of Canon (over-mixing, gate-learned value negative, or
  projection-shift mismatch). Reject; close.

**Portfolio orthogonality (the taste reviewer's crowding concern,
addressed):** 020–030 is dense with **inside-attention**
replacements — forgetting, softpick, gated, ssmax, qk-norm,
v-norm, fire×qknorm, moonlight×qknorm all rewrite the score or
value computation *after* Q/K are projected. MTA-on-QK is
**pre-score** (a convolution on the projected Q and K streams
*before* the per-head reshape) and is **outside** that family.
It is the closest cousin to 023-Canon (residual-side local
mixing); the repitch is the one that explicitly A/Bs against
Canon. So 046 is the *right* attention-adjacent slot to fill
after 023, not a redundant 9th in the 020–030 cluster.
