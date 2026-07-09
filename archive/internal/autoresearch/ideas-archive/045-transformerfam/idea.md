---
id: 045-transformerfam
status: tasting
round: 2
updated: 2026-06-10T23:49:43Z
transfer-risk: high
---

# 045 — TransformerFAM (feedback attention as working memory)

## Source
Hwang et al., "TransformerFAM: Feedback attention is working memory"
(arXiv:2404.09173), 2024. Paper's published gains are on long-context
tasks at 1B / 8B / 24B; FAM has not been measured at <1B or on dense
pre-training val loss.

## Mechanism
Feed the model's own latent states back into a second attention
pass so each token can read a compact working-memory trace from
earlier computations, with the feedback gate zero-initialised so
the step-0 path is the unmodified baseline. Concretely: take the
block's post-attention hidden state, project it to a small
bottleneck (≤ d/4), and concatenate that bottleneck with the
per-token K/V of the *next* block. The feedback gate is a learned
scalar per block, init to 0, multiplying the projected memory
before it is added to K/V; a single transformer-block add plus one
extra small linear projection keeps the implementation under ~100
LoC. No tokenizer change, no seq-len change, no extra recurrence
across the sequence axis.

## Scale evidence
Largest published gain: long-context QA / retrieval at 1B–24B.
**Transfer-risk for our tier is `high`**: the published mechanism
only fires when the attention window is the bottleneck. Our
seq=2048 fits comfortably in a single attention layer's window at
d=192 / 6 heads, so the recurrent feedback loop has no extra
information to read that the standard K/V pass does not already
attend to. The FAM loop is essentially a slower re-read of the
same receptive field. The tag reflects *our* tier (tiny1m3m,
seq=2048, 3M tokens), not the paper's — corrected from r1's `low`
(021 / 023 / 024 results also show the family is hard to isolate
at this scale, see the marginal-contribution note below).

## Why it's worth a slot — the bet, sharpened
**Single-sentence hypothesis:** with a proper FIRE-equipped
control (`trt = FAM + FIRE` vs `ctrl = FIRE alone`), expect
`Δ_val ≤ -0.02` because the recurrent feedback loop gives the
model an *explicit, time-indexed* working memory that a static
cross-block V shortcut (021 V-residual) and a content-routed
cross-block attention (044 AttnRes) cannot, *or* — more likely
at this scale — a clean null. **A clean null is informative**: it
closes the *feedback-loop subfamily* of cross-layer shortcuts at
our tier (seq=2048, 6L, 0.94M params), which lets us move on from
recurrent-style levers and focus the remaining portfolio on
static shortcuts (V-residual, AttnRes) plus the actually-different
recurrent family (020 Forgetting-Attn, 025 SSMax edit-priors).
The A/B is cheap either way: zero-init gate, ~100 LoC, single
fire-equipped ctrl, seed 42.

### Marginal contribution vs the cluster
This is the 3rd distinct cross-layer / recurrent attention
mechanism filed in two weeks. The portfolio is crowded but the
axes are now distinct:
- **021 V-residual (done, WIN w/ caveat):** static, additive V
  carry from block `i` to block `i+1` — a *value* shortcut, no
  routing, no time index.
- **044 AttnRes (tasting):** learned softmax over *all past block
  updates* — a *content-routed* cross-block shortcut, no
  recurrence, no time index.
- **045 FAM (this):** recurrent feedback of the block's own
  hidden state into the *next* block's K/V — adds a *time index*
  (block position) and a non-linear projection through a small
  bottleneck; closest analog in the cluster is 020
  Forgetting-Attn, which is a different forget gate inside the
  same block's attention rather than a feedback path between
  blocks.

So the marginal-contribution question this A/B answers is
concrete: **does a time-indexed recurrent feedback path between
blocks beat (a) a static V shortcut (021) and (b) a content-routed
softmax over past updates (044) at our scale?** If yes, the
feedback-loop family is open and we keep mining it. If no, the
family is closed at tiny1m3m regardless of recurrence and we
route the remaining attention-prior budget to the static-shortcut
and edit-prior subfamilies.

## Constraints preserved from r1
- Zero-init feedback gate (step-0 ≡ baseline, asserted by a unit
  test mirroring 023-canon's gate-of-zeros identity check).
- No tokenizer / seq-len / data change; no >200-LoC block.
- Single seed (42); no multi-seed, no per-seed means.
- FIRE-equipped control to isolate the FAM effect (same
  correction as 021/024/025 — the 009 FIRE lever is in the
  baseline, the ctrl must match).
