---
id: 041-talking-heads
status: reviewing
round: 1
updated: 2026-06-10T23:50:36Z
transfer-risk: med
---

# 041 — Talking-Heads Attention (post-softmax weight mix)

## Source
Shazeer et al., "Talking-Heads Attention" (arXiv:2003.02436), 2020. The
paper actually proposes two cross-head mixes: a *pre-softmax score mix*
(the dominant lever per their ablations) and a *post-softmax weight mix*
(the variant the name "talking heads" comes from — heads literally
communicate via the soft weights). This repitch commits to the
**post-softmax weight mix only** (one H×H on the probability tensor),
not the pre-softmax mix, so the slot is orthogonal to 042-knocking-heads
(Q/K/V feature mix, pre-attention).

## Mechanism
Standard MHA: scores = Q·Kᵀ/√d, P = softmax(scores) per head, O_h =
P_h·V_h, then concat heads and project with W_O. Talking-heads inserts
a learned H×H matrix **after softmax and before the per-head V-weighted
sum**, on the probability tensor P (B, H, T, T). The mix is across H
only — it permutes/linearly combines the soft-weight vectors that each
head will use to read from V. Both H×H matrices (or in this repitch,
just the post-softmax one) are initialized to identity, so at step 0 the
output is bit-exact baseline MHA: zero parameter overhead, zero FLOPs
overhead, clean A/B.

At tiny1m3m (n_heads=4, n_kv_heads=2 GQA), the mix is a 4×4 matrix
acting on the 4-head probability tensor. The score tensor also has H=4
(scores are computed from MHA-side Q against shared K/V groups), so a
4×4 mix on H spans the full linear span of the head space. ~16
parameters and one einsum per block.

## Scale evidence
Shazeer et al. report perplexity and downstream-task gains on T5
pretraining (hundreds of millions of params, fixed compute per layer)
and on ALBERT/BERT fine-tuning. The T5 evidence is mid-scale pretraining
with paper-reported head counts of 12-24, not a 100M+ CLM head-to-head.
We rate `transfer-risk: med` because (a) the mechanism is real and
complements the cross-head W_O that already exists, and (b) the
head-count gap between paper and tiny1m3m means the gain at H=4 is
mechanistically informative even if numerically small.

## Why it's worth a slot
We expect a **measurable but small** win at H=4 because the post-softmax
weight mix is a *second-order* communication channel on top of the
concat+W_O that already exists — at H=4, heads have very little
individuality, so reweighting their soft votes may be redundant. The
bet: if even a 4×4 mix on the probability tensor moves val loss at
H=4, head-isolation is a real constraint and the 135M recipe (H≈8-12)
should keep this slot; if null, head-isolation is not the binding
bottleneck and 135M can skip post-softmax mixing and spend the
parameters elsewhere (e.g., a second W_O, or a head-axis MoE).

**Sharpened null** (specifically not the generic "head-mixing isn't
binding"): at H=4 GQA, the concat+W_O already mixes the 4 head
outputs; a null on a 4×4 post-softmax mix tells us the *probability-side*
cross-head term is redundant when concat-side mixing is unconstrained.
That rules out post-softmax weight mixing as a 135M recipe slot — not
just a generic "head mixing doesn't help."

**Why a distinct slot from 042-knocking-heads**: 042 inserts the
cross-head mix on Q/K/V *features* (B, H, T, d) before the dot product,
which is the pre-attention channel. 041 inserts the cross-head mix on
the *attention probabilities* (B, H, T, T) after softmax, which is the
post-attention channel. Two different tensors, two different
information pathways. Identity init on both keeps the A/B clean.

**Why identity init is the redeeming feature** (and why the head-count
gap doesn't kill the slot): a 4×4 H×H mix at identity init is the
*full* rank linear transformation on the 4-head space — it's not
"near-trivial" in a math sense, it's exactly the constraint we want
to test. The full-rank-on-4-dim property means a win is a real
positive signal even at H=4, and a null cleanly rules out the
probability-side channel.
