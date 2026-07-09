---
id: 042-knocking-heads
status: needs-plan
round: 1
updated: 2026-06-11T01:34:07Z
transfer-risk: low
---

# 042 — Knocking-Heads Attention (Q/K/V feature mix, pre-attention)

## Source
Zhou et al., "Knocking-Heads Attention" (arXiv:2510.23052), 2025. The
paper's headline lever is a *pre-attention* cross-head mix on the
**Q, K, V feature channels** (the per-head d_k axis), before the
dot-product. The paper also tests post-attention variants; this
repitch commits to the **Q/K/V feature mix only** (three H×H
matrices, one per projection), which is the dominant lever in the
paper's ablations and is the **feature channel** that 041-talking-
heads explicitly stepped away from in its round-2 repitch.

## Mechanism
Standard MHA at our tier: x ∈ R^{B,T,d} → reshape to (B, H, T, d_k)
→ W_Q/W_K/W_V per-head projection → attention. Knocking-heads
inserts, after the W_Q/W_K/W_V projection and before the dot-product,
a learned **H×H matrix M_Q, M_K, M_V** (one per projection) applied
along the H axis of the (B, H, T, d_k) tensor:

    Q' = einsum('ij,bjtd->bitd', M_Q, Q)   # and same for K, V

This is *feature-channel* mixing: each head's per-position feature
vector becomes a learned linear combination of all H heads' feature
vectors. The Q·Kᵀ dot product then sees a richer "feature
committee" per position, even with H=4.

**Initialization = identity**: each M_* is initialized to I_H with
no 1/H or 1/√H rescale, so Q'[b,h,t,d] = Q[b,h,t,d] at step 0. Step
0 is byte-for-byte identical to baseline MHA — the ablation is
parameter-clean (192 free parameters per block at H=4, d_k=16) and
has no FLOP overhead beyond three small einsums. ~30 LoC in
`models/layers.py` + a `use_knocking_heads: bool` flag in
`LLMConfig`, well under the 200-LoC budget.

### Why this fires at H=4 (mechanistic argument, not "trust the paper")
At H=4, d_k=16, the mix is small in absolute terms (4×4 matrices)
but it does something **structurally distinct** from anything else
in the repo:

- Per-head scalars (the `use_q_gain` lever in idea 037, and the
  symmetric K-gain in 042) act on a single dimension per head —
  they reweight, they don't *mix*.
- Concat + W_O at the end of attention mixes the H outputs *after*
  they've already been softmax-weighted by V — a post-attention
  channel, conceptually downstream of the dot-product.
- The knock mix sits **before** the dot-product, on the features
  that the dot-product sees. At H=4, the 4×4 mix is the *full*
  rank-4 linear span of the head-feature space, so even a tiny
  off-diagonal entry changes which feature basis each head
  effectively attends to.

The bet is that with H=4 heads, a position can carry a richer
feature basis at no head-count cost — the heads effectively share
their feature spaces via a learned committee. The paper's billion-
scale gain is one signal that the mechanism is real; the H=4
argument is that the *cross-head feature-share* degree of freedom
is one we currently have *zero* of in this repo, so even a small
measurable move is informative.

## Scale evidence
Zhou et al. report large-scale pretraining on 1T tokens with a
6.1B total / 1.01B active MoE model, plus 2.3B-14.6B MoE and
0.61B-3.94B dense ablations. Transfer-risk is **`low`** because the
mechanism already held up at billion-scale and reduced training
spikes. **Important scope note**: the "reduced training spikes"
half of the paper's headline gain is a *stability* effect over 1T
tokens and **almost certainly will not fire in tiny1m3m's ~92 update
steps**. This slot is betting on the *structural feature-mixing*
half of the gain, not the spike-suppression half. A null on spike
suppression at our tier is **not informative** — it's an
out-of-scope test, and we will not interpret it against the paper.

The H=4 head-count gap (paper's smallest dense ablation is
~0.61B, presumably H≥12) is real and the lever's absolute gain at
H=4 will be smaller than the paper's at H=12-16. The bet is that
the *direction* of the effect transfers.

## Why it's worth a slot
**Sharp bet**: at tiny1m3m (H=4, GQA n_kv=2) and the 700-step
milestone, we expect **Δval ∈ [-0.010, -0.025]** with a **pass bar
of -0.005**. The mechanism is the dominant paper lever, applied
at a head-count 3-4× below the paper's tested range, so we anchor
on a small-but-positive expected effect — any measurable move
favors keeping the slot for 135M, and a clean null with the
correct sign-magnitude partitions the failure modes.

**Pass/fail partitioning of the null** (this is the part the prior
pitch was vague on):
- **Δ ≥ -0.005 (flat or hurt)**: cross-head feature sharing
  actively misfires at H=4, *or* the diagonal init has a
  calibration bug (we'll A/B against an H=4 head-count scaling
  test if this hits). Drop the slot.
- **-0.010 ≤ Δ < -0.005 (sub-bar small)**: cross-head feature
  sharing is below our H=4 head-count threshold. Re-test at
  H=8-12 on the screen20m ladder — the lever may simply need
  more heads to express.
- **Δ ≤ -0.010 (pass)**: hit, scale up to 135M as a recipe
  candidate.
- **Δ ≤ -0.025 (strong pass)**: H=4 has more headroom than
  expected; the recipe can keep a small per-block H×H mix even at
  135M (H≈8-12) and probably grow it.

**Why a distinct slot from 041-talking-heads** (re-stating the
partition 041's round-2 repitch established, to lock the division):
- 041 inserts its H×H mix on the **attention probabilities**
  (B, H, T, T) **after** softmax — the *post-attention* channel.
- 042 inserts its H×H mix on the **Q/K/V features** (B, H, T, d_k)
  **before** the dot product — the *pre-attention* channel.
- Two different tensors, two different information pathways.
  A 041 hit + 042 null would localize the gain to the
  probability-side channel (and the 135M recipe would keep
  041 only); a 042 hit + 041 null would localize to the
  feature-side channel (and 135M would keep 042 only); a double-
  hit confirms the cross-head mix is useful in *both* channels
  and 135M could consider stacking; a double-null cleanly rules
  out the entire cross-head-mix family as a binding bottleneck
  and frees the parameter budget for other slots (e.g., a second
  W_O or a head-axis MoE, as 041's repitch notes).

**Identity init, explicit**: M_Q = M_K = M_V = I_4 (no 1/H, no
1/√H rescale). Q'[b,h,t,d] = sum_h' I[h,h'] · Q[b,h',t,d] =
Q[b,h,t,d] exactly. Step 0 is byte-for-byte baseline; a null
cannot be blamed on a wiring artifact. The implementer will
verify with a `torch.allclose(model(use_knocking_heads=True,
step=0), model(use_knocking_heads=False, step=0))` test before
launching the GPU run.

**Why the head-count gap doesn't kill the slot**: same argument
as 041's repitch — the H×H mix at identity init is the *full*
rank-4 linear span of the head-feature space. A 4×4 mix is not
"near-trivial" in a math sense; it spans exactly the linear
subspace we're testing. The downside is the gain magnitude will
be smaller than the paper's at H=12-16; the upside is the
informativeness — a hit is a real positive signal, a null is a
real negative signal, and we have a clean pass/fail bar that
selects the right next move in each case.
