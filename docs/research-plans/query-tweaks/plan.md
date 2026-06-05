# Query / W_Q tweaks — consolidated plan (29 experiments, 6 batches)

All query-side ideas in one place. **Round 1** = original 26-idea bank
(see [critique.md](critique.md)). **Round 2** = filled out to 29
experiments across 6 batches. Every row in this plan has a 1:1 entry in
[manifest.md](manifest.md) (the implementation checklist).

Context: `q_size == d_model` (W_Q square). Already shipped on Q:
query-embed (#30), scalar `q_gain` (#37), tied-QK (#72), QK-norm
position (#49), `rope_base` (#63), `attn_sink` (#99). New flags wire
into [models/layers.py](../../../models/layers.py) `MultiHeadAttention`
forward + a `Screen10M20M<Name>Config` recipe.

---

## Protocol (addresses critique #3, #4)

- Baseline = clean `Screen10M20MConfig`; record 3-seed mean val loss
  before claiming anything.
- Screen tier **does not** transfer-promote — any winner re-runs on
  the Full ladder (≥25M) before it's a claim.
- Every new lever must be **identity/zero-init** (step-0 == baseline)
  so the A/B isolates the mechanism, not a re-seed. **One exception:
  Q27 (feature-map attention) is explicitly not identity-init — see
  note below.**
- Confirm optimizer routing: 2D params (mixing matrices, low-rank) →
  Muon; per-head scalars/bias → AdamW. Verify zero/identity-init
  params actually get gradient under Muon.

---

## Implementing-AI notes (read these before wiring)

1. **Batch 4 needs one prerequisite wire.** The current
   `qk_norm_type` flag ties Q and K to the *same* norm. Batch 4
   sweeps Q-specific norms (pnorm, clip, channelscale, manhattan,
   center, none) on Q *only* (K stays at the default `rmsnorm`).
   Solution: add a separate `q_norm_type` flag that **defaults to the
   value of `qk_norm_type`** at config-construction time (so existing
   configs are bit-identical), and overrides independently when set.
2. **Q27 (feature-map attention) is NOT identity-init.** It replaces
   `Q @ K.T / sqrt(d_k)` with a learned feature map `phi(Q) @ phi(K).T`
   where `phi` has its own learnable parameters. Step-0 output is
   *not* the standard softmax baseline — it has a different geometry.
   The control for Q27 must be `phi = identity` (or
   `phi = linear projection that starts as identity`), and the
   interpretation is "feature map is an upgrade over dot-product,"
   not "Q27 helps the same way other levers help." Plan a separate
   gated control run for Q27.

---

## Batches

### Batch 1 — high-signal levers (Q1–Q4)

Cheap, identity-init, expected to be the most-likely-to-win batch.
Run on `screen20m` from the start (no tiny detour needed for these).

| # | Idea | Spec | Step-0 base? | Params/block |
|---|---|---|---|---|
| Q1 | AlibiBias (per-head linear distance) | `scores += -m_h · (i - j)`, per-head learnable slope `m_h` | m=0 → yes | n_heads = 6 |
| Q2 | QTempToken (token-conditioned temperature) | `Q *= (1 + tanh(x · w_h))` per head — temperature **from the token** | w=0 → yes | n_heads·d_model = 864 |
| Q3 | CosineAttn (L2-norm + learnable temp) | Q,K L2-normalize; per-head learnable temperature | τ=1 → yes (≈) | n_heads = 6 |
| Q4 | QKBilinear (per-channel relevance) | `score = Q^T · diag(d_h) · K`, learnable per-channel `d_h` (init 1) | d=1 → yes | n_heads·d_k = 144 |

### Batch 2 — flagship + positional (Q5–Q7)

A flagship mechanism (talking-heads, the real version) and two
positional axes that haven't been wired.

| # | Idea | Spec | Step-0 base? | Params/block |
|---|---|---|---|---|
| Q5 | TalkingHeadsQ (logit-mix across heads) | learned `n_heads × n_heads` M (M=I init) applied to attention **logits** pre-softmax | M=I → yes | n_heads² = 36 |
| Q6 | PerHeadRopeBase (per-head rotary freq) | each head gets its own learnable `θ_h` (init = global `rope_base`) — applied to both Q and K | θ=base → yes | n_heads = 6 |
| Q7 | PartialRotary (rotate fraction p of Q/K) | rotate only `p` of Q/K dims, leave rest position-free (GPT-NeoX/J) | p=1 → yes | 0 |

### Batch 3 — exotic (Q8–Q10)

Only run after 1+2 are screened. Exotic mechanisms with low prior but
cheap to wire.

| # | Idea | Spec | Step-0 base? | Params/block |
|---|---|---|---|---|
| Q8 | QExpansion (multi-query/token) | project Q to 2·q_size, run 2 attention reads, mean outputs | 2nd-query zero-init → yes | +q_size (proj) = 144 |
| Q9 | DecoupledContentPos (DeBERTa) | two score streams: `Q_c^T K_c` + `Q_p^T K_p`, summed | pos zero-init → yes | 2 small projs |
| Q10 | AntisymQK (skew coupling) | add `Q^T · S · K` term, S skew-init 0 | S=0 → yes | d_k² shared = 576 |

### Batch 4 — query-norm zoo (Q11–Q16)

Sweep Q-side normalization. **Requires `q_norm_type` flag** that
defaults to `qk_norm_type`. Each config = "set `q_norm_type` to X."
Sweep is "nearly free" — no model code, just routing.

| # | Idea | Spec | Step-0 base? | Params/block |
|---|---|---|---|---|
| Q11 | Q-pnorm | `q_norm_type = "pnorm1.5"` (p=1.5 Lp norm) | yes | 0 |
| Q12 | Q-clip | `q_norm_type = "clipnorm3"` (winsorize k=3) | yes | 0 |
| Q13 | Q-channelscale | `q_norm_type = "channelscale"` (learnable pre-scale) | yes | d_k = 24 |
| Q14 | Q-manhattan | `q_norm_type = "manhattan"` (L1 MAD) | yes | 0 |
| Q15 | Q-center | `q_norm_type = "center"` (mean-only) | yes | 0 |
| Q16 | Q-none | `q_norm_type = "none"` (skip Q norm entirely) | yes (K still normed) | 0 |

### Batch 5 — learnable-param zoo (Q17–Q23)

A zoo of small learnable additions to the Q path. Sweep "what if Q
got *just one more thing*." All identity/zero-init.

| # | Idea | Spec | Step-0 base? | Params/block |
|---|---|---|---|---|
| Q17 | per-head bias (b_h) | `Q += b_h` after q_norm and RoPE | b=0 → yes | n_heads·d_k = 144 |
| Q18 | per-channel gain (g_d) | `Q *= g_d` (per-channel) after RoPE | g=1 → yes | d_k = 24 |
| Q19 | head×channel gain (g_hd) | `Q *= g_hd` (per-head×channel) after RoPE | g=1 → yes | n_heads·d_k = 144 |
| Q20 | norm-gate (gated by ‖x‖) | per-head scalar `g_h = σ(a_h·‖x‖ + b_h)` | g≈1 → yes | 2·n_heads = 12 |
| Q21 | low-rank refine (Q += u·v^T·x) | per-layer low-rank residual: `Q ← Q + (W1·x) @ W2` | zero-init → yes | 2·r·q_size |
| Q22 | LayerScale on Q | multiply Q by `(1 + ls_q)` (per-channel) post-RoPE | ls=0 → yes | d_k = 24 |
| Q23 | softplus gain (Q *= softplus(g_h)) | per-head positive scalar — always >= 0 | g=0 → yes | n_heads = 6 |

### Batch 6 — architecture / mixing (Q24–Q29)

Bigger / weirder. Sweep "what if the Q path is structured differently."
Q27 is **not identity-init** (see note above).

| # | Idea | Spec | Step-0 base? | Params/block |
|---|---|---|---|---|
| Q24 | head-mix (Q mix across heads pre-attention) | `Q_mixed[h] = Q[h] + Σ_h' M[h,h']·Q[h']` (M=I init) | M=I → yes | n_heads² = 36 |
| Q25 | time-conv (1D conv on Q across positions) | `Q += conv1d(Q, kernel=3)` (zero-init) | zero → yes | 3·d_k·d_k = 1728 |
| Q26 | EMA-smooth (Q ← α·Q + (1−α)·Q_prev) | per-layer EMA over the position axis | α=1 → yes | 0 |
| Q27 | feature-map (phi(Q) phi(K)^T) | **NOT identity-init** — phi is a small learnable MLP; needs own control | — | small MLP |
| Q28 | per-token-rope (per-token rotary freq) | each token has its own θ_t (small per-token MLP produces it) | θ=base → yes | small MLP |
| Q29 | noise-reg (add noise to Q during training) | `Q += N(0, σ²)` in training only, σ learnable scalar | σ=0 → yes | 1 |

---

## Run order

- **Batch 1 first** (Q1–Q4) — most likely to win, run on `screen20m` directly.
- **Batch 2 next** (Q5–Q7) — flagship + positional; Q5 (talking-heads)
  is the big swing, Q6/Q7 are safer.
- **Batch 3 last** (Q8–Q10) — exotic; only run if 1+2 have a signal.
- **Batches 4–6 are breadth screens.** Run on `tiny1m` first; promote
  only the top ~2 per batch to a 3-seed `screen20m`. **Net: 19
  small-cost ideas vs 19 full screen20m runs.**

Per-idea results land in the Status section below. Each row gets a
tag, metrics.json, and a "wins / wash / loses" verdict vs the
screen20m control.

---

## Status / results

(add per-idea branches, A/B numbers, transfer checks here)
