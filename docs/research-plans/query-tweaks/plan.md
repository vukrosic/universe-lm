# Query / W_Q tweaks — consolidated plan

All query-side ideas in one place. Round 1 = original list **with critique verdicts applied** (see [critique.md](critique.md)). Round 2 = next set of new ideas.

Context: `q_size == d_model` (W_Q square). Already shipped on Q: query-embed (#30), scalar `q_gain` (#37), tied-QK (#72), QK-norm position (#49), `rope_base` (#63), `attn_sink` (#99). Wire new flags in [models/layers.py](../../../models/layers.py) attention forward + a `Screen10M20M<Name>Config`.

## Protocol (addresses critique #3, #4)
- Baseline = clean `Screen10M20MConfig`; record 3-seed mean val loss before claiming anything.
- Screen tier **does not** transfer-promote — any winner re-runs on the Full ladder (≥25M) before it's a claim.
- Every new lever must be **identity/zero-init** (step-0 == baseline) so the A/B isolates the mechanism, not a re-seed.
- Confirm optimizer routing: 2D params (mixing matrices, low-rank) → Muon; per-head scalars/bias → AdamW. Verify zero/identity-init params actually get gradient under Muon.

---

## Round 1 — adjusted after peer review

| # | Idea | Status | Adjusted spec | Conf | Why |
|---|---|---|---|---|---|
| 1 | Per-head Q bias | **Revise → run** | `Q += b_h` applied **after q_norm and after RoPE** (constant prior, not rotated). A/B head-to-head vs `attn_sink`. | med | Token-independent attention anchor: adds `b_h·k_j` to every score, a learned content prior. Must beat the sink to justify. |
| 2 | Talking-heads on Q | **Keep (flagship)** | Implement the **real** version: mix attention **logits** across heads via learned `n_heads×n_heads` M (M=I init), not Q-vector mixing. | med | Cross-head information flow — nothing in the repo does it. ~36 params/layer at 6 heads. |
| 3 | Q centering | **Revise** | Rewrite hypothesis: subtracting `mean(Q)` does **not** cancel in softmax (changes each logit by `mean(Q)·k_j`). Frame as a real per-key effect or drop. | low | Only worth it if reframed as a genuine score modifier; the "stabilizer/no-op" rationale was wrong. |
| 4 | Per-channel Q-gain | **Revise (narrow)** | Pre-RoPE per-channel scale = existing `q_norm` affine → skip. Only test **post-RoPE per-*pair*** scale (respects RoPE rotation pairs). | low | Justify vs scalar `q_gain`; suspect it adds little. Low priority. |
| 5 | Identity-init Q-proj | **Reclassify** | Treat as an **init study**, not a mechanism (brushes the excluded "init std" rule). Run only as a cheap curiosity. | low | Init tricks usually wash by the Full ladder; Muon may re-discover standard init in a few steps. |
| — | Learnable softmax temp | **Killed** | Identical to shipped `q_gain` (#37). | — | No experiment. |
| — | Low-rank Q residual | **Killed** | W_Q already full-rank & trainable → low-rank residual adds zero expressivity (LoRA logic needs a frozen base). | — | No experiment. |
| — | Per-head RoPE-freq (Q-only) | **Killed** | Q-only freq scale breaks RoPE's relative-position identity; both-sides = `rope_base` (#63). | — | No experiment. |

**Run order (round 1):** Talking-heads (logit-mixing) → Per-head Q bias (vs `attn_sink`). Defer 3–5.

---

## Round 2 — next set of ideas

New Q-side mechanisms, checked against existing flags (none duplicate `q_gain`/`q_norm`/`rope_base`/`attn_sink`).

| # | Idea | What | Extra params/block | Step-0 == base? | Conf | Why |
|---|---|---|---|---|---|---|
| 6 | Learned distance bias (ALiBi-style) | scores `+= -m_h·(i−j)`, per-head slope `m_h` | n_heads | m=0 → yes | med | A *non-rotary* positional prior, additive on logits. Distinct from RoPE; can stack with it or partially replace it. Known to transfer (ALiBi). The repo only has RoPE/NoPE — this is a fresh positional axis on the Q-side score. |
| 7 | Token-conditioned query temperature | `Q *= (1 + tanh(x·w_h))` per head — temperature **from the token** | n_heads·d_model | w=0 → yes | med | Strictly more expressive than static `q_gain`: each token sharpens/softens its own attention per head. Tests whether attention temperature should be *dynamic*, not a fixed scalar. Not redundant. |
| 8 | Query expansion (multi-query/token) | project Q to `2·q_size`, run 2 attention reads/head, mean the outputs | +q_size (proj) | 2nd query zero-init → yes | low-med | Cheap capacity on the query side only (K/V untouched). Tests if a token benefits from issuing >1 query per head — a poor-man's mixture-of-queries. |
| 9 | Query gate from residual norm | per-head scalar gate `σ(a_h·‖x‖ + b_h)` scaling Q | 2·n_heads | gate≈1 init | low | Lets high-energy tokens attend more sharply (norm-conditioned). A 2-param/head dynamic alternative to #7; cheaper, weaker. |
| 10 | Antisymmetric Q–K coupling | add a small learned skew term to the Q·K score (`Qᵀ S K`, S skew-init 0) | d_k² (shared) | S=0 → yes | low | Lets attention encode *ordered/directional* relations beyond the symmetric dot. Exotic; only if 6/7 wash. |

**Run order (round 2):** ALiBi-style bias (#6) → token-conditioned temperature (#7). #8 if either wins; #9/#10 are long shots.

---

## Round 3 — expanded idea bank

Grouped by axis. "Lift" = implementation size (S = few lines, M = a module, L = real surgery). All must keep the identity/zero-init rule unless noted.

### A. Positional (Q/K rotation & position priors)

| # | Idea | What | Params/block | Step-0 base? | Lift | Conf | Why |
|---|---|---|---|---|---|---|---|
| 11 | Per-head learnable RoPE base | each head gets its own rotary frequency (both Q&K), learnable, init = global base | n_heads | yes | S | med | Heads pick their own positional resolution — multiscale position without multiscale windows. Generalizes the global `rope_base` (#63). |
| 12 | Partial rotary | rotate only a fraction `p` of Q/K dims, leave the rest position-free (GPT-NeoX/J) | 0 | yes (p=1) | S | med | Lets some channels be pure content, some positional. Cheap, known to help; a structural middle ground between RoPE and NoPE. |
| 13 | Decoupled content/position attention | two score streams — content `Q_cᵀK_c` + position `Q_pᵀK_p` — summed (DeBERTa/TXL disentangled) | +2 small projs | yes (pos zero-init) | L | med | Separates "what" from "where" explicitly instead of entangling both in one rotated dot. Strong in the literature. |
| 14 | CoPE (contextual position) | position counted by gated tokens, not raw index; query controls the gate | small | yes | L | med-low | Position measured in *semantic* units (e.g. "2 sentences back"). Recent strong result; biggest lift here. |

### B. Score / similarity function

| # | Idea | What | Params/block | Step-0 base? | Lift | Conf | Why |
|---|---|---|---|---|---|---|---|
| 15 | Per-channel relevance (bilinear) | score = `Qᵀ diag(d_h) K`, learnable `d_h` (init 1) | n_heads·d_k | yes | S | med | Learns *which channels* matter for matching, beyond the uniform dot. Distinct from per-head scalar temperature. |
| 16 | Cosine attention | L2-normalize Q&K, then learnable per-head temperature | n_heads | ~yes | S | med | Bounds the logits (Swin-v2 fix): decouples match direction from magnitude, kills attention-logit blow-up. |
| 17 | Additive (Bahdanau) term | blend in `wᵀ tanh(W_qQ + W_kK)`, gated zero-init | small | yes | M | low | Tests whether the dot-product similarity itself is the bottleneck. Exotic but a clean question. |
| 18 | QK shared low-rank factor | factor W_Q, W_K through a shared bottleneck (partial tie, not full like #72) | saves params | no | M | low-med | Between independent QK and fully-tied QK — tests the right amount of Q/K coupling. |

### C. Capacity / structure on the query

| # | Idea | What | Params/block | Step-0 base? | Lift | Conf | Why |
|---|---|---|---|---|---|---|---|
| 19 | Register / memory tokens | k learnable persistent K/V slots every query can attend (value zero-init) | k·2·d_model (shared) | yes | M | med | Global scratchpad / attention off-load. The *learnable-content* generalization of the zero `attn_sink` (#99). |
| 20 | Query subspace bottleneck | low-rank W_Q (rank r < d_k) — compress the query | saves params | no | S | low-med | Tests whether full-rank queries are needed; frees budget for depth at fixed total. (The meaningful inverse of the killed low-rank *residual*.) |
| 21 | Mixture-of-queries | router picks 1 of N query projections per token/head | N·q_size | yes (pick base) | L | low | Conditional query computation — more capacity at ~constant FLOPs. MoE on the Q path. |
| 22 | FFN-gated query temperature | tiny per-head MLP on the token produces the gain (richer than linear #7) | small MLP | yes | M | low-med | Non-linear, token-conditioned sharpening — upper bound on how much dynamic temperature can buy. |

### D. Stability / regularization

| # | Idea | What | Params/block | Step-0 base? | Lift | Conf | Why |
|---|---|---|---|---|---|---|---|
| 23 | Head-orthogonality penalty | aux loss pushing heads' Q subspaces apart | 0 | yes | S | low | Reduce head redundancy, force specialization. Borderline "mechanism vs regularizer" — keep low priority. |
| 24 | Query dropout | drop Q channels during training | 0 | yes | S | low | Regularizes the attention pattern; cheap to try, weak prior at this scale. |
| 25 | Attention-logit entropy reg | penalize over-peaked/over-flat attention | 0 | yes | S | low | Shapes attention sharpness without a hard temperature. Tuning-adjacent — lowest priority. |
| 26 | Q centering, per-key correct | (from R1 #3, promoted) subtract a *learned* per-head query offset, justified as a real per-key shift | n_heads·d_k | yes (0 init) | S | low-med | The honest version of round-1 centering: a learnable Q offset that genuinely moves logits per key. |

**Triage across all rounds — run these first:** Talking-heads logit-mixing (R1-2), ALiBi bias (#6), per-channel relevance (#15), cosine attention (#16), register tokens (#19). Big-but-promising bets: decoupled content/position (#13), CoPE (#14).

---

## Status / results
(add per-idea branches, A/B numbers, transfer checks here)
