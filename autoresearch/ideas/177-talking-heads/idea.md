---
id: 177-talking-heads
status: needs-taste
round: 1
updated: 2026-06-15T01:35:00Z
transfer-risk: med
plain: Let attention heads talk to each other: each head's pre-softmax scores are mixed through a learned H×H matrix (and similarly post-softmax on the value output), starting at identity so step-0 is byte-identical.
---

# 177 — Talking-Heads Attention (Learnable Cross-Head Linear Mix on Pre-Softmax Scores)

## Source
- Shazeer, Lan, Cheng, Mao, Le, "Talking-Heads Attention" (arXiv:2003.02436, March 2020). The paper inserts two learnable H×H linear projections: one on the **pre-softmax logits** (cross-head logit mixing) and one on the **post-softmax outputs** (cross-head value mixing). Both projections are initialized to identity (no-op at step 0). Validated on Transformer-Big WMT'14 En-De/En-Fr (~220M params): +1.0 BLEU on En-De vs baseline Transformer-Big, +0.5 on En-Fr. Adopted in some recent architectures as a cheap parameter-shared mixing layer.
- In-repo context: the mechanism is **already implemented** at `models/layers.py:1796-1800` (pre-softmax `talking_heads_M = nn.Parameter(torch.eye(n_heads))`) and `:1929-1933` (post-softmax `talking_heads_out_M = nn.Parameter(torch.eye(n_heads))`). Identity init ⇒ bit-identical at step 0. **The mechanism is built but has never been tested at tiny1m3m** — no `177-talking-heads` idea file, no entry in `closed.md`.
- Closest in-repo analog: 152-attn-logit-bias closed null (per-head additive bias on QK^T). 155-per-head-temp closed null (per-head multiplicative temperature on QK^T). 160-rms-gain-per-head closed null (per-head multiplicative gain on AV output). 166-t5-rpe closed null (per-head additive bucketed bias). All four are **per-head scalar/bias** levers — 177 is a **per-head × per-head cross-mix** lever, qualitatively different (cross-head, not per-head).

## Mechanism
Standard attention (per batch, position):
1. `scores[b, h, t, s] = Q[b, h, t, :] · K[b, h, s, :] / √d_k` — shape `[B, H, T, T]`.
2. `attn_w[b, h, t, s] = softmax(scores[b, h, t, :], dim=-1)` — same shape.
3. `out[b, h, t, d] = attn_w[b, h, t, :] @ V[b, h, :, d]` — shape `[B, H, T, d_k]`.
4. concat heads → `[B, T, d_model]` → `W_O`.

With talking heads (pre-softmax only — the lever is per the paper):
- After step 1, mix across heads:
  `scores[b, h_new, t, s] = Σ_h M[h_new, h] · scores[b, h, t, s]` — `[H, H]` mix.
- Then continue with softmax (step 2) using the mixed scores.

With talking heads out (post-softmax):
- After step 3, mix across heads:
  `out[b, h_new, t, d] = Σ_h M_out[h_new, h] · out[b, h, t, d]`.
- Then concat → W_O.

**Step-0 bit-identical**: both M and M_out are initialized to identity. `M @ x = x` for any `x` ⇒ scores, softmax, attn_w, out unchanged ⇒ **byte-identical to baseline at step 0 (max-abs-diff = 0.0)**.

The two levers (talking-heads-Q pre-softmax, talking-heads-out post-softmax) are independent. Each can be enabled separately, both can be enabled together (paper's full form), or one at a time.

## Design sketch
- **Files**:
  - `models/layers.py` — the mechanism is **already implemented** at
    lines 1796-1800 (init pre-softmax M=I) and 1929-1933 (init
    post-softmax M=I), and applied at lines 3028-3033 (pre-softmax mix)
    and 3042-3045 (post-softmax mix). The implementation work is
    purely the **config wiring**:
    - `configs/llm_config.py` — add `use_talking_heads_q: bool = False`
      and `use_talking_heads_out: bool = False` on `LLMConfig` (default
      off) and a `Tiny1M3MTalkingHeadsConfig` subclass with both True.
    - Verify that both flags are read from config and threaded into
      the four `TransformerBlock(...)` / `MultiHeadAttention(...)`
      construction sites in `models/llm.py` (the sites that already
      thread `use_talking_heads_q` per `models/layers.py:3977`).
- **Config flags**:
  - `use_talking_heads_q: bool = False` (default off, pre-softmax mix).
  - `use_talking_heads_out: bool = False` (default off, post-softmax mix).
  - **Treatment**: both True (paper's full form). **Ablations**: Q-only
    and Out-only if first run shows signal.
- **Step-0 byte-identical**: with `M = torch.eye(H)` and
  `M_out = torch.eye(H)` (the existing init at lines 1799 and 1932),
  `scores @ M = scores` and `out @ M_out = out` exactly ⇒ output
  unchanged ⇒ **byte-identical to baseline at step 0 (max-abs-diff = 0.0,
  no tolerance needed — the operation is a literal identity matrix
  multiplication)**.
- **Intuition (why it might lower val loss)**: at 0.94M/12L/4H, each
  attention head has only 16 dims and operates on a relatively small
  effective subspace. Cross-head mixing lets heads *share* information
  pre-softmax (so head 0 can borrow head 1's logit signal) and
  post-softmax (so head 0's value output can borrow head 1's). This
  is qualitatively different from per-head scalar levers (152, 155,
  160, 166, all closed null): it lets heads *interact*, not just be
  tuned independently. The closed nulls all operated per-head; the
  cross-head mix is a strictly richer axis (H² params vs H params,
  and the H×H matrix is rank-deficient at init — the optimizer has
  room to develop a richer cross-head communication pattern).
- **LoC**: ~10 lines of config wiring (mechanism is already in
  `models/layers.py`). Smallest implementation cost after 172 and
  175.

## Scale evidence
- Talking-Heads Attention validated on Transformer-Big WMT'14 (~220M
  params), +1.0 BLEU En-De. **Direct validation at ≥100M.**
- In-repo at 0.94M: four per-head-attention-shape levers have closed
  null (152, 155, 160, 166). 177 is the *cross-head* analog — a
  structurally different axis from any of the four.
- **Transfer risk: med** (validated at ≥100M in translation; not
  directly validated at GPT-style causal LM at ≥100M. The mechanism
  is scale-free so the bet is plausible.)

## Why it's worth a slot
The bet: cross-head mixing is a *richer* lever than the four closed
per-head scalar levers because the H×H matrix allows heads to *share
information*, not just be tuned independently. At 0.94M/12L/4H the
cross-head space is small (4×4 = 16 params per layer) but the rank-
deficient init gives the optimizer room to develop a useful mixing
pattern. We expect Δval ∈ [-0.005, -0.020] (modest, similar to the
per-head-shape family). A null tells us cross-head mixing is also
absorbed by Q/K gradient updates at this tier (the same closure
pattern as 152, 155, 160, 166) and per-head-attention-shape axes are
exhausted at 0.94M. A win unlocks the cross-head mixing family at
Phase-2 ≥135M where each head has more gradient signal to develop a
useful cross-head communication pattern. Smallest implementation cost
of the five filed levers (~10 LoC of config wiring only — the
mechanism is built in `models/layers.py` already).