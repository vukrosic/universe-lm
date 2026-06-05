# Peer review — Query / W_Q micro-tweaks

Reviewer stance: hostile but fair. Grading against this repo's own rules: (1) must be a real *mechanism*, not HP/init tuning; (2) must plausibly transfer 25M→135M; (3) must not duplicate an already-implemented lever. Most of these are per-head scalar tweaks — the prior here should be that small per-head scalars **wash at scale**, so the bar is high.

## Verdict table

| Idea | Verdict | Killer issue |
|---|---|---|
| Per-head Q bias | **Revise** | Overlaps `use_attn_sink` (#99); placement vs RoPE/norm is load-bearing and unspecified |
| Per-channel Q-gain | **Kill / Revise** | Pre-RoPE per-channel scale **is** the `q_norm` affine that already exists; only post-RoPE is novel, and that breaks RoPE pair structure |
| Learnable softmax temp | **Kill** | Mathematically identical to scalar `q_gain` (#37), already implemented |
| Identity-init Q-proj | **Revise** | Init-only change — brushes the excluded "init std" rule; init tricks usually wash at scale |
| Low-rank Q residual | **Kill** | W_Q is already full-rank square; a low-rank residual on a trainable full-rank matrix adds zero expressivity |
| Talking-heads on Q | **Keep** | Mislabeled (real talking-heads mixes *logits*, not Q); still the most defensible novel idea |
| Per-head RoPE-freq scale (Q) | **Kill** | Q-only freq scale breaks RoPE's relative-position identity; scaling both Q+K = `rope_base` (#63) |
| Q centering | **Revise** | The stated rationale is self-contradictory (see below) |

Net: of 8, **1 clean keep, 3 revise, 4 kill/redundant.**

## Per-idea detail

### 1. Per-head Q bias — Revise
- **Redundancy:** a constant `b_h` added to Q adds a key-dependent term `b_h·k_j` to every score → a *content-based, token-independent attention prior*. That is precisely the job of the attention-sink slot already implemented as `use_attn_sink` (#99). Need to argue why a learned Q-bias beats a sink, or A/B them head-to-head.
- **Placement is everything and unspecified:** if added before `q_norm`, RMSNorm largely erases it. If before RoPE, it gets rotated (no longer a constant prior). Only "after norm **and** after RoPE" gives the clean constant-bias semantics — state this.
- **Param cost mislabeled "the bias RMSNorm stacks lack":** the residual-stream norm lacking a bias is a different object from a Q bias. Drop that framing.

### 2. Per-channel Q-gain — Kill or Revise
- `q_norm` is an RMSNorm **with a learnable per-channel gain**. A per-channel multiplicative gain on Q *before RoPE* is therefore already in the model. Not novel.
- *After* RoPE it is novel but suspect: RoPE rotates dimension **pairs**; an independent per-channel scale breaks the within-pair norm and destroys the rotation's relative-position property. Likely neutral-to-harmful.
- If kept, the only honest version is "post-RoPE per-*pair* scale" — and justify why that beats the existing scalar `q_gain`.

### 3. Learnable softmax temperature — Kill
- `q_gain` (#37) multiplies Q by `(1+g_h)` per head, which scales the logits by `(1+g_h)` → **exactly** a per-head softmax temperature. This is the same lever under a different name. No experiment to run.

### 4. Identity-init Q-proj (`W_Q = I + Δ`) — Revise
- This is an **initialization scheme**, not a mechanism. The repo's exclusion list rules out "init std" tuning; identity-init is adjacent and needs a justification for why it's structural rather than HP.
- Interacts with **Muon**: Muon orthogonalizes 2D weight updates; an `I + Δ` parametrization changes the operating point Muon sees. Confirm it doesn't just re-discover the standard init after a few steps.
- Transfer prior is weak: init tricks routinely wash by the Full ladder. Mark Conf **low**, not med.

### 5. Low-rank Q residual (`Q += (xA)B`) — Kill
- W_Q is `d_model×d_model`, trained from scratch, **already full rank**. Adding a low-rank residual term to a trainable full-rank matrix is a pure reparametrization with no expressivity gain at convergence (unlike LoRA, where the base is *frozen*). 
- The stated motivation — "tests if the query proj is rank-limited" — is false: it isn't rank-limited. The only possible effect is optimization dynamics, which is an HP-flavored claim. Cut it.

### 6. Talking-heads on Q — Keep (with corrections)
- Genuinely distinct: cross-head information flow, which nothing in the repo does.
- **Mislabeled:** Shazeer's Talking-Heads Attention mixes the *attention logits/probabilities* across heads, not the Q vectors. Mixing Q vectors pre-dot is a weaker, different operator. Either rename ("cross-head Q mixing") or implement the real logit-mixing version.
- GQA caveat: with `n_kv_heads=2`, multiple Q heads already share K/V; mixing their Q's may be partly redundant. Worth stating.
- Transfer: talking-heads gains are known to shrink as head count grows; still small at 135M (n_heads=9). Reasonable bet, Conf **med** is fair.

### 7. Per-head RoPE-freq scale on Q only — Kill
- RoPE encodes *relative* position because Q and K rotate at the **same** frequencies, so `q_m·k_n` depends on `m−n`. Scaling Q's frequencies independently of K **breaks** this identity — the dot product no longer cleanly encodes relative position. Likely harmful, not a knob.
- Scaling Q **and** K together is just changing the rotary base → already covered by `rope_base` (#63) and its sweep configs. Redundant either way.

### 8. Q centering — Revise (rationale is wrong)
- Stated why: "removes common-mode Q (cancels in softmax anyway)." This is self-contradictory. Softmax is shift-invariant only to a constant added **across keys**. Subtracting Q's mean changes each logit by `mean(Q)·k_j`, which **varies per key** and therefore does **not** cancel. So either it's a real change (and the rationale is wrong) or it's a no-op (and there's nothing to test). Rewrite the hypothesis before running.

## Cross-cutting issues

1. **Redundancy with shipped levers** is the dominant problem: #2 (vs `q_norm` affine), #3 (vs `q_gain`), #7 (vs `rope_base`), #1 (vs `attn_sink`). Half the table re-tests known mechanisms. Audit against `LLMConfig` flags before promoting any.
2. **Q-only is an arbitrary cut.** Several of these only make sense symmetrically (K bias, K freq). Isolating Q risks measuring noise. Decide whether the research question is "the Q side specifically" or "attention scaling in general."
3. **No stated eval protocol.** Screen-tier results explicitly *do not transfer-promote* (per the config docstrings). The plan needs: baseline number + seed count, screen→Full promotion gate, and a transfer check at ≥25M. Without it these are vibes.
4. **Optimizer interaction unspecified.** 2D params (talking-heads `M`, low-rank `A/B`) route to Muon; per-head scalars/bias route to AdamW. Confirm shapes and that zero/identity-init params actually receive gradient under Muon's orthogonalization.
5. **Confidence column is optimistic.** Per-head scalar tweaks on a small model are the classic "wins the screen, washes at Full" trap. Down-weight #3, #4, #7.

## Recommended triage (Round 1)
- **Run first (real + cheap):** Talking-heads on Q (fixed to logit-mixing), Per-head Q bias (post-norm-post-RoPE, A/B'd against `attn_sink`).
- **Kill outright:** Learnable softmax temp, Low-rank Q residual, Per-head RoPE-freq (Q-only).
- **Fix the writeup before running:** Per-channel Q-gain (post-RoPE-pair only), Q centering (rewrite hypothesis), Identity-init (reclassify as init study, low conf).

---

# Peer review — Rounds 2 & 3

Same stance. New recurring failure modes to watch: (a) **RoPE-pair breakage** from per-channel post-RoPE ops; (b) **absorbable** reparametrizations that add no expressivity; (c) ideas that **kill the SDPA/flash-attn fast path** (a real cost on the screen); (d) ideas that **inflate or shrink the param budget** without reallocation, making them model-size tests not mechanism tests; (e) **loss-regularizers with a coefficient knob**, which brush the repo's HP-tuning exclusion.

## Verdict table

| # | Idea | Verdict | Killer issue |
|---|---|---|---|
| 6 | ALiBi-style distance bias | **Keep (temper)** | Its headline win is length *extrapolation* — moot at fixed train=test seq 2048; only the recency prior counts. Decide replace-vs-stack with RoPE (stacking double-encodes position). |
| 7 | Token-conditioned Q temperature | **Keep (low)** | Fights `q_norm` (which already fixes Q magnitude); dynamic temps often collapse to a constant (`w→0`). |
| 8 | Query expansion (2 q/head) | **Revise** | Mean of two softmax reads ≈ one softer read; not clearly more expressive than +heads, and it doubles attention FLOPs. |
| 9 | Norm-conditioned Q gate | **Kill** | Post-norm `‖x‖≈√d` is ~constant → near no-op; pre-norm `‖x‖` confounds with depth-growing residual energy. |
| 10 | Antisymmetric Q–K coupling | **Kill/defer** | Directional/ordered relations are already carried by RoPE's relative encoding; likely redundant. |
| 11 | Per-head learnable RoPE base | **Keep (low-med)** | Clean, but a per-head scalar — classic "wins screen, washes at Full." |
| 12 | Partial rotary | **Keep** | Borderline a 1-knob sweep (fraction p); pick p=0.5, test vs full — don't sweep p as tuning. |
| 13 | Decoupled content/position | **Keep (flagship, param-match)** | Adds projections → must param-match the baseline or the win is just more params. |
| 14 | CoPE | **Defer** | High lift; documented gains are counting/long-context — uncertain transfer to plain LM loss at 2048, and the sequential gate **kills flash-attn**. |
| 15 | Per-channel relevance (bilinear) | **Revise/low** | If applied pre-norm it's **absorbable into W_Q** (no gain); post-RoPE it breaks rotation pairs (same flaw as R1 #4). |
| 16 | Cosine attention | **Revise/low-med** | `q_norm`/`k_norm` (RMSNorm) already nearly do this; novelty is only "scalar temp + exact unit-norm vs per-channel affine." Near-redundant. |
| 17 | Additive (Bahdanau) term | **Kill/defer** | Materializes the T×T map → kills SDPA; exotic with weak prior. |
| 18 | QK shared low-rank factor | **Revise** | "Saves params" is only a fair test if the saved budget is **reallocated**; otherwise it's just a smaller model. |
| 19 | Register / memory tokens | **Keep** | Plausible global memory, but partly overlaps `attn_sink`'s escape valve; A/B against it. |
| 20 | Query subspace bottleneck | **Kill/low** | d_k is already tiny (~24); rank-limiting Q below that is aggressive and likely just removes capacity. |
| 21 | Mixture-of-queries | **Kill/defer** | MoE trades params for FLOPs — wrong frame for a fixed-param-budget lab; routing instability + load-balance aux loss. |
| 22 | FFN-gated Q temperature | **Gate on #7** | Strictly heavier #7; if #7 washes, this does too. Only run if #7 wins. |
| 23 | Head-orthogonality penalty | **Defer (low)** | A loss-shaping regularizer, not an architecture mechanism; weak transfer. |
| 24 | Query dropout | **Kill** | Repo trains at dropout=0; on a small undertrained screen, dropout usually hurts. |
| 25 | Logit-entropy reg | **Kill** | Regularizer with a coefficient = the HP-tuning the rules exclude. |
| 26 | Learned per-key Q offset | **Kill (dup)** | This *is* Round-1 #1 (per-head Q bias). Merge, don't double-count. |

Net Rounds 2–3: of 21, roughly **5 keep, 4 revise, 12 kill/defer.**

## Cross-cutting (Rounds 2 & 3)

1. **Budget frame.** #18, #20, #21 change param count. Either param-match or reallocate explicitly — otherwise you're measuring size, not mechanism (the same trap flagged for embedding-factorization in `RESEARCH_IDEAS.md`).
2. **Flash-attn cost.** #14, #17 (and partly #13) materialize scores or serialize the position computation, dropping the SDPA fast path. On the screen tier that's a real wall-clock hit — budget for it or descope.
3. **Redundancy with `q_norm`/RoPE persists.** #15 and #16 re-approach what RMSNorm-on-QK + RoPE already do. Before coding, write the exact math showing the *delta* vs the shipped path.
4. **Positional benefit at fixed length.** #6, #11, #14 lean on positional reach/extrapolation. The repo is dense at a fixed 2048 (train==test), so length-extrapolation upside doesn't score — only the inductive-prior part does. Re-state each "why" accordingly.
5. **Regularizer creep.** #23–#25 are loss terms with coefficients. They live closer to the excluded HP-tuning bucket than to "new mechanism." Keep them out of the main ladder.

## Recommended triage (Rounds 2 & 3)
- **Run first (real + cheap):** ALiBi bias (#6, as a RoPE *replacement* probe), per-channel relevance (#15, **post-norm pre-RoPE** placement only), cosine attention (#16, vs `q_norm` baseline), register tokens (#19, vs `attn_sink`).
- **Flagship bet (budget for the lift):** decoupled content/position (#13), param-matched.
- **Defer (high lift / uncertain transfer):** CoPE (#14), mixture-of-queries (#21), additive attention (#17).
- **Kill outright:** norm-conditioned gate (#9), query subspace bottleneck (#20), query dropout (#24), logit-entropy reg (#25), learned per-key offset (#26, duplicate of R1 #1).
