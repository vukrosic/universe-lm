# Review log — 165 k-only-norm

## r1 — 2026-06-14 — verdict: approve
- **Lever is the clean K-mirror of 162-q-only-norm (approved).** Read
  idea.md, taste.md (accept), and 162's review.md r1/r2 for the cleared
  precedent. 165 normalizes K only (leaves Q untouched) pre-softmax
  post-RoPE; 162 normalizes Q only (leaves K untouched); 016 is the
  symmetric QK-norm WIN. The 3-way axis test (Q-only / K-only / QK) is
  the plan's stated bet and is well-framed. Mechanism, scope, and
  distinctness all check out the same way 162's reviewer cleared them.
- **Distinct from every closed lever — verified vs closed.md.**
  - 016-qk-norm — symmetric QK RMSNorm pre-softmax (WIN); 165 is the
    K-only asymmetric ablation, distinct by construction.
  - 152-attn-logit-bias — per-head additive bias on QK^T *post* matmul,
    pre-softmax (closed null at 0.94M); 165 normalizes K *before* the
    matmul, not after.
  - 155-per-head-temp — per-head learnable τ on logits (closed null);
    165 is per-token K-side rescaling, not per-head temperature.
  - 160-rms-gain-per-head — per-head scalar on attention *output*
    post-AV (closed null); 165 fires pre-softmax pre-AV on K only.
  - 159-emb-layernorm — pre-block LN on embeddings (closed DRIFT);
    165 is mid-block K-side RMSNorm, pre-matmul.
  - 161-dyt-temp — per-layer learnable scalar τ_l (closed null); 165
    is RMSNorm, not a temperature schedule.
  - 162-q-only-norm (in queue at needs-run, review.md r1/r2 approved)
    is the orthogonal partner, not a duplicate — 162 normalizes Q,
    165 normalizes K; the two together with 016 are the 3-way axis
    test the plan describes.
  - Not in `_closed/`. Not in the closed-axes section of closed.md.
- **Source citations are the same soft set as 162 — non-blocking.**
  Cohere Command-R / R+ (2024) and "Henry et al. arXiv:2002.12928" are
  the same soft attributions 162's r1 flagged; per that review the
  lever's transfer story rests on the broader RMSNorm family
  (LLaMA-3 / Qwen-2.5 / Mistral at 1B-70B+), not on the asymmetric QK
  cites specifically. The arXiv ID 2002.12928 is unverifiable on its
  face (no widely-known paper by that exact ID matches the cited title
  "QKNorm: Mitigating Transformer Attention Sink"; the canonical
  symmetric QKNorm is Dehghani et al. 2023, arXiv:2302.05442). Plan
  should tighten the citation set to one verifiable primary source
  when the implementer writes plan.md — same finding 162 had, and
  162's plan did not address it. Non-blocking: the lever survives the
  citation looseness, the same way 162 did.
- **Mechanism is mechanism, not hyperparameter.** K-only RMSNorm is a
  structural change (where the norm fires + whether Q is co-normalized),
  not an LR/schedule/init-constant lever. Standard RMSNorm init
  (weight=1, bias=0), no tunable. Real arch lever.
- **🔴 tiny1m3m scope OK.** Explicit; no reference to screen20m or any
  larger tier. Single-tier scope confirmed. Seed 42 only — no
  multi-seed protocols.
- **Identity-init tolerance acknowledged, same trade-off as 162/159.**
  `nn.RMSNorm(d_k, eps=1e-6)` weight=1, bias=0 ⇒ at step 0 the lever
  rescales K to unit RMS per head-dim per token. Spec calls out the
  fp32 max-abs-diff < 1e-3 tolerance (same precedent as 159-emb-
  layernorm, 162-q-only-norm). Acceptable; not a blocker. Plan may
  additionally multiply by `sqrt(mean(k²))` post-norm for strict
  byte-identity, matching 162's optional knob — code-gate call.
- **Implementability verified against current line numbers.** The
  plan's line citations are mostly accurate vs the working tree (which
  has 162's hunks already landed):
  - configs/llm_config.py:577 — `use_q_only_norm: bool = False` is
    already present; `use_k_only_norm: bool = False` slots in next to
    it (sibling of the existing 162 field).
  - configs/llm_config.py:5315-5319 — `Tiny1M3MQOnlyNormConfig` (the
    `@dataclass`-decorated subclass, NOT a bare `class C(...):`
    annotation, per the 162/159/155/161 precedent that bare-class
    annotation breaks dataclass field inheritance) is the sibling
    site for `Tiny1M3MKOnlyNormConfig`.
  - models/layers.py:853 — `use_q_only_norm: bool = False` kwarg is
    already present on `MultiHeadAttention.__init__`; 165's kwarg
    slots in next to it.
  - models/layers.py:1039-1041 — `self.q_only_norm = nn.RMSNorm(...)`
    is already registered here when `use_q_only_norm` is on; 165's
    `self.k_only_norm = nn.RMSNorm(self.d_k, eps=1e-6)` slots in
    alongside.
  - models/layers.py:2122-2144 — three forward branches (nope/cope,
    use_qk_norm_post_rope, default pre-RoPE) already have the
    `if self.use_q_only_norm:` short-circuits around the symmetric
    `q_norm/k_norm` path; 165's `if self.use_k_only_norm:` arm goes
    in the symmetric position (Q untouched, K-only normed), mutually
    exclusive with 162 (assert at the top of forward, matching the
    `use_cope ∧ use_qk_norm_post_rope` pattern at :1916-1917).
  - models/layers.py:2437-2450 — MoA `extra_K` branch already has
    use_q_only_norm gating; 165 threads through symmetrically.
  - models/layers.py:3080 — `TransformerBlock.__init__` has
    `use_q_only_norm: bool = False`; 165's kwarg slots in alongside.
  - models/layers.py:3490 — pass-through into MHA constructor;
    165's parameter slots in.
  - models/llm.py:440 — `self.use_q_only_norm = getattr(...)`; 165
    reads alongside. :685 and :941 — pass-throughs into both
    `TransformerBlock(...)` constructors; 165's parameter slots in.
  Total lever LoC in layers.py ≈ 8-10 (matches 162's footprint);
  configs/llm_config.py ≈ 4 LoC for the field + the dataclass
  subclass; llm.py ≈ 4 LoC (3 reads + 2 pass-throughs); plus the
  treatment stub `_arq_165-k-only-norm.py` (~30 LoC). Well under 200
  LoC.
- **Falsifiable 3-way bar, tightened per 162's r1 lesson.** PASS:
  treatment val ≤ 016-qk-norm's recorded val by ≥ 0.005 (same shape
  as 016's own bar — match-or-beat the symmetric WIN). NULL: |Δ vs
  bare no-norm ctrl| < 0.005 (isolates the K-side / symmetry null
  hypothesis). DRIFT: ≥ ctrl + 0.005. The plan correctly avoids the
  "taste's ~half-of-016's-gain (-0.007)" framing that sits inside the
  ±0.04 noise band — same bar shape 162's r1 finding asked for, and
  162's plan delivered. Acceptable; tight enough to call WIN vs NULL
  at tiny1m3m.
- **Transfer-risk: low justified.** ## Scale evidence section cites
  LLaMA-3 / Qwen-2.5 / Mistral (1B-70B+, RMSNorm family) and Cohere
  Command-R (35B+, asymmetric QK). The K-only axis is a sub-claim but
  the normalization primitive is production-validated at scale. The
  `low` tag holds.
- **Source vs transfer-risk tag consistency:** the cited RMSNorm
  family papers all validate at ≥100M (LLaMA-3 smallest variant is
  ~1B), which is well above tiny1m3m's 0.94M. The Cohere Command-R
  cite is at 35B+, also well above. Tag=low is justified for the
  family; the K-only axis itself is a sub-claim but the leverage
  primitive carries the transfer story. Holds.
- **Forward flag from taste addressed in plan:** taste flagged that
  K-norm placement should be post-RoPE for consistency with 162's
  post-RoPE placement. Plan.md confirms post-RoPE placement (line
  in Mechanism section: "K-norm applies *post-RoPE* to be consistent
  with 162's post-RoPE placement"). ✓
- **Coordination note (non-blocking).** The taste artifact also notes
  this idea was racing the implement-button at 2026-06-14T06:21:01Z;
  the taste verdict reset status to needs-review with round=1, which
  is the state I'm reviewing. Working tree currently has 162's hunks
  landed (and 165's hunks not yet landed — code-impl gate's job).
  No conflict.

**Verdict**: approve → `needs-plan`. Reset `round` to 1 for the code
gate per protocol. Plan should (a) tighten the citation set when the
implementer writes plan.md (drop arXiv:2002.12928, prefer the
canonical symmetric QKNorm cite, or omit the unverifiable ID and
rely on the RMSNorm-family production validation), and (b) confirm
mutual-exclusion assertion between `use_k_only_norm` and
`use_q_only_norm` at the top of `MultiHeadAttention.forward`
(matching the existing `use_cope ∧ use_qk_norm_post_rope` assertion
pattern at models/layers.py:1916-1917). Neither is a definition-gate
blocker — both are code-gate polish.