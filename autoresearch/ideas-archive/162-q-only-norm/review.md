# Review log — 162 q-only-norm

## r2 — 2026-06-14 — verdict: approve
- **No new content finding; r1's review still stands.** I read the full log
  (idea.md, plan.md, review.md r1, log.jsonl). The lever is unchanged — still
  asymmetric Q-only RMSNorm pre-softmax, K untouched, +192 params, 016's
  orthogonal ablation. Mechanism, scope, distinctness, LoC, transfer-risk,
  identity-init tolerance, and the loosened pass/fail bar (tightened in
  plan.md) all check out the same way r1 verified them. No new evidence to
  contradict; not re-litigating settled findings.
- **Current bounce-back is operational, not content.** `log.jsonl` shows the
  r2 cycle was: code-impl writes code (commit `41ca33e`) → daemon pre-queue
  smoke bounces because box's `git pull --ff-only` can't see un-pushed local
  commits → runner bounces `needs-run → needs-recode` → code-impl recodes
  with the same commit → runner bounces again → auto-implement agent gives
  up after 3 failed runs and punts `needs-recode → needs-review` (a deviation
  from the documented state machine, but the reviewer doesn't correct state
  machine drift — we judge the idea). The lever itself is sound; the blocker
  is **git push** which is human-only per repo convention. The reviewer is
  not on the push path and shouldn't invent a content finding to mask an
  operational one.
- **`plan.md` already addresses r1's pass/fail bar finding.** r1 noted
  taste's "~half of 016's gain ~-0.007" sits inside the ±0.04 noise band;
  plan.md now specifies the bar as a 3-way frame: PASS (match or beat 016's
  WIN by ≥0.005, same shape as 016's own bar), NULL (|Δ vs no-norm ctrl|
  < 0.005, isolating the K-side / symmetry null hypothesis), DRIFT (worse
  than ctrl by ≥0.005). That closes the r1 finding.
- **`Tiny1M3MQOnlyNormConfig` already wired and CPU-build smoke-passed.**
  Per plan.md: `MinimalLLM(Tiny1M3MConfig())` = 949,056 params (baseline) and
  `MinimalLLM(Tiny1M3MQOnlyNormConfig())` = 949,248 params (+192 = +0.02%,
  one `nn.RMSNorm(d_k=16)` weight × 12 blocks, no bias). The flag-off path
  is bit-identical to the existing 016 baseline (verified locally,
  max-abs-diff 0.0 on a 16-token forward at seed 42). The lever survives
  another code-gate review with no new hunks.
- **Not a duplicate of any closed lever.** Verified vs `closed.md`:
  - 016-qk-norm — symmetric QK RMSNorm pre-softmax (WIN); 162 is the
    Q-only orthogonal ablation, distinct by design.
  - 152-attn-logit-bias — per-head additive bias on QK^T *post* matmul,
    pre-softmax (closed null); 162 normalizes Q *before* the matmul, not
    after.
  - 155-per-head-temp — per-head learnable τ on logits (closed null);
    162 is per-token Q-side rescaling, not per-head temperature.
  - 160-rms-gain-per-head — per-head scalar on attention *output*
    post-AV (closed null); 162 fires pre-softmax pre-AV on Q only.
  - 159-emb-layernorm — pre-block LN on embeddings (closed DRIFT);
    162 is mid-block Q-side RMSNorm, pre-matmul.
  - 154-rebased-attn (WIN), 163-q-v-mix, 164-q-carry — these are the
    orthogonal axis partners per the closed-side test plan in `idea.md`.
    162 is the Q-side norm ablation; together they triangulate whether
    the binding axis is Q-side, K-side, or symmetric.
- **Transfer-risk: low justified.** RMSNorm family is production-validated
  at LLaMA-3 / Qwen-2.5 / Mistral (1B-70B+). The Cohere Command-R / Henry
  citations r1 flagged as "soft" are non-blocking — the lever's transfer
  story rests on the broader RMSNorm family, not on the asymmetric QK
  cites specifically. The `low` tag holds.
- **Verdict**: approve → `needs-plan`. Reset `round` to 1 for the code gate
  per protocol. The next attempt will still bounce on the un-pushed commit
  until a human runs `git push` — that is documented in plan.md and is not
  this reviewer's call. Approving here puts the lever back on the plan/run
  path; the queue stays warm.

## r1 — 2026-06-14 — verdict: approve
- **Pass/fail bar is loose, tighten in the plan (not blocking here).** taste's
  stated expectation is "WIN → ~half of 016's gain (~-0.007)". That sits inside
  the ±0.04 noise band, so the headline half-detection is *not* detectable at
  tiny1m3m. The plan should specify either (a) the existing two-ctrl WIN rule
  vs fresh no-norm ctrls with the bar Δ ≤ -0.005×2 cleared (matching 016's
  bar shape), or (b) frame the win as "match or beat 016's qk-norm A/B at
  tiny1m3m" with the explicit null hypothesis "Q-only ≡ no-norm at 0.94M,
  K-side was carrying 016". Plan-writer owns; not a definition-gate blocker.
- **Source citations are loose; do not reject.** "Cohere Command-R / R+ (2024)
  uses L2-normalized Q with raw K" — Cohere models do discuss QK
  normalization in their tech reports, but the L2-vs-RMS distinction is not
  pinned to a verifiable sentence in my read; "Henry et al. 2020 QKNorm" is
  also a soft attribution (the canonical symmetric QKNorm cite is Dehghani
  et al. 2023, ViT-22B, arXiv:2302.05442 — which is what 016 cites). The
  architectural lever itself (asymmetric QK pre-softmax RMSNorm) is
  well-validated by the broader RMSNorm family (LLaMA 3, Qwen 2.5, Mistral),
  which is what the transfer-risk:low tag actually rests on. Plan should
  tighten the citation set to one or two verifiable primary sources — the
  lever survives either way.
- **Distinct from closed axes — verified.** 152 (post-softmax per-head logit
  bias, mathematically null), 155 (per-head learnable τ, null at 0.94M), 160
  (per-head gain on attention *output*, post-AV) all live on the *output*
  side of the QK matmul; 016 is symmetric pre-soft QK RMSNorm. 162 is the
  only lever that norm-isolates Q pre-softmax with K untouched. Not a
  re-pitch; not in `_closed/`; not in the closed-axes section of closed.md.
- **tiny1m3m scope OK.** explicit; no reference to screen20m or larger.
- **Identity-init tolerance acknowledged.** `nn.RMSNorm(d_k, eps=1e-6)` init
  is weight=1, bias=0 ⇒ at step 0 the lever rescales Q to unit RMS per
  head-dim. Spec calls out the fp32 max-abs-diff < 1e-3 tolerance (same
  trade-off as 159-emb-layernorm). Acceptable; precedent set.
- **Implementability verified.** Diff in working tree already wires
  `use_q_only_norm` into `MultiHeadAttention.__init__` (models/layers.py:825,
  module registered at :1012-1013) and into the three forward branches
  (pre-RoPE at :2034-2036, post-RoPE at :2027-2029, nope/cope at :2021-2025),
  with `LLMConfig.use_q_only_norm: bool = False` at configs/llm_config.py:558
  and the flag read by `MinimalLLM.__init__` at models/llm.py:445 and passed
  into both block builders at :680 and :935. Total lever LoC in layers.py
  ≈ 8-10 (the spec's ~6 was a minimum). Wiring looks correct; code gate can
  do its own review.
- **Mechanism is mechanism.** RMSNorm on Q only is a structural change (where
  the norm fires + whether K is co-normalized), not a hyperparameter. Init
  is standard RMSNorm, not a tunable. Real arch lever.
- **Transfer-risk: low justified.** RMSNorm family production-validated at
  1B+ (LLaMA 3, Qwen 2.5, Mistral). The `low` tag holds even if the Cohere /
  Henry citations are soft.

**Verdict**: approve → `needs-plan`. Reset `round` to 1 for the code gate.
Plan must tighten the pass/fail bar (see first finding) and consider
pinning citation to verifiable primaries.