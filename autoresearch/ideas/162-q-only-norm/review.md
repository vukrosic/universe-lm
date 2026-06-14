# Review log — 162 q-only-norm

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