# Taste log — 054 PolyCom

## r1 — 2026-06-11 — verdict: accept
- Unmined axis: FFN activation. Active queue (031-080) and `closed.md` have zero
  FFN-activation A/Bs — SwiGLU/GELU/ReLU swaps have never been tested in this
  pipeline. Slot is non-crowded (cf. the 10-deep 031-040 optimizer pile-up).
- Crisp bet, both directions informative. Win = FFN was under-expressing local
  function class at tiny1m3m; null = polynomial curvature is a large-model
  trick. Idea already names the null payoff explicitly.
- Niche-fit clean: mechanism (not HP), identity-init-able via zero coefficients
  on the extra polynomial terms (starts bit-equal to GELU baseline; meets the
  zero-init standard 020/023/024 use), and the swap sits inside the existing
  `GELUFeedForward` / `SwiGLUFeedForward` slot in `models/components.py` →
  small LoC budget. FFN is the dominant per-token nonlinearity at every scale,
  so the lever is observable at tiny1m3m.
- transfer-risk: low is supported — paper evidence is 1B dense LLM pretraining
  loss/PPL plus 1B-active MoE, i.e. direct LM transfer not sequence-length or
  long-context driven. Distinct from 053-reluformer (attention softmax swap,
  high-risk) — keep both in flight, they probe different layers.
- Numerical-stability caveat (poly degree > 1 can blow up — cf. our 020/022 NaN
  episodes) is a *definition-gate* concern (PolyNorm scaling, init magnitudes,
  fp16 paths) — not a taste-gate disqualifier.
- Routing: `flip.sh 054-polycom needs-review taste "accept: unmined FFN-act
  axis, identity-init poly composition, 1B+ LLM evidence, informative null" 1`.
