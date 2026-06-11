# Review log — 054 PolyCom

## r1 — 2026-06-11 — verdict: revise
- **No falsifiable bar.** `## Why it's worth a slot` reads "if a higher-order
  activation helps... if it fails..." — that's a vibe, not a pass/fail. Add a
  `## Pass/fail bar` section pinned to tiny1m3m noise (~±0.01 val loss per
  PIPELINE.md): treatment must beat the control by ≥0.01 val loss to WIN;
  |Δ|<0.005 = inconclusive null; ≥+0.005 = LOSS. Cite the bracket-control
  convention already used by 015/016 evidence (two ctrls, must clear ctrl-gap).
- **Pin the variant.** Idea hedges "PolyReLU/PolyNorm" — these are mechanically
  different: PolyReLU is `Σ aᵢ·ReLU(x)^i` (3 scalar coefs, no norm); PolyNorm
  interleaves per-power normalization (≈ RMSNorm before each power). Pick ONE
  for the A/B and name it. Recommend PolyNorm (per Zhuo et al. §3.2, PolyNorm
  is the headline result; PolyReLU is the ablation). State init exactly:
  e.g. PolyNorm with `a₀=0, a₁=1, a₂=0, a₃=0` so step-0 reduces to plain
  `RMSNorm(up_proj(x))` (or whichever base activation the paper uses) →
  identity-init standard 020/023/024 use.
- **Name the control variant.** `Tiny1M3MConfig` inherits
  `ffn_variant: str = "squared_relu"` (`configs/llm_config.py:582`). The paper
  compared PolyNorm against GELU/SwiGLU/ReLU — NOT squared_relu. Decide:
  (a) A/B vs `squared_relu` (our existing tiny baseline) — apples-to-apples
  with the leaderboard, but the paper's gain story isn't measured against
  Primer-style squared_relu, so a null is harder to interpret; or
  (b) A/B vs `gelu` (paper-matched base; we have `GELUFeedForward` in
  `models/components.py:55`) — clean replication of the paper's reported
  delta. Spec the control config class name (e.g. `Tiny1M3MGELUConfig` as
  ctrl, `Tiny1M3MPolyComConfig` as treatment). Recommend (b) — replication
  comes first, then a second round vs squared_relu if it wins.
- **Address numerical stability up front.** Polynomial powers ≥2 amplify
  outliers and have repeatedly NaN'd this repo in bf16 (cf. 020 forgetting-attn
  and 022 softpick NaN episodes already in `_arq_022.py` traces). Add a
  `## Numerical stability` section that pins: (i) per-power RMSNorm (the
  PolyNorm choice) is mandatory, not "optional"; (ii) higher-power terms
  computed in fp32 then cast back, or whole FFN forced to fp32 for the A/B;
  (iii) initial coef magnitudes (paper §3 init values — quote them, don't
  invent); (iv) gradient-clip stays at the trainer default. The taste log
  flagged this exact concern as definition-gate work — discharge it here.
- **Param-count parity.** PolyNorm adds 3 scalar coefs + per-power RMSNorm
  weights per FFN layer. At d_ff=256, that's tiny but non-zero (≈3·256 = 768
  params per layer · 12 layers ≈ 9k extra). The current tiny1m3m baseline is
  0.94M params — 1% drift is borderline. State whether the comparison is
  param-matched (e.g. shrink `d_ff` by ~1 to absorb) or unmatched-with-note.
  Recommend unmatched-with-note (overhead is <1%, well inside leaderboard
  reporting noise) but say so explicitly.
- **Scale-evidence citation tightening.** `## Scale evidence` says "lower
  training loss, lower validation PPL, and better downstream performance" —
  give the headline number (e.g. "PolyNorm-LLaMA 1B: -0.04 val PPL vs SwiGLU
  per Zhuo §4, Table 2"). Reviser must check the actual number against the
  paper before transcribing — don't fabricate. transfer-risk: low is fine
  IFF the headline number survives the lookup; if the 1B win is <0.02 PPL,
  this is closer to noise-at-scale and the tag should bump to med.

## r2 — 2026-06-11 — verdict: approve
- The exact paper number is now pinned in scale evidence.
- The variant, control, numerical stability, and param-count choices are explicit.
- This is ready to move into planning.
