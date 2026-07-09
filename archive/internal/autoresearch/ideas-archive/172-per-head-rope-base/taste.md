## r1 — 2026-06-14 — verdict: accept

- **Zero implementation risk.** Mechanism is already built and wired in
  `models/layers.py:1753-1754` (init), `:2031-2034` (use), `:2828/:2851` (forward
  graph), `:3833` (constructor threading). The only work is ~10 LoC of config
  wiring (`use_per_head_rope_base` flag on `LLMConfig` + a `Tiny1M3MPerHeadRopeConfig`
  subclass). Step-0 byte-identical via `exp(0)=1.0` init ⇒ no drift risk.

- **Sharp bet, principled extension of a closed axis.** The "RoPE base sweep —
  500k winner" is in `closed.md:22` at our target tier, which means the axis has
  signal at tiny1m3m. 172 takes the next step: 500k is a *compromise* across
  heads, and per-head learning should specialize each head's frequency band.
  This is the right follow-up — it closes the per-head-RoPE axis that the
  closed sweep didn't probe. Hypothesis sentence is crisp:
  "global 500k base is a compromise; per-head learning finds better
  specialization." Expected Δval ≈ -0.003 to -0.012.

- **Both outcomes are informative.** Win → per-head RoPE axis is real, carry to
  135M where each head has more gradient signal. Null → global base is
  sufficient at this scale; per-head frequency specialization doesn't bind at
  0.94M. Either closes the axis cleanly.

- **Transfer-risk: med, correctly tagged.** Per-head RoPE is a principled
  extension (every LM already has per-head Q/K/V projections) but not directly
  validated at ≥100M. Closest analog `partial_rotary_p` is validated at 1B+;
  172 extends it on the *frequency* axis. Tagged correctly — accept.

- **Family-fit flag (not a `revise`).** Recent attention-positioning cluster
  has been busy: 154-rebased-attn (WIN, record break), 155-per-head-temp
  (null), 161-dyt-temp (null, per-layer τ_l fought canonical attention scale
  prior). 172 is in this family but on a different axis (frequency band, not
  temperature or rebasing) and anchors on a different known-live axis (the
  500k base sweep). Different enough — don't reject on crowding.

- **Small leverage is the real concern.** 48 scalars (+0.005%) — gradient
  signal per head may be weak at 0.94M. The expected Δval sits right at the
  ±0.04 noise band edge. But: clean null is still informative (above), and
  the implementation is essentially free, so the cost-of-test is low enough
  to take the bet.

Verdict: **accept**. Reset `round` to 1 for the definition gate's own budget.
