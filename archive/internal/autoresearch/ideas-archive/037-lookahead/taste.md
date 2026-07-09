## r2 — 2026-06-11 — verdict: accept

- **Portfolio crowding — addressed.** r1 said the cluster (031-040) was
  optimizer-replacement-crowded; the r2 pitch is structurally different:
  Lookahead is the only *wrapper* in the cluster, leaves the inner
  Muon+AdamW(vocab) untouched. Distinct lever class, not a 5th momentum
  variant. The 018-ademamix distinction (slow-EMA inner modifier vs
  bounded per-snap pullback) is correctly drawn.
- **Crisp bet — addressed.** Failure mode named: late-stage AdamW
  trajectory on embedding/head oscillates around its running mean (LR
  decayed but per-step gradient noise on sparse vocab rows hasn't).
  015 Moonlight-Muon closed the *magnitude* axis of the inner step; 037
  would close the *temporal* axis (trajectory oscillation). Compositional
  hypothesis: 015 × 037 should be additive on the embedding/head slot.
  One-sentence bet in place: "Δ in [−0.01, −0.03] because per-snap pullback
  bounds late-stage oscillation that 015 leaves untouched."
- **Step-budget math — addressed, correctly.** r1's 018-ademamix worry
  (slow EMA half-life ~7k vs 92 steps, init-dominated) is *not* this
  mechanism: Lookahead is a per-snap pullback (φ ← φ + α(θ-φ)), not an
  asymptotic EMA. At k=5, α=0.5 over 92 steps = 18 bounded updates to
  φ, each touching 50% of (θ-φ) at that moment. φ is a running anchor
  with full trajectory memory, not init-weighted. The 18-snap count is
  enough to fire the mechanism inside this 92-step window. Defaults are
  defensible (k=5, α=0.5 = paper's "fast" setting).
- **Informative null — addressed.** Both win and null teach us something:
  WIN → 015×037 follow-up planned (composability hypothesis confirmed);
  NULL → wrapper class closed at this scale, 015 stays the place to
  spend optimization budget. The observable is logged
  (per-step val-loss std-dev in last 20 steps; pullback should reduce it
  on a WIN, leave it unchanged on a NULL).
- **Niche fit — solid.** Identity at α=0 (clean A/B), zero-init-able
  (φ = θ₀), ~30 LoC, tiny1m3m-runnable. Transfer-risk: med is fair and
  honestly conceded (paper has no 100M+ LLM-pretrain headline; the
  compositional-with-015 story is the strongest transfer argument).
- **Crisp observable.** Per-step val-loss Δ vs ctrl, plus std-dev
  comparison in the last 20 steps. WIN = |Δ| ≥ 0.01 with reduced std;
  NULL = |Δ| < 0.01, std unchanged. Plan bar: -0.01.

This is a sharp, high-leverage bet that closes the wrapper class
informatively at our scale. Move to needs-review.

## r1 — 2026-06-11 — verdict: revise

- **Portfolio crowding (dominant gap).** 9 optimizer-family ideas currently sit
  in `needs-taste`: 031 Adam-mini, 032 AdamS, 033 Sophia, 034 Adan, 036 LAMB,
  037 Lookahead, 038 SWAN, 039 APOLLO, 040 Adafactor. The taste rule is
  explicit: "the 5th optimizer-momentum variant in a row is a `revise`
  (diversify) even if each is individually fine." Re-pitch must either
  (a) make a sharp case for *this* lever over the rest of the cluster, or
  (b) drop it in favor of a non-optimizer family.
- **Vague bet.** The idea reads "slow-weight shadow *may* smooth bad curvature
  without changing the base optimizer" — that's a vibe, not a falsifiable
  prediction. Sharpen: name the specific failure mode of our current
  Muon+AdamW(vocab) trajectory at tiny1m3m that Lookahead would address
  (e.g., Muon ortho-step variance early in training, late-step Adam
  HP-thrash, …), and the observable in the run that would confirm/refute it.
  "We expect val-loss Δ X because Y" in one sentence.
- **Step-budget question (the 018-ademamix problem).** tiny1m3m is ~92
  optimizer steps. With Lookahead's standard k=5, α=0.5 you get only ~18
  slow-weight syncs across the whole run; the slow weights spend most of the
  trajectory still init-dominated. Show — with math, not vibes — that the
  variance-smoothing mechanism actually fires inside this 92-step window
  (or pick k/α defensible at this scale). Cf. `closed.md`:36 (018-ademamix
  taste-reject: "slow EMA half-life ~7k steps vs ~92 step run; lever only
  fires at ≥100k steps").
- **Informative-null framing.** Lookahead is a 2019 vanilla wrapper; "doesn't
  help at tiny1m3m" is *not* new information by itself. Frame the bet so
  the null is still worth logging — e.g., "Lookahead k=5 α=0.5 stacked on
  Moonlight-Muon should/should-not be additive because <X>", which would
  tell us something about the slot already won by 015. As written, both
  outcomes (small win or null) leave the queue no smarter.
- **Niche fit (passable, not strong).** Mechanism is identity-able
  (α=0 → no slow update) and tiny1m3m-runnable. Transfer-risk: med is fair
  — Lookahead's broader-task evidence is real but never carried an LLM
  pretrain headline the way Sophia/LAMB tried to. That's not enough on its
  own to clear the bar in a crowded portfolio.
