# Evidence — 164 q-carry

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, driver 580.159.03)
- baseline: cached mean=6.4346 ±0.0458 (box key 5b8a7fea8963, measured 2026-06-14T06:58:42Z, n=3 ctrls on commit 42ed363)
- treatment val: 6.4706   Δ vs baseline: +0.0360
- bpb: n/a (pending harness)
- pass/fail bar (from plan.md): WIN Δ≤-0.01; NULL |Δ|<0.01; DRIFT Δ>+0.01
  → Δ=+0.036 > +0.01 → plan DRIFT (loss, ~3.6× the plan null band); cache NULL (inside 6.4346±0.0458 band)
- box check: ctrl mean 6.4346 vs box-class leaderboard 6.4394±0.04 — within 0.005, no drift
- raw: remote-results/2026-06-14-vast-tiny1m3m/{164-q-carry_52674.log,run_06-54.json}
- date: 2026-06-14

## Transfer note
Q-side cross-block residual mixing (analogous to 021-value-residual which
carried V and WON at Δ=-0.034). Plan explicitly framed the question: is V
special, or does the cross-block residual-mixing axis generalize to Q?
Δ=+0.036 is *wrong-sign but inside the cache noise band*, so per §5 the
cache verdict is NULL. The plan's tighter bar (DRIFT >+0.01) classifies this
as a *plan-DRIFT null* — a meaningful hostile-direction signal that V is in
fact the binding axis. Closes the dual axis (V binds, Q doesn't) at 0.94M.
This is consistent with 150-xlayer-feedback (reject, r3 cap) — cross-block
attention-pathway levers don't bind at this tier. Transfer-risk: med; the
mechanism should be re-tested at ≥135M where the Q-learning path has more
gradient signal per token (~140× larger), but the wrong-sign direction at
3.6× the plan null band is a real null result at 0.94M, not a "tight null
that might win at scale."