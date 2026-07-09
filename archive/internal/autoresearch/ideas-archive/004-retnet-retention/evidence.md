# Evidence — 004 retnet-retention

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast 220.82.52.202:34386 (RTX 3060, sm_86)
- treatment val: 6.4162 (r1) — n=1
- control bracket: ctrl=6.3875, ctrl2=6.4050 (gap 0.0175)
- Δ vs ctrl: +0.0287 (treatment is *worse* than ctrl)
- Δ vs ctrl2: +0.0112 (treatment is *worse* than ctrl2)
- pass/fail bar (idea.md screen20m legacy 4.5864 / n/a tiny1m3m): n/a — v1 ships
  the kernel + probe, not the production attention rewrite. The arq-r1 run
  exercised the **probe** path (no retention wired into `MultiHeadAttention`),
  so this measurement is a sanity check that the kernel is bit-stable, not
  a real A/B against the retention attention. The v2 wiring PR will do the
  real A/B.
- two-ctrl rule: treatment > both ctrls → NULL (worse than both). Plan
  `pass: tiny1m3m val ≤ 6.4237` is *not* met in the WIN sense (treatment
  is higher than both ctrls).
- box check: ctrl 6.3875 vs leaderboard 6.4287 = -0.0413 (within 0.04 noise band)
- raw: remote-results/2026-06-09-vast-tiny1m3m/arq-r1/{004-retnet-retention.log,ctrl.log,ctrl2.log}
- date: 2026-06-09

v1 ships a working kernel (kernel + 4 invariants in pytest). The probe ran
without NaN/Inf and produced a stable val_loss. v2 (the real A/B) is a
separate PR — filed for the next pipeline cycle.
