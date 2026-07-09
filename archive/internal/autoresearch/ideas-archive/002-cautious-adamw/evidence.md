# Evidence — 002 cautious-adamw

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast-34386 (220.82.52.202:34386, RTX 3060, sm_86)
- control val: 6.4403
- treatment A val: 6.4406  (embedding bucket: `token_embedding` + `emb_proj`)  Δ: +0.0003
- treatment B val: 6.4337  (gain bucket: `*.norm.weight` + 1D scalars)            Δ: -0.0066
- pass/fail bar (tiny1m3m): both A (Δ +0.0003) and B (Δ −0.0066) are well inside
  run-to-run variance (~0.04 on this box) → no effect
- → NULL. tiny1m3m is the only tier; there is no larger-tier re-test.
- box check: ctrl 6.4403 vs leaderboard 6.4287 (+0.0116)
- raw: remote-results/2026-06-09-vast-tiny1m3m/results.json
- date: 2026-06-09

The 002 wiring (`use_cautious_adamw` flag + `CautiousAdamW` subclass) is in place
and bit-identical when `"none"` (default); `boxval` smoke (max diff 2.98e-08)
confirms the gate. Closed as a tiny1m3m null.
