# Evidence — 172 per-head-rope-base

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52646
- baseline: cached mean=6.3975 ±0.04 (box 5b8a7fea8963, measured 2026-06-15T05:15:46Z, n=4)
- treatment val: 6.4084   Δ vs baseline: +0.0109 (wrong sign, inside band)
- bpb: n/a (pending harness)
- pass/fail bar: idea.md expects Δ -0.003 to -0.012 → not met (Δ +0.011 wrong-sign; inside |Δ|<0.04 band)
- box check: baseline mean 6.3975 vs cached prior 6.4447 = DRIFT (cache re-baselined this queue)
- raw: remote-results/2026-06-15-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-15

## Transfer note
Per-head learnable RoPE base (one learnable scalar per head per block) was a tiny lever (~48 scalars per layer × 12 layers = 576 extra params) expected to win small (-0.003 to -0.012). Result: +0.011 wrong-sign at tiny1m3m, well inside the ±0.04 noise band. The 500k global base from the closed-axes sweep remains the right setting at 0.94M; per-head frequency specialization does not bind at this scale. Transfer risk is **low** (RoPE family is well-validated at ≥100M; this is a narrow specialization), but the sub-claim — that *different heads want different rotation rates* — finds no support at 0.94M. Closes the **per-head RoPE base** axis at tiny1m3m. Re-evaluate at ≥135M where per-head gradient signal is richer; if the per-head bases converge to the same 500k, the global optimum is genuinely global; if they diverge, the axis is real.