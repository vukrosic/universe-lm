# Evidence — 167 logit-zloss

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674
- baseline: cached mean=6.4455 ±0.0524 (box 5b8a7fea8963, RTX 3060 sm_86 driver 580.159.03, measured 2026-06-14T09:00:17Z, n=11)
- treatment val: 6.4437   Δ vs baseline: -0.0018
- bpb: n/a (pending harness)
- pass/fail bar: PASS Δ ≤ -0.005; NULL |Δ| < 0.005; DRIFT > +0.005  → NULL (Δ=-0.0018 inside ±0.005)
- box check: baseline mean 6.4455 vs leaderboard 6.4216 — within noise (~0.024, < band 0.0524)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log: 167-logit-zloss_52674.log)
- date: 2026-06-14

## Transfer note
At tiny1m3m (0.94M params, 3M tokens) the z-loss auxiliary penalty at λ=1e-4 produced a Δ of −0.0018 — well inside both the per-idea ±0.005 plan bar and the cached ±0.0524 noise band. The penalty was active (z_loss_val was logged per step) and finite; the lever is wired through the loss path correctly. The lack of any measurable effect means logit-magnitude pressure is not the binding constraint at this tier — capacity is, which is consistent with all five closed 066-070 loss-shape axes also being NULL at tiny1m3m. At 135M scale where the logit distribution has more room to drift and bf16 quantization eats tail-mass, z-loss is known to help (PaLM, Gemma). Transfer risk is **med**: the lever is mechanically sound but only fires once logit explosion becomes a real failure mode, which does not happen at 0.94M. A future scale-up pass at ≥30M should re-test before adopting.
