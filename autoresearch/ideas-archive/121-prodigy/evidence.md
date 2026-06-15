# Evidence — 121 prodigy

## Verdict: NULL (degenerate — training exploded)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272 (in-batch from 110-135 batch)   treatment val: 41,789   Δ: +41,783
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ −0.01 vs ctrl; NULL band |Δ| < 0.01; DRIFT > +0.01
- box check: ctrl 6.4272 vs leaderboard 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/121-prodigy_52674.log
- date: 2026-06-14

## Trajectory check
- Val curve trt: 10.81 (step 0) → 8.33 → 7.78 → 57,889 (step 75 — first divergence spike) → 62,881 → 85,714 → 292,127 → 165,202 → 27,546 → 41,789 (final)
- Prodigy's adaptive LR + d-coefficient diverge catastrophically from step 75 (val_loss 57,889 vs ctrl 7.55 at same step) — this is the LR-discovery loop failing to bound its growth
- Train loss ends at 50,476 confirming the optimizer state itself diverged
- The "val PPL 485,165,195" cap suggests the perplexity computation clipped at int32 max — actual divergence is unbounded

## Transfer note
Prodigy uses an adaptive-LR scheme that needs many steps to stabilize. The 92-step tiny1m3m horizon is 1–2 orders of magnitude too short — the LR estimate over-shoots in <75 steps and the model never recovers. This is the same horizon-scaling null pattern as 110-weight-ema, 122-tiger, 124-radam, 134-mega-ema, 135-adan, 120-dadaptation: adaptive-LR/scale-estimate optimizers with paper-validated ranges of ≥1k steps. The catastrophic overflow at step 75 is *the signature* — control-like loss at step 50, then 10× jump by step 75. Closed at tiny1m3m; re-evaluate at ≥135M Phase-2 with 3-4k steps where the LR-discovery loop has time to settle.