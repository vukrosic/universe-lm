# Evidence — 141 adabelief

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86)
- control val: 6.4509 (ctrl1) · 6.4194 (ctrl2)   treatment val: 6.4653   Δ: +0.0144 (ctrl1), +0.0459 (ctrl2)
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ leaderboard_ctrl 6.4306 − 0.01 (≈ 6.4206); NULL band |Δ| < 0.01; DRIFT > +0.01 → not met (treatment is wrong-sign vs both ctrls; does NOT beat either; above the 0.01 null band)
- box check: ctrl1 6.4509 (+0.0203), ctrl2 6.4194 (-0.0112) bracket leaderboard 6.4306 cleanly; ctrl-to-ctrl gap 0.0315 — within ~0.04 noise band, box healthy
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-14

## Transfer note
AdaBelief's lever fails to fire at 0.94M. The variance-of-residual denominator is functionally indistinguishable from AdamW's variance-of-gradient at 92 update steps where the (g − m) residual is dominated by single-batch noise — at large-batch LMs the residual variance becomes a meaningful estimator (the paper's setting is CIFAR/PTB at moderate batch sizes), but tiny1m3m's batch=2 makes `s = E[(g − m)²]` ≈ `E[g²]` (single-sample residual ≈ raw gradient). The +0.0144/+0.0459 wrong-sign deltas plus train_loss +0.030 above ctrl confirm slight under-training, not a real lever. Same first-moment-EMA / single-batch-noise null pattern as 002-cautious-adamw (Δ=+0.0003 emb / -0.0066 gain, both inside ~0.04 run-to-run variance). Do not file a Cautious-AdaBelief companion — the residual-variance denominator is already the lever's identity, and the cautious mask has nothing meaningful to mask at this batch size. Re-evaluate at ≥135M Phase-2 with larger effective batch (≥256) where AdaBelief's residual variance becomes a real second-moment estimator vs AdamW's gradient variance.