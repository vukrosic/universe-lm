# Evidence — 153 Squared-ReLU FFN Activation

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, driver 580.159.03, commit 7f7fe90)
- baseline: cached mean=6.4394 ±0.04 (box 5b8a7fea8963, measured 2026-06-14 from 3 ctrls this queue)
- treatment val: **6.4341**   train=6.4081   acc=0.1443   Δ vs baseline: **-0.0053**
- pass/fail bar (plan.md): Δ≤-0.005 vs baseline (real WIN only, not null band) → **trt -0.0053 < bar -0.005 by 0.0003** (numerically a pass by a hair), but Δ sits inside the 0.04 noise band → **NULL by band rule**
- bpb: n/a (pending harness — never omit)
- box check: cached mean 6.4394 vs leaderboard 6.4306 Δ=+0.0088 (within noise, **NO DRIFT**)
- raw: `autoresearch/remote-results/2026-06-14-vast-tiny1m3m-2/{results.json, 153-relu2-ffn.log}`
- date: 2026-06-14

## Run trace
- step 0: val_loss=10.8125, val_acc=0.0000 → **bit-identity preserved at init** ✓
- step 100: val_loss=7.3941, val_acc=0.0845 (descending normally)
- step 200: val_loss=6.9881, val_acc=0.1051
- step 300: val_loss=6.7697, val_acc=0.1174
- step 400: val_loss=6.6353, val_acc=0.1292
- final (step 732): val_loss=**6.4341**, train=6.4081, acc=0.1443 (training stable, no divergence)

Stable training throughout. No anomalies. The val-loss gap is essentially noise.

## Transfer note
Primer (So et al. 2021) reported `ReLU²` matches SwiGLU at 125M-1.5B with no quality loss and one fewer matmul; Mercury Coder uses it in production. At 0.94M the d_ff=256 / d_model=64 regime is so small that the activation choice is dominated by Q/K/V/embedding gradients and the FFN activation is just a near-identity nonlinearity. A null at this tier is consistent with Primer's own finding that the win shows up at ≥125M (where the FFN's representational capacity is large enough for the activation curvature to matter). The lever should be re-evaluated at any future mid-scale tier; for tiny1m3m the axis is closed as "no measurable effect at 0.94M, mechanism qualitatively validated upstream."
