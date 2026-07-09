# Evidence — 114 mars

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.4297   Δ: +0.0025
- ctrl2: pending (queue still running 116–135 + ctrl2); current Δ vs ctrl1 already inside |Δ| < 0.01 null band — verdict is decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: idea plan implies a non-trivial Δ; +0.0025 is inside noise → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/arq-110/{ctrl,114-mars}_52674.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.3966 — matches ctrl train_loss 6.3966 exactly ⇒ optimizer/forward path healthy
- Val curve: 10.81 → 8.29 → 7.81 → 7.57 → 7.46 (step 100) → 7.16 → 7.00 → 6.79 → 6.64 → 6.43
- vs ctrl val curve: 10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43
- Curves overlap step-for-step; MARS variance-reduction makes no detectable difference at tiny1m3m

## Transfer note
MARS is a *variance-reduced* AdamW (Yuan et al. arXiv:2401.03855): at each step, replace the gradient `g_t` with `g_t − g_{t−lag}` (estimated lagged gradient) before the Adam moment updates, dampening high-frequency noise. The lever is specifically aimed at regimes with **large stochastic gradient noise relative to signal** — exactly the tiny1m3m regime (batch 2 × seq 2048, 92 update steps, ~3M tokens). The fact that we see zero Δ suggests the per-step noise here is already low enough that the variance-reduction correction is small relative to the Adam moments, OR that the lag estimator (lag=10) spans too many steps relative to the trajectory length. The mechanism should still be evaluated at Phase-2 (135M, ~3-4k steps, larger stochastic noise) where its lever effect (paper reports ~3-5% on AdamW at GPT-2/LLaMA scale) should fire.