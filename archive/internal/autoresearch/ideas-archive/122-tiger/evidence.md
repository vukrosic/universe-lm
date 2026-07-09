# Evidence — 122 tiger

## Verdict: NULL (tier-mismatch, optimizer under-shoots)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 7.6891   Δ: +1.2619
- ctrl2: pending (queue still running 123+ + ctrl2); even a favorable ctrl2 cannot recover this magnitude — verdict is decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), standard null band |Δ|<0.01 → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/122-tiger.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 7.6573 vs ctrl 6.3966 ⇒ +1.26 train-loss gap ⇒ Tiger's adaptive LR is severely under-shooting
- Tiger LR schedule: 7e-5 (step 0) → 9.8e-4 (step 25, peak) → 8.8e-4 (step 100) → 4.6e-4 (step 400)
- Compare to ctrl LR schedule: 1.7e-3 (step 0) → 2.4e-2 (step 25, peak) → 1.7e-2 (step 200) → 1.1e-2 (step 400) — ctrl's LR is **~20× larger** than Tiger's at every step
- Treatment val curve: 10.81 → 9.61 → 7.94 (step 50) → 7.81 (step 75) → 7.78 (step 100) → 7.74 → 7.73 → 7.71 → 7.70 → 7.69 final — flat-lines from step 75 onward
- Val accuracy: 4.25% throughout after step 50 — model has learned to predict one class
- The optimizer is not diverging (no NaN, no large spikes) but is moving so slowly that ~92 update steps × tiny effective LR ≈ no learning past warmup

## Transfer note
Tiger is signSGD with an adaptive coordinate-wise scale `c` (Chen et al. 2024, "An Empirical Study of Tiger Optimizer"). The lever is supposed to be a fast, scalable alternative to Adam that doesn't need a d-coefficient. Tiger's d_estimate = `‖g‖²/‖g‖` style estimate needs *enough* gradient mass to converge — at 92 update steps with batch=2, the per-step gradient norm is so noisy that the scale estimate is *small* (hence the tiny LR). This is the same pattern as 110-weight-ema (effective EMA window >> training horizon) and 123-came, 124-radam (likely to behave similarly): adaptive-LR optimizers that need a longer horizon to converge their internal scale estimate. The mechanism should be re-evaluated at Phase-2 (135M, ~3-4k steps) where the gradient norm estimate stabilizes. Do not promote on this null; the issue is structural, not numerical.
