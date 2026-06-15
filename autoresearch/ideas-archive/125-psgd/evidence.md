# Evidence — 125 psgd

## Verdict: NULL (tier-mismatch)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 7.6916   Δ: +1.2644
- ctrl2: pending (queue still running 126+ + ctrl2); even a favorable ctrl2 cannot recover this magnitude — verdict is decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), standard null band |Δ|<0.01 → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/125-psgd.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 7.6525 vs ctrl 6.3966 ⇒ +1.26 train-loss gap ⇒ PSGD's preconditioner never converges
- Treatment val curve: 10.81 → 10.54 → 9.65 (step 50) → 8.51 (step 75) → 7.80 (step 100) → 7.77 → 7.74 → 7.72 → 7.71 → 7.69 final — flattens from step 100
- Val accuracy stuck at 4.24% after step 100 (one class prediction)
- Same "optimizer under-shoots" pattern as 110-weight-ema, 122-tiger, 124-radam: the preconditioner / scale estimate is not converging in 92 update steps

## Transfer note
PSGD (Li et al. ICML 2024, "Preconditioned SGD") maintains a low-rank preconditioner matrix `P` that is updated each step via `P ← P · exp(η · (g gᵀ − something))`. The lever is supposed to be a fast diagonal/sparse preconditioner that outperforms Adam at scale. The preconditioner is updated in online fashion and *needs enough steps* for the running estimate to stabilize — at 92 update steps the preconditioner is still in its early "stumble" phase and produces under-scaled updates. The same pattern repeats for any second-order / adaptive-precondition optimizer: 92 steps is too short. Re-evaluate at Phase-2 (135M, ~3-4k steps) where the preconditioner converges before the run ends. Do not promote on this null.
