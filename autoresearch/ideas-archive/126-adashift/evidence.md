# Evidence — 126 adashift

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.9837   Δ: +0.5565
- ctrl2: pending (queue still running 127+ + ctrl2); even a favorable ctrl2 cannot recover this magnitude — verdict is decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), standard null band |Δ|<0.01 → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/126-adashift.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.9158 vs ctrl 6.3966 ⇒ +0.52 train-loss gap ⇒ AdaShift's decoupled v̂ from g² accumulation is destabilizing early training
- Treatment val curve: 10.81 → 10.19 → 11.92 (step 50, val went UP) → 7.96 (step 75, recovered) → 7.80 → 7.72 → 7.52 → 7.34 → 7.20 → 6.98 final — slow descent, never catches ctrl
- Ctrl val curve: 10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final — clean monotonic descent
- The early up-tick (val loss INCREASING from 10.81 → 11.92 between steps 25 and 50) is the smoking gun: AdaShift's decoupling of `v̂` from `g²` (using `n=3` lag) means the second-moment estimate is stale in the first ~25 steps, so the denominator `√v̂ + ε` is *under-estimating* the actual gradient noise → over-scaled updates → val loss spikes

## Transfer note
AdaShift (Zhou et al. 2019) modifies Adam to use `v̂_t = (1/n) · Σ_{k=0}^{n-1} g²_{t-k}` (average of last n squared gradients) instead of the EMA `v̂_t = β₂ · v̂_{t-1} + (1−β₂) · g²_t`. The intuition: EMA has a "warmup" problem where the first few steps have very small `v̂` and the optimizer takes dangerously large steps. The `n=3` (paper default) variant averages the last 3 squared gradients, giving a less stale second-moment estimate. The +0.56 wrong-sign result here says: in 92 update steps × tiny1m3m, the 3-step average is also stale (each step's gradient is from a single 2×2048 batch — very noisy), and the resulting under-estimation of `√v̂` produces over-scaled updates that *never recover*. The same pattern repeats for any optimizer that relies on a short lookback for its second-moment estimate at this noisy-batch regime. Re-evaluate at Phase-2 (135M, larger effective batch, 3-4k steps) where `n=3` is closer to enough history. Do not promote on this null.
