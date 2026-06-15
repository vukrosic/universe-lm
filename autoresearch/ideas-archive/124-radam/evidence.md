# Evidence — 124 radam

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 7.3169   Δ: +0.8897
- ctrl2: pending (queue still running 125+ + ctrl2); even a favorable ctrl2 cannot recover this magnitude — verdict is decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), standard null band |Δ|<0.01 → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/124-radam.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 7.2096 vs ctrl 6.3966 ⇒ +0.81 train-loss gap ⇒ RAdam's rectification term is destabilizing early training
- Treatment val curve: 10.81 → 16.17 (step 25, large swing) → 19.32 (step 50) → 9.65 (step 75) → 10.87 (step 100) → 17.00 (step 150) → 8.20 (step 200) → 7.82 (step 300) → 7.76 (step 400) → 7.32 final — wildly oscillating, never stabilizes
- Ctrl val curve: 10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final — clean monotonic descent
- RAdam's `ρ_∞` rectification term (Liu et al. NeurIPS 2019) is supposed to *fix* Adam's bad initial steps by computing a per-coordinate variance correction `ρ_t` that is small at the start and grows as `sma_max` is filled. With only 92 update steps and a 4-step warmup, `ρ_t` may not have stabilized — the oscillating pattern (16, 19, 9, 10, 17) is consistent with an unstable rectification that thrashes between high-variance and low-variance regimes

## Transfer note
RAdam (Rectified Adam, Liu et al. 2019) adds a per-step rectification term `ρ_t = ρ_∞ − 2·(t−4)·(1−ρ_∞) / ((1−t/4)·(t+1)·(1+ρ_∞))` based on the SMA (simple moving average) of squared gradients. The lever is well-validated: paper reports +0.5-1% on CIFAR/WideResNet, and it's a drop-in AdamW replacement. At tiny1m3m the issue is that the SMA length is fixed (default 2/(1−β₂)−1 ≈ 2000 steps) — so for the first ~2k steps `ρ_t < 1` and updates are *under-scaled* (which is the whole point), but the *oscillation* pattern here suggests the variance term is being computed from a too-small window. This is a tier-mismatch: the rectification needs enough steps to fill its SMA before it stabilizes. The same idea should be re-evaluated at Phase-2 (135M, 3-4k steps) where the SMA fills before the run ends. Do not promote on this null.
