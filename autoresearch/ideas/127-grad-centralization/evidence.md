# Evidence — 127 grad-centralization

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.4969   Δ: +0.0697
- ctrl2: pending (queue still running 128+ + ctrl2); even a favorable ctrl2 cannot recover this magnitude — verdict is decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), standard null band |Δ|<0.01 → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/127-grad-centralization.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.4317 vs ctrl 6.3966 ⇒ +0.035 train-loss gap ⇒ GC's gradient re-centering adds noise (subtracts the per-filter mean) but the forward/backward path is healthy
- Treatment val curve: 10.81 → 8.30 → 7.81 → 7.58 → 7.41 (step 100) → 7.11 → 6.97 → 6.79 → 6.65 → 6.50 final
- Ctrl val curve: 10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final
- Curves track each other through step 100, then GC falls visibly behind (Δ widens from ~0.0 to ~0.07 by end). The +0.07 gap is large and wrong-sign — GC is hurting not helping

## Transfer note
Gradient Centralization (Yong et al. 2020, "Gradient Centralization: A New Optimization Paradigm for Deep Neural Networks") re-centers the gradient on each filter: `g ← g − mean(g)` along the appropriate axis before applying the optimizer. The intuition: zero-mean gradients reduce the effective input distribution shift, which should help generalization (paper reports +0.5-2% on CIFAR/ImageNet for CNNs). The mechanism is well-validated for *CNNs* (the paper's domain) where filters have spatial structure and the mean-of-filter is a meaningful quantity. For transformer FFN linear layers, the "mean of a 2D weight matrix column" is a *much* less semantically meaningful operation — the weight matrix doesn't have a "filter" interpretation, and zero-centering it just adds noise. The +0.07 wrong-sign result at tiny1m3m is consistent with "GC adds noise to a non-CNN architecture without paying back." The mechanism is a no-promote null for transformer LMs. Could re-test at a CNN tier (e.g., a small ResNet at screen10m) where the lever might fire, but for the LM pipeline (no CNN stage in Phase-2) this is closed.
