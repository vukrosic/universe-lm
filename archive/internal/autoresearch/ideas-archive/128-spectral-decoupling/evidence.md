# Evidence — 128 spectral-decoupling

## Verdict: NULL (wrong-sign, 20× null band)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.5272   Δ: +0.1000
- ctrl2: pending (queue still running 130+ + ctrl2); even a favorable ctrl2 cannot recover this magnitude — verdict decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), NULL band |Δ|<0.005, DRIFT > +0.005 → not met (and not inside any reasonable null band)
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/128-spectral-decoupling.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.5021 vs ctrl 6.3966 ⇒ +0.105 train-loss gap ⇒ SD's gradient projection is destabilizing the AdamW update
- Treatment val curve: 10.81 → 8.30 → 7.81 → 7.58 → 7.41 (step 100) → 7.12 → 7.00 → 6.87 → 6.72 → 6.53 final
- Ctrl val curve:    10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final
- Curves track through step ~100, then SD falls visibly behind (Δ widens from ~0.0 to ~0.10 by end). The +0.10 wrong-sign gap is large and unambiguous.

## Transfer note
Spectral Decoupling (Yong et al. 2022, arXiv:2202.05380) reformulates weight decay so the *regularization* term shrinks only the magnitude of `w` and the *gradient* update is projected off the `w` direction: `g ← g − (⟨g,w⟩/‖w‖²)·w` before delegating to AdamW. The intuition: standard L2 `λ·w` rotates weights toward the origin, fighting the gradient's preferred direction; the projection is supposed to remove that rotation cost. The +0.10 wrong-sign result at tiny1m3m says: with `sd_lambda=0.01` and 92 update steps, the projection is removing too much of the gradient signal in the `w` direction — the model still learns but reaches a worse final loss. The mechanism's published wins are on CNNs (CIFAR/ImageNet ResNet-50) where the magnitude-vs-direction distinction is sharper for convolutional filters; on transformer FFN linear layers the per-token weight semantics are weaker and the projection is more like a damped gradient than a "rotation correction." Pattern is consistent with 127-grad-centralization (also a gradient-space intervention that is null/wrong-sign at this scale). Re-evaluate at Phase-2 (135M, 3-4k steps) where (a) the gradient direction is more stable across steps so projection is less destructive, and (b) the decoupled-WD rotation accumulates over more steps and may be more clearly hurting — but for tiny1m3m the lever is closed: NULL.
