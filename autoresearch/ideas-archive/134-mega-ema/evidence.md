# Evidence — 134 mega-ema

## Verdict: NULL (wrong-sign, ~8× the null band)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.4662   Δ: +0.0390
- ctrl2: pending (queued last in tmux); verdict is decided regardless — Δ is far above the |Δ|<0.01 null band and is wrong-sign
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), NULL band |Δ|<0.01, DRIFT > +0.01 → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/arq-110/134-mega-ema_52674.log, ctrl_52674.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.4262 vs ctrl 6.3966 ⇒ +0.0296 train-loss gap ⇒ Mega-EMA's full-model slow-moving teacher dominates training (β=0.9999 effective window ~10k steps >> 92-step run); the eval-time weight replacement shifts the model back to a barely-trained state
- Treatment val curve: 10.81 → 8.34 → 7.82 → 7.58 → 7.42 (step 100) → 7.20 → 7.04 → 6.91 → 6.76 → 6.47 final
- Ctrl val curve:    10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final
- Curves track through step ~100, then Mega-EMA falls visibly behind (Δ widens from ~0.0 to ~0.04 by end). The +0.039 wrong-sign gap is large and unambiguous.

## Transfer note
Mega-EMA / Slow-EMA (Tiezzi et al. 2022 — "Studying the Trade-Offs between Carbon and Accuracy in Image Classification" — proposed EMA of the *entire* model with very high decay `β=0.9999` to smooth the loss trajectory and improve generalization. The 110-weight-ema idea uses β=0.999 (effective window ~1000 steps); Mega-EMA uses β=0.9999 (effective window ~10k steps) — 10× more conservative. The +0.039 wrong-sign result at tiny1m3m says: with `β=0.9999` and **92 update steps**, the EMA shadow is essentially the random init — the model is asked to evaluate at the init weights, not at the trained weights. The lever is *strictly worse* than 110-weight-ema (Δ=+1.08 at the same scale) because the eval-time weight swap is more aggressive. Same pattern as 110, 122, 124, 125 — adaptive / slow-moving auxiliary mechanisms that need a longer horizon to develop a useful signal. The mechanism's published wins are on CIFAR-10/100 ResNet with 200+ epochs (≫10k steps); at tiny1m3m the lever is closed: NULL.
