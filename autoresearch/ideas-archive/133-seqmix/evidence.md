# Evidence — 133 seqmix

## Verdict: NULL (wrong-sign, just above null band)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.4394   Δ: +0.0122
- ctrl2: pending (queued last in tmux); verdict is decided regardless — Δ is above the |Δ|<0.01 null band and wrong-sign
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ 6.4206, NULL band |Δ|<0.01, DRIFT > +0.01 → not met (DRIFT)
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/arq-110/133-seqmix_52674.log, ctrl_52674.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.4168 vs ctrl 6.3966 ⇒ +0.0202 train-loss gap ⇒ SeqMix's Beta(α,α)-interpolated sequences produce *noisy* CE targets that the small 0.94M model cannot fit reliably
- Treatment val curve: 10.81 → 8.34 → 7.83 → 7.59 → 7.42 (step 100) → 7.16 → 7.01 → 6.86 → 6.72 → 6.44 final
- Ctrl val curve:    10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final
- Curves track through step ~100, then SeqMix falls visibly behind (Δ widens from ~0.0 to ~0.012 by end). The +0.0122 wrong-sign gap is DRIFT (>0.01).

## Transfer note
SeqMix (Jindal et al. 2020, "SeqMix: Augmenting Data Sequence for Improved Sequence Models") is a Mixup-style input-space augmentation: for each step, sample two distinct sequences x and x_b from the same batch, mix them as `x_mix = λ·x + (1−λ)·x_b` with `λ ~ Beta(α,α)`, and compute the loss as `λ·CE(x_mix, y) + (1−λ)·CE(x_mix, y_b)`. The intuition: smooth interpolation between real examples is a load-bearing regularizer, especially for small data. The +0.0122 wrong-sign result at tiny1m3m says: with `α=0.4` (Mixup-style strong mixing) and 92 update steps, the interpolated sequences do *not* correspond to real text — token-level mixing of two unrelated sequences produces out-of-distribution inputs that the small model cannot generalize from. The lever's published wins are on (a) small data + medium model (Mixup for CIFAR-10) and (b) NLP at ≥100M; the 0.94M model is *too small* to absorb the input noise as a regularizer (it just learns the noise). Pattern matches Born-Again 132 (also a small data / short-horizon augmentation-style lever that nulls at this scale). Re-evaluate at Phase-2 (135M, 3-4k steps) where the model has capacity to filter the input noise — but for tiny1m3m the lever is closed: NULL.
