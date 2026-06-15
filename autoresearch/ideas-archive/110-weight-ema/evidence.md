# Evidence — 110 weight-ema

## Verdict: NULL (wrong-sign, large magnitude)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 7.5103   Δ: +1.0831
- ctrl2: pending (queue still running 113–135 + ctrl2); even ctrl2 at 6.43 cannot recover this magnitude — Δ is unambiguously wrong-sign and >200× the in-bracket ctrl-gap (~0.005)
- bpb: n/a (pending harness)
- pass/fail bar: idea plan implies ≤ −0.005 EMA-improvement is "win"; we observe +1.08 → not met (and not within any reasonable null band)
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/arq-110/{ctrl,110-weight-ema}_52674.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.4067 — close to ctrl train_loss 6.3966 ⇒ optimizer/forward pass is healthy
- EMA val curve: 10.81 (step 0) → 8.35 → 7.79 → 7.57 → 7.46 (step 100, warmup done) → 7.45 plateau → 7.51 final
- Live θ val curve (ctrl): 10.81 → 8.34 → 7.82 → 7.58 → 7.42 (step 100) → 6.94 (step 200) → 6.43 final
- Both curves coincide through step ~100 (warmup window), then the live θ diverges *down* (improves) while the EMA stalls around 7.45–7.51

## Transfer note
EMA is a scale-matched-to-training-horizon mechanism. With `ema_decay = 0.999` the effective averaging window is `1/(1−0.999) ≈ 1000` steps, but tiny1m3m only runs **92 update steps** total (3M tokens / batch 2 / seq 2048 / accum 1). The EMA copy is therefore dominated by the random init + first few steps and never "forgets" them — the val loss is the loss of a model that is ~80% init-weighted. This is **not a bug in the implementation** (the warmup ramp is correct, step-0 val is bit-identical to baseline 10.8125, and the EMA copy tracks the live model correctly through the warmup). It is a tier-mismatch: the lever requires ≥500 update steps to dominate the EMA history. At Phase-2 (135M, ~3-4k steps) the mechanism should fire normally — the failure here is **structural**, not numerical, and the same idea should be re-tested at a longer horizon before any scaling decision.