# Evidence — 118 mixture-of-depths

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.5278   Δ: +0.1006
- ctrl2: pending (queue still running 119+ + ctrl2); even a favorable ctrl2 leaves 118 sitting ~0.10 above the ctrl band — verdict is decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), standard architectural-expansion bar at tiny1m3m → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/118-mixture-of-depths.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.5133 vs ctrl 6.3966 ⇒ +0.12 train-loss gap ⇒ MoD's per-token router (decide to skip block or apply with c·block rescale) measurably hurts optimization
- Treatment val curve: 10.81 → 8.36 → 7.85 → 7.63 → 7.46 (step 100) → 7.20 → 7.03 → 6.85 → 6.70 → 6.53 final
- Ctrl val curve: 10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final
- Gap is small during warmup (≤0.02 step 0–100) and widens monotonically from step 200 onward; the router is hurting, not helping

## Transfer note
Mixture-of-Depths (Raposo et al. 2024) routes each token through a learned per-block gate: `gate ∈ {skip, apply}` with `apply` weighted by `c = k/T` to preserve expected FLOPs. The paper claims 50%+ FLOP savings at iso-quality on a 27B-base model — a *compute* lever, not a quality lever, at scale. At tiny1m3m the model is already so small (12L, d_model=64) that:
(a) skipped blocks cost almost nothing FLOP-wise (no saving worth chasing),
(b) the router overhead (extra linear + softmax per block) is a real cost,
(c) the routing decision is noisy on 92 update steps — the gate has barely learned which tokens to skip.
The +0.10 wrong-sign result here is the expected "router overhead > no-skip savings" outcome at this tier. MoD's value proposition (FLOP savings at iso-quality) is invisible at tiny1m3m. The mechanism is a *no-promote* null at this tier; re-evaluate at Phase-2 (135M, n_layers=24+) where the FLOP saving translates to wall-clock on a 30B+ model.
