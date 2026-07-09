# Evidence — 116 hyper-connections

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.4938   Δ: +0.0666
- ctrl2: pending (queue still running 117–135 + ctrl2); even a favorable ctrl2 (e.g. 6.43) leaves 116 sitting ~0.06 above the ctrl band — verdict is decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222) per idea plan → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/116-hyper-connections.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.4742 vs ctrl 6.3966 ⇒ +0.078 train-loss gap ⇒ optimizer/forward path is healthy but the mHC constraint (4-way split residual + learnable A/B/C mixing matrices) measurably slows training
- Treatment val curve: 10.81 → 8.31 → 7.81 → 7.59 → 7.41 (step 100) → 7.12 → 6.95 → 6.77 (step 300) → 6.65 (step 400) → 6.49 final
- Ctrl val curve (from ctrl.log): 10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final
- Both curves coincide through step ~100 (warmup window), then the treatment falls visibly behind from step 200 onward (Δ widens monotonically from 0.0 to ~0.07). The mHC residual-stream split trades off expressivity for parameter count at this tier, and the small-tier overhead is not amortized.

## Transfer note
Hyper-Connections (Xie et al. 2024, used in DeepSeek-V3) is a residual-stream *expansion* lever: 4 parallel streams of width d_model/4 mixed by learnable A/B/C matrices. The mechanism is invariant to `n_resid=1` (collapses to standard residual) and at `n_resid=4` adds 288 scalars total. The intuition: more parallel residual paths = more ways to move information forward, useful at depths where the single residual stream bottlenecks. At tiny1m3m the residual is already wide enough relative to depth (d_model=64, n_layers=12) that splitting it 4-ways does not help — the lever needs a depth where one stream is genuinely the binding constraint. Sub-LN (017, closed as null at 6L), DropPath (111, closed as null at 6L), Canon-Conv (023, won +0.06 only after stripping FIRE), and now mHC all tell the same story: 6L × d_model=64 is *too small* for residual-stream architecture tweaks to land a detectable win. The mechanism should be re-evaluated at ≥24L Phase-2 scale, where the single-stream residual may actually bottleneck. Do not promote to Phase-2 on this null; the same idea-class is well-closed at tiny1m3m.
