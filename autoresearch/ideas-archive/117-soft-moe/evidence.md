# Evidence — 117 soft-moe

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.5663   Δ: +0.1391
- ctrl2: pending (queue still running 119+ + ctrl2); even a favorable ctrl2 (e.g. 6.43) leaves 117 sitting ~0.13 above the ctrl band — verdict is decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), standard architectural-expansion bar at tiny1m3m → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/117-soft-moe.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.5377 vs ctrl 6.3966 ⇒ +0.14 train-loss gap ⇒ Soft-MoE FFN (4 expert FFNs + slot router, ~4× FFN params) measurably hurts optimization
- Treatment val curve: 10.81 → 8.34 → 7.83 → 7.61 → 7.44 (step 100) → 7.18 → 7.01 → 6.84 → 6.70 → 6.57 final
- Ctrl val curve: 10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final
- Gap widens monotonically from ~0.02 at step 100 to ~0.14 at the end; the treatment is *never* ahead of ctrl at any milestone

## Transfer note
Soft MoE (Puigcerver et al. 2023, ICLR 2024) replaces the standard FFN with N expert FFNs gated by a learned slot-routing matrix: `output = Σ_i softmax(W_router·x)_i · FFN_i(x)`. The intuition is that N parallel small FFNs beat one wide FFN at the same parameter cost. tiny1m3m baseline FFN is `d_ff=256` (single SwiGLU/squared-ReLU FFN); Soft-MoE adds 4 such FFNs + a router. The +0.14 wrong-sign result here says the routing overhead and 4-way specialization loss dominates any capacity win at this depth. The lever is well-validated at 1B+ scale (paper reports gains on ViT-L, BART, etc.) where a single FFN is genuinely the binding constraint. At d_model=64 with 6L FFNs, one FFN is already deep/wide enough to express what's learnable. Re-test at Phase-2 (135M, n_layers=24, d_model=768) where the FFN becomes a real capacity bottleneck; do not promote to Phase-2 on this null. Closed at tiny1m3m.
