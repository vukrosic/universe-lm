# Evidence — 001 cautious-muon

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast 220.82.52.202:34386 (RTX 3060)
- treatment val: 6.4125 (r1), 6.4156 (r2)  — n=2
- control: ctrl/ctrl2 over two batches = 6.3875 / 6.4050 / 6.4322 / 6.4009 (mean 6.4064)
- Δ vs mean ctrl: +0.006 (wrong sign — treatment is slightly *worse*)
- pass/fail bar: needed ≤ ~-0.01 vs ctrl → not met
- box check: ctrl spread 6.3875–6.4322 (~0.045, flash-kernel nondeterminism); within expected
- raw: remote-results/2026-06-09-vast-tiny1m3m/arq-r1 + arq-r2 (results + logs)
- date: 2026-06-09

Note: an earlier orphan sweep saw cautious-Muon at Δ-0.0249/-0.0094; with a proper
two-ctrl bracket it lands inside noise (and slightly negative). Clean null.
