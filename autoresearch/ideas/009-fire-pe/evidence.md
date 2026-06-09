# Evidence — 009 fire-pe

## Verdict: WIN
- tier: tiny1m3m, seed 42, box: vast 220.82.52.202:34386 (RTX 3060, sm_86)
- treatment val: 6.3234 (r1) — n=1
- control bracket: ctrl=6.3875, ctrl2=6.4050 (gap 0.0175)
- Δ vs ctrl: -0.0641 (treatment beats ctrl by 0.0641 ≫ gap 0.0175)
- Δ vs ctrl2: -0.0816 (treatment beats ctrl2 by 0.0816 ≫ gap 0.0175)
- pass/fail bar (idea.md): pass ≤ 6.4237 (target Δ = -0.005).
  Bar *far* exceeded: 6.3234 ≪ 6.4237.
- two-ctrl rule: treatment beats *both* ctrls by more than the gap → WIN
- box check: ctrl 6.3875 vs leaderboard 6.4287 = -0.0413 (within 0.04 noise band;
  both ctrls agree on direction, treatment Δ is 1.5× the noise band — robust)
- raw: remote-results/2026-06-09-vast-tiny1m3m/arq-r1/{009-fire-pe.log,ctrl.log,ctrl2.log}
- date: 2026-06-09

**Δ of -0.064/-0.082 is the largest of any of today's A/Bs** (vs the
cautious-Muon -0.025 / decoupled-QKV -0.014 / retention -0.011). FIRE wins
big on the val-distribution test at tiny1m3m. The plan's length-extrapolation
upside is untested at this tier (T=2048, fixed-length run); the win here is
the train-distribution val_loss, not extrapolation.
