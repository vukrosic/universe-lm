# Evidence — 139 lion

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86)
- control val: 6.4509   treatment val: 7.5134   Δ: +1.0625 (wrong-sign)
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ leaderboard_ctrl 6.4306 − 0.01 (≈ 6.4206); NULL band |Δ| < 0.01; DRIFT > +0.01 → not met (Δ is 106× the WIN-bar ceiling in the wrong direction)
- box check: ctrl 6.4509 vs leaderboard ctrl 6.4306 = +0.0203 (within ~0.04 box noise, box healthy)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-14

## Transfer note
Lion's lever fails to fire at 0.94M with the paper's `lion_lr=3e-4` default — train_loss 7.48 vs ctrl 6.41 confirms the model effectively under-trains across all 92 update steps. The sign-update's bounded magnitude and decoupled WD scaling interact badly with the lr scheduled for the much larger AdamW peak 0.006: a 20× LR gap cannot be recovered by the +1/-1 saturation. Same horizon-scaling wrong-sign null pattern as 110-weight-ema (Δ=+1.08), 122-tiger (Δ=+1.26), 124-radam (Δ=+0.89), 134-mega-ema (Δ=+0.039), 135-adan (Δ=+0.054) — all optimizer-wave ideas where the LR/EMA-window schedule is paper-tuned for ≥100k steps and the second-momentum buffer is essentially at init by step 92. The Cautious-Lion idea (separate slot, gating the sign-mask on top of bare Lion) inherits this baseline null and is **also null** at tiny1m3m without a separate A/B; do not file a fresh re-test at the current tier. The lever should re-evaluate at ≥135M Phase-2 with 3-4k steps where Lion's two EMA buffers (β1=0.9 effective window ~10 steps, β2=0.98 effective window ~50 steps) have time to develop a distinguishable posterior vs AdamW's m/√v path.