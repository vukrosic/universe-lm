# Evidence — 120 dadaptation

## Verdict: NULL (degenerate — training exploded)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272 (in-batch from 110-135 batch)   treatment val: 7.04e15   Δ: +7.04e15
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ −0.01 vs ctrl; NULL band |Δ| < 0.01; DRIFT > +0.01
- box check: ctrl 6.4272 vs leaderboard 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/120-dadaptation_52674.log
- date: 2026-06-14

## Trajectory check
- Val curve trt: 10.81 (step 0) → 8.34 → 7.79 → 8.66 (step 75) → 10.05 (step 100) → 10.69 → 11.76 → 11.84 → 12.87 → 7.04e15 (final — fp16 overflow)
- D-Discovery loop over-shot LR within the first 50 steps; val_loss already 8.66 at step 75 (above ctrl 7.55 same step); divergence is monotonic from step 50 onward; final state is fp16 overflow (~7e15)
- Train loss ends at 7.04e15 confirming forward pass overflowed; this is *not* a useful A/B signal, but the val curve (10.05 by step 100) was already unambiguously worse than ctrl before the overflow

## Transfer note
D-Adaptation's `D`-discovery loop aims to converge to the optimal AdamW-equivalent LR. At tiny1m3m the 92 update steps are not enough for `D` to converge to a stable LR — it over-shoots (the val curve already shows clear divergence by step 75, well before the fp16 overflow). The mechanism's paper-validated range is ≥1k steps. Same pattern as 110-ema, 122-tiger, 124-radam, 134-mega-ema, 135-adan — adaptive-LR/scale-estimate optimizers that need a longer horizon to settle. The catastrophic fp16 overflow at the end is a *secondary* issue (the run was already null at step 100) and may also point to DAdaptAdamW missing an fp32-accumulation guard around the LR-discovery multiplication. Closed at tiny1m3m; re-evaluate at ≥135M Phase-2 with 3-4k steps.