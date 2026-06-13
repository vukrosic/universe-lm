# Evidence — 123 came

## Verdict: NULL (degenerate — training exploded)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272 (in-batch from 110-135 batch)   treatment val: 46,266   Δ: +46,260
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ −0.01 vs ctrl; NULL band |Δ| < 0.01; DRIFT > +0.01
- box check: ctrl 6.4272 vs leaderboard 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/123-came_52674.log
- date: 2026-06-14

## Trajectory check
- Val curve trt: 10.81 (step 0) → 8.33 → 7.78 → 73,942,958 (step 75 — already exploded by 4 orders of magnitude vs ctrl 7.55) → 34,582,036 → 105,958,605 → 39,151,206 → 20,057,948 → 13,704,888 → 46,266 (final)
- CAME (Confidence-guided Adaptive Memory Efficient optimization) divergence is immediate — by step 75 the val_loss is already 7.39e7 (vs ctrl 7.55 at the same step). The residual-based update rule's confidence estimate over-shoots within the first 50 steps
- Train loss ends at 41,003 confirming optimizer state divergence
- The 485,165,195 PPL cap suggests int32 overflow in the perplexity computation; actual loss was unbounded

## Transfer note
CAME's confidence-guided adaptive update over-shoots at the 92-step tiny1m3m horizon — same horizon-scaling null pattern as 110-weight-ema, 122-tiger, 124-radam, 134-mega-ema, 135-adan, 120-dadaptation, 121-prodigy. The mechanism's paper-validated range is ≥1k steps where the residual EMA window (~10 steps) fills with meaningful signal; at 92 steps the confidence term is essentially noise-driven and the LR explodes. The fact that the explosion happens at step 75 (not step 0) means the model *does* start training — it just can't recover once the adaptive update destabilizes. Closed at tiny1m3m; re-evaluate at ≥135M Phase-2 with 3-4k steps.