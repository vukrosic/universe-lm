# Evidence — 137 adamp

## Verdict: NULL (wrong-sign, undertraining)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4641 (bracket ctrl) / 6.4225 (bracket ctrl2)   treatment val: 6.5550   Δ: +0.0909 / +0.1325
- bpb: n/a (pending harness)
- pass/fail bar (from idea.md): PASS ≤ −0.01 vs ctrl; NULL band |Δ| < 0.01
- box check: bracket ctrl 6.4641 vs leaderboard 6.4306 = +0.0335; bracket ctrl2 6.4225 = −0.0081 (within ~0.04 noise band, box healthy); ctrl-to-ctrl gap 0.0416
- raw: remote-results/2026-06-14-vast-tiny1m3m/137-adamp_52674.log
- date: 2026-06-14

## Trajectory check
- Val curve trt: 10.81 (step 0) → 9.89 → 8.60 → 8.00 → 7.79 → 7.54 → 7.35 → 7.12 → 6.94 → 6.5550 (final, val_acc 0.1322)
- Train loss ends at 6.5094; val_acc 0.1322 (vs ctrl 0.1416/0.1424)
- Training is *not* degenerate (val_acc grows from 0 → 0.13, train_loss decreases monotonically); the run is just slightly undertrained relative to AdamW ctrl
- Projection-based update shrinks the effective update magnitude (orthogonal-to-weight projection removes some of the magnitude-aligned component); the lever's intent is correct for scale-invariant weights but at 0.94M the model has very few scale-invariant-only weights (most 2-D matrices have meaningful magnitude changes too), so the projection *throttles* legitimate updates
- Same direction as 002-cautious-adamw — partial-update variants that need either a longer horizon to amortize the projection cost OR a larger lr to compensate

## Transfer note
AdamP's projection trick (He et al. 2020) targets ResNet/ViT-style scale-invariant weights where magnitude and direction should be decoupled. At tiny1m3m (0.94M, 12L, d_model=64) the 2-D matrices are too small for scale-invariance to dominate — every weight matrix has both meaningful magnitude and meaningful direction updates. The projection therefore *shrinks* the effective update without any of the intended benefit. Paper-validated range (CIFAR/ImageNet/ViT/DETR) is all ≥25M params with deep stacks where scale-invariance emerges from layer-norm. Re-evaluate at ≥135M Phase-2 where scale-invariance is meaningful and a slightly higher `adamp_lr` (1.5–2× AdamW) can compensate for the projection. Closes the projection-based optimizer axis at 0.94M.
