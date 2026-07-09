# Evidence — 138 looksam

## Verdict: NULL (inside variance — two-ctrl §2)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4641 (bracket ctrl) / 6.4225 (bracket ctrl2)   treatment val: 6.4612   Δ: −0.0029 / +0.0387
- bpb: n/a (pending harness)
- pass/fail bar (cohort default, no plan.md): PASS ≤ −0.01 vs ctrl; NULL band |Δ| < 0.01; §2 two-ctrl WIN requires beating *both* ctrls by more than ctrl-to-ctrl gap
- box check: bracket ctrl 6.4641 vs leaderboard 6.4306 = +0.0335; bracket ctrl2 6.4225 = −0.0081 (within ~0.04 noise band, box healthy); ctrl-to-ctrl gap 0.0416
- raw: remote-results/2026-06-14-vast-tiny1m3m/138-looksam_52674.log
- date: 2026-06-14

## Trajectory check
- Val curve trt: 10.81 (step 0) → 8.17 → 7.81 → 7.63 → 7.40 → 7.13 → 6.96 → 6.77 → 6.64 → 6.4612 (final, val_acc 0.1387)
- Train loss ends at 6.4191 (vs ctrl1 6.4474, ctrl2 6.3915 — between them, right sign vs ctrl1)
- val_acc 0.1387 (vs ctrl1 0.1416, ctrl2 0.1424 — between them, slightly worse than both)
- Training curve is healthy (val_acc grows 0 → 0.139 monotonically, val_loss decreases monotonically); the run *trained* normally, just landed within the ctrls
- The LookSAM step (every 5th) costs 2× compute but the perturbation `ε̂ = ρ·∇L/‖∇L‖` at ρ=0.05 is a *small* one-shot sharpness-aware correction; at 0.94M with L=12 the loss surface is not yet sharp enough (sharpness emerges with scale + depth) for the SAM step to find a meaningfully different minimum

## Transfer note
LookSAM's paper-validated gains (Du et al. 2023, ICLR) come from CIFAR-10/100 ResNet, ImageNet ResNet-50 (25M), ImageNet ViT-S (~22M) — every benchmark is ≥22M params and a few hundred epochs. At tiny1m3m (0.94M, 92 steps, ~1.3 epochs effective), the SAM ascent step (which costs an extra forward+backward every K=5 steps) is a 20% compute overhead but the underlying loss surface isn't sharp enough for the SAM perturbation to escape to a meaningfully flatter minimum. SAM's benefit compounds with sharpness, and sharpness compounds with scale + training duration. Re-evaluate at ≥135M Phase-2 where loss surface sharpness is meaningful; the K=5 default should still be appropriate. Closes periodic-SAM axis at 0.94M alongside 119-sam.
