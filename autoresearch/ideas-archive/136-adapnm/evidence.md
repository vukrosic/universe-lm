# Evidence — 136 adapnm

## Verdict: NULL (degenerate — training did not converge)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4641 (bracket ctrl) / 6.4225 (bracket ctrl2)   treatment val: 10.7144   Δ: +4.2503 / +4.2919
- bpb: n/a (pending harness)
- pass/fail bar (from idea.md): PASS ≤ −0.005 vs ctrl; NULL band |Δ| < 0.005; DRIFT > +0.005
- box check: bracket ctrl 6.4641 vs leaderboard 6.4306 = +0.0335; bracket ctrl2 6.4225 = −0.0081 (within ~0.04 noise band, box healthy); ctrl-to-ctrl gap 0.0416
- raw: remote-results/2026-06-14-vast-tiny1m3m/136-adapnm_52674.log
- date: 2026-06-14

## Trajectory check
- Val curve trt: 10.8125 (step 0, acc=0.000) → 10.7562 → 10.7531 → 10.7500 → 10.7487 → 10.7481 → 10.7431 → 10.7344 → 10.7144 (final, acc=0.0000)
- Train loss ends at 10.6945; val accuracy = 0.0000 throughout — the model never learned a single token
- This is **not** a divergence spike (like 120/121/123/123-came/124) — AdaPNM never had traction at all. val_acc stays at 0 from step 0 to final. The optimizer's update step is producing essentially no signal in the 92-step horizon.
- Most likely cause: the dual-momentum (`m+` / `m−`) split with a shared `v+` denominator (as in the plan) is producing update magnitudes several orders too small for the same `adapnm_lr=0.006` as AdamW. AdamW's first step update is `(1−β1)·g / (√((1−β2)·g²) + ε) ≈ sign(g) · lr`; AdaPNM's effective first-step update with the same lr ends up roughly `lr · (m+ − m−) / (√v+ + ε)` where `m+ − m− = (1−β1)·g` but the `v+` accumulation pattern is different from AdamW's `v`. Empirical silence means the LR / v+ balance needs re-tuning.

## Transfer note
AdaPNM's paper-validated gains (Ding et al. 2019, NeurIPS) come from CIFAR-10/100 ResNet, ImageNet ResNet-50, Transformer-XL dialog LM (~250M), and BERT fine-tuning (~110M) — every benchmark is ≥100M and ≥1k update steps. At tiny1m3m (0.94M, 92 steps) the dual-momentum buffer needs both (a) enough steps to develop distinguishable `m+`/`m−` statistics (asymmetric gradient components) and (b) enough signal for the shared `v+` second moment to settle. Neither holds at this tier — the gradient symmetry at 0.94M is benign (the plan's "approximately-bit-identical at step 0" predicts null) and the 92-step horizon is too short for `v+` to stabilize. Re-evaluate at ≥135M Phase-2 with 3-4k steps AND a re-tuned LR (the failure mode is LR-magnitude, not the mechanism itself; a 10× LR scale may be necessary to test AdaPNM properly). Closes the dual-momentum optimizer axis at 0.94M.
