# Evidence — 142 LayerScale

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast-1.208.108.242:52674 (RTX 3060 sm_86, PyTorch 2.12.0+cu130)
- control val: 6.4409 / 6.4225   treatment val: 6.4397   Δ: -0.0012 / +0.0172
- bpb: n/a (pending harness)
- pass/fail bar (plan.md): trt val <= 6.4306 - 0.01 = 6.4206 with two-ctrl rule → not met
- box check: ctrl 6.4409 (+0.0103) and ctrl2 6.4225 (-0.0081) bracket leaderboard 6.4306 cleanly; ctrl-to-ctrl gap 0.0184 — well within the ~0.04 noise band; box healthy
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-14

## Transfer note
LayerScale's per-channel diagonal gain `g_ℓ ∈ ℝ^{d_model}` with init ε=1e-4 is *qualitatively* different from ReZero (130, scalar, init 0) and Sub-LN (017, sandwich-norm) — the per-channel selectivity is the genuinely new ingredient the paper proposes for deep ViTs. At 12L the lever's soft-warmup never had to fire (the model reaches the ctrl-pair variance floor before γ learns anything meaningful): final val 6.4397 sits inside the 0.0184 ctrl-pair gap, and train_loss 6.4554 is on the wrong side of ctrl 6.4242 (Δ +0.0312). Pattern matches the four closed depth-conditional levers at 12L — Sub-LN (017, +0.021 wrong-sign), DropPath (111, +0.018 wrong-sign), ReZero (130, +0.004 inside band), mHC (116, +0.066 wrong-sign) — adding a fifth null at d_model=64,n_layers=12. Per-channel diagonal gain still unproven for shallow LMs and the depth-sensitivity claim from the original ViT paper (biggest gain at depth ≥ 50) does not transfer to 12L. Re-evaluate at Phase-2 L=24+ where each layer has more compounding responsibility for the residual stream — the lever's gradient-suppression-on-γ-then-growth pattern may finally have room to express itself.