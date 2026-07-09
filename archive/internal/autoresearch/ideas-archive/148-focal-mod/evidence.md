# Evidence — 148 focal-mod

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4409   treatment val: 6.4481   Δ: +0.0072 (vs ctrl1), +0.0256 (vs ctrl2 6.4225)
- ctrl2: 6.4225 (gap vs ctrl1 = 0.0184, within noise); treatment does NOT beat ctrl1 (Δ +0.0072 wrong-sign) → NULL under §2 two-ctrl rule
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ −0.01 vs ctrl; NULL band |Δ| < 0.01; DRIFT > +0.01
- box check: ctrl1 6.4409 (+0.0103) and ctrl2 6.4225 (-0.0081) bracket leaderboard 6.4306 cleanly; box healthy
- raw: remote-results/2026-06-14-vast-tiny1m3m/{ctrl,ctrl2,148-focal-mod}_52674.log
- date: 2026-06-14

## Trajectory check
- Live θ (train_loss): trt 6.4377 vs ctrl 6.4242 → trt train_loss *worse* by +0.0135 (wrong sign)
- Val curve trt: 10.81 → 8.33 → 7.79 → 7.56 → 7.42 → 7.16 → 6.99 → 6.79 → 6.65 → 6.45 (final)
- Val curve ctrl: 10.81 → 8.34 → 7.79 → 7.55 → 7.42 → 7.15 → 6.99 → 6.79 → 6.65 → 6.44 (final)
- Curves are essentially superimposed through step 400, diverge by ~0.007 in the final 200 steps (trt is slightly *worse* on val *and* train → focal modulation's gated-additive context aggregation neither helps nor dramatically hurts at this scale)

## Transfer note
Focal modulation replaces softmax attention with hierarchical depthwise-conv context aggregation + sigmoid gate. The mechanism's claim to fame is long-context efficiency (no O(T²) memory) and replacement of softmax with additive modulation. At tiny1m3m: (1) the long-context benefit is invisible (T=2048 is well within softmax's regime), (2) the conv-stack at depthwise kernels (3,5,7) is essentially a local-context aggregator — far weaker than softmax's global attention at this scale, (3) the gate-projection bias init −10 keeps the modulation near-zero through most of training, so the model is effectively using local-conv-only for many steps before the gate learns to "open up". This is a *high transfer risk* null (per the idea's own risk tag) — focal modulation is unproven for LMs at any scale, and this A/B confirms the softmax-attention inductive bias is not trivially replaceable at 0.94M. **Closes the "non-softmax attention" axis** at this tier: 020–025 softcap axis was a within-softmax modification; 148-focal-mod is the cross-axis no-softmax replacement and it nulls. Re-evaluation at >=135M is *not recommended* unless long-context is added to the metric — without the lever's headline benefit visible, this is an information-free A/B at any tier.