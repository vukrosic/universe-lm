# Evidence — 146 sparse-ffn

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4409   treatment val: 6.4466   Δ: +0.0057 (vs ctrl1), +0.0241 (vs ctrl2 6.4225)
- ctrl2: 6.4225 (gap vs ctrl1 = 0.0184, within noise); treatment does NOT beat ctrl1 (Δ +0.0057 wrong-sign) → NULL under §2 two-ctrl rule
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ −0.01 vs ctrl; NULL band |Δ| < 0.01; DRIFT > +0.01
- box check: ctrl1 6.4409 (+0.0103) and ctrl2 6.4225 (-0.0081) bracket leaderboard 6.4306 cleanly; box healthy
- raw: remote-results/2026-06-14-vast-tiny1m3m/{ctrl,ctrl2,146-sparse-ffn}_52674.log
- date: 2026-06-14

## Trajectory check
- Live θ (train_loss): trt 6.4033 vs ctrl 6.4242 → trt train_loss is *better* by −0.0209 (right sign, real signal that the optimizer saw something useful)
- Val curve trt: 10.81 → 8.33 → 7.79 → 7.56 → 7.41 → 7.16 → 6.99 → 6.79 → 6.65 → 6.45 (final)
- Val curve ctrl: 10.81 → 8.34 → 7.79 → 7.55 → 7.42 → 7.15 → 6.99 → 6.79 → 6.65 → 6.44 (final)
- Curves are essentially superimposed through step 400, diverge by ~0.005 in the final 200 steps (trt is slightly *worse* on val despite better train_loss → mild overfit from the 4× FFN capacity not translating to held-out)

## Transfer note
Switch FFN is the simplest MoE variant — top-1 hard routing with `argmax`. At 0.94M the 4× FFN capacity injection (e.g. d_ff×4) is *real* param budget the model has to use, but the routing signal at 92 update steps / 3M tokens is too sparse for any expert to specialize meaningfully — `train_loss` improved (capacity helps in-distribution) but `val_loss` did not (overfit on the routing distribution). The MoE axis was already nulled at this tier by 117-soft-moe and 118-MoD (both closed 2026-06-13 with wrong-sign +0.14/+0.10), so this third null is consistent and *conclusive* — the MoE mechanism does not fire at 0.94M regardless of routing style. Re-evaluation at >=135M Phase-2 where (a) the routing signal has 30× more update steps to specialize experts and (b) the FFN is the binding capacity constraint (not routing overhead) is the right next stop, but should not be scheduled until Phase-2 is the active tier.