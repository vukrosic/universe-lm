# Evidence — 130 rezero

## Verdict: NULL (inside null band, wrong sign)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.4313   Δ: +0.0041
- ctrl2: pending (queued last in tmux); verdict is decided regardless — Δ is inside the |Δ|<0.01 null band and is wrong-sign
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), NULL band |Δ|<0.01, DRIFT > +0.005 → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/130-rezero.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.4063 vs ctrl 6.3966 ⇒ +0.010 train-loss gap ⇒ ReZero's α_l ≈ 0 init produces an *identity function* at step 0; the model recovers as α_l grows, but the 92-step horizon doesn't give it enough ramp time
- The +0.0041 final val gap is *inside* the null band — ReZero and ctrl are statistically indistinguishable at tiny1m3m

## Transfer note
ReZero (Bachlechner et al. 2020, "ReZero is All You Need: Fast Convergence at Large Depth") introduces a learnable per-residual scalar `α_l` init at 0, so the block becomes the identity at step 0: `x ← x + α_l · f_l(x) = x + 0 = x`. The intuition: identity init lets gradients flow through deep residual stacks without exploding, so very deep networks (≥100L paper) can train from scratch. The +0.0041 null result at tiny1m3m (12L) says: at this depth the *baseline* residual init is already well-conditioned — there is no exploding-gradient problem for 12L, so the α_l=0 init buys nothing. The lever's published wins are at L=100+ (the paper's 10B-param, 128-layer model). At 12L the gain is structural zero. Pattern matches SubLN-sandwich 017 (also a depth-conditional lever that nulls at 6L). Re-evaluate at Phase-2 (135M, L=24+) where depth is the binding constraint, but at tiny1m3m the lever is closed: NULL — no detectable effect.
