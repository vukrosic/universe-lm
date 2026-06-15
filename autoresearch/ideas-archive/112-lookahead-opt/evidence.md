# Evidence — 112 Lookahead Optimizer Wrapper

## Verdict: DRIFT (FAIL)
- tier: tiny1m3m, seed 42, box: vast-52649 (RTX 3060, sm_86)
- ctrl val: 6.4047   trt val: 7.0381   Δ vs ctrl: +0.6334
- ctrl2 val: 6.4088  trt vs ctrl2 Δ: +0.6293
- ctrl-to-ctrl gap: 0.0041
- bpb: n/a (pending harness — measured val loss only at this tier)
- pass/fail bar: ≤ 6.3997 (ctrl − 0.005) → not met. NULL band |Δ| < 0.005
  exceeded by ~127× → far outside the null band, well past the DRIFT
  threshold (> +0.005). The trt is worse than *both* controls by ~0.63,
  more than **150× the in-bracket ctrl gap** of 0.0041, and ~16× the
  measured ~0.04 box variance on this tier. This is a clean, large
  DRIFT, not a marginal null.
- box check: ctrl 6.4047, ctrl2 6.4088 — within the ~0.04 in-bracket
  noise; no DRIFT, the box is healthy.
- raw: remote-results/2026-06-13-vast-tiny1m3m/112-lookahead-opt/{results.json,
  ctrl.log, 112-lookahead-opt.log, ctrl2.log}
- date: 2026-06-13

## Val-loss trajectory (treatment)
| step | trt val_loss | (ctrl trajectory for reference) |
|------|--------------|----------------------------------|
| 0    | 10.8125      | (ctrl step 0: 10.8125)          |
| 25   | 9.6869       | —                                |
| 50   | 7.8206       | —                                |
| 75   | 7.7138       | —                                |
| 100  | 7.5947       | —                                |
| 150  | 7.4356       | —                                |
| 200  | 7.2684       | —                                |
| 300  | 7.1425       | (ctrl step 300: 6.7309)         |
| 400  | 7.0828       | (ctrl step 400: 6.6016)         |
| final| **7.0381**   | (ctrl final: 6.4047)            |

The treatment is worse than ctrl **from step 50 onward**, with the gap
widening through training (Δ at step 100: ~+1.0, Δ final: +0.63). This
is not a late-training collapse — it's a systematic under-convergence
across the whole run.

## Transfer note
Lookahead at `k=5, α=0.5` (paper defaults) over a 12L, 0.94M-parameter
causal LM with the inner Muon+AdamW stack and 732 inner steps, hurts by
a clear margin (val +0.63 vs both ctrls). The mechanism is a
trajectory-level wrapper: every 5 inner optimizer steps, the slow
weights pull halfway toward the fast weights and the fast weights reset
to slow (the inner optimizer's state is cleared to avoid stale
momentum carrying across the slow reset). With Lookahead active, the
slow trajectory advances at half the per-step speed of the fast
trajectory, so in 732 inner steps the slow weights see only ~146
effective Lookahead-updates — far fewer than the inner optimizer's
~732 effective updates. The combined effect is that the slow
trajectory lags behind what the baseline trajectory would have done
over the same step count, and the inner optimizer's state-clear at
every outer step discards the momentum that the baseline would have
accumulated.

Two plausible explanations, both consistent with a trajectory-averaging
wrapper that costs more than it returns at this scale:

1. **The "k=5 inner, 1 outer" cadence is too coarse for 732 steps.**
   Paper-scale training (ResNet-50 ImageNet, ~90 epochs ≈ tens of
   thousands of steps) has many outer-step cycles, so the slow
   trajectory has time to amortize the trajectory-averaging cost. At
   732 inner steps, only ~146 outer-step syncs happen — the slow
   weights are pulled back 146 times toward a fast trajectory that
   is, in turn, partly reset by the same outer step. The result is
   that the slow weights effectively see only ~146 effective updates
   worth of progress, while the baseline (no Lookahead) sees the full
   732. Even if the slow trajectory is *smoother*, it is *shorter*,
   and the model hasn't trained long enough for the smoothing to
   amortize.

2. **State-clearing the inner optimizers at every outer step discards
   information that the baseline relies on.** Both Muon
   (`momentum_buffer`) and AdamW (`exp_avg`, `exp_avg_sq`) are
   momentum-based — at the 0.95/0.9 β settings used here, the
   momentum buffers carry significant state across the last ~10-20
   inner steps. Wiping them every 5 steps means the inner optimizers
   have to rebuild their momentum estimates from scratch at every
   outer sync. This is exactly the kind of "reset" cost that the
   paper's authors note is necessary to avoid overshoot but which
   shows up as a clean loss increase when the step count is too
   short for the smoother trajectory to catch up.

A clean DRIFT here is informative: **trajectory averaging at k=5 with
this base optimizer stack costs more than it returns at tiny1m3m
(0.94M params, 12L, 732 inner steps)**. The mechanism may still help
at a longer horizon (paper-scale: 90 epochs ≈ 10⁴+ steps) or with a
larger k (e.g. k=10-20, fewer outer-step cycles), but those are
different experimental designs that need their own A/B brackets. The
lever stays closed for tiny1m3m / 135M-class model recipes.
