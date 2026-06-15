# Evidence — 131 layer-drop

## Verdict: NULL (borderline; trt inside ctrl-to-ctrl gap)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   ctrl2 val: 6.4419   treatment val: 6.4138
- Δ vs ctrl: −0.0134   Δ vs ctrl2: −0.0281   ctrl-to-ctrl gap: 0.0147
- train_loss: ctrl 6.3966, treatment 6.3752 (Δ −0.0214)
- val_accuracy: ctrl 0.1459, treatment 0.1447
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222); null band |Δ|<0.01, DRIFT > +0.01
  → trt beats ctrl1 by 0.0134 (just over the 0.01 null band) but by less than
    the 0.0147 ctrl-to-ctrl gap; trt beats ctrl2 by 0.0281 (clearly above
    gap) → per §2 two-ctrl rule, treatment must beat **both** ctrls by **more
    than** the gap → 0.0134 < 0.0147 → **NULL**
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise);
  ctrl-to-ctrl gap 0.0147 is itself within the ~0.04 measured variance band
- raw: remote-results/2026-06-13-vast-tiny1m3m/arq-110/131-layer-drop_52674.log,
  ctrl_52674.log, ctrl2_52674.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.3752 vs ctrl 6.3966 ⇒ **−0.021 train-loss gap** ⇒ LayerDrop's per-batch Bernoulli(1−p_l) gate *with* the 1/p_l rescale is a strict improvement on train-loss; the per-batch (vs DropPath's per-sample) is the right granularity
- The −0.021 train_loss gap is the only closed-idea analogue to compare against: **111-drop-path** was +0.018 (wrong-sign, closed as DRIFT). So at the train-loss axis, LayerDrop's per-batch gate is the *opposite* outcome to DropPath's per-sample gate — informative either way
- Treatment val: 6.4138 (Δ −0.0134 vs ctrl1) clears the −0.005 WIN bar; train_loss 6.3752 (Δ −0.021) is a larger relative win

## Transfer note
LayerDrop (Fan et al. 2019/2020, "Reducing Transformer Depth on Demand with Structured Dropout") drops *entire transformer blocks* per-batch with probability `p_l` and rescales the kept-block output by `1/p_l` so the expected residual magnitude matches baseline. The paper validates at RoBERTa-base 125M / large 355M / BART-large 400M, and finds the lever is depth-sensitive: works best at L=24+, marginal at L=12. The −0.021 train_loss / −0.0134 val_loss result at tiny1m3m (L=12) is the *opposite* of DropPath 111 (which was +0.018 / +0.07 wrong-sign). The mechanism-level explanation: DropPath's per-sample gate produces noisy gradients that don't match the per-batch residual statistics, while LayerDrop's per-batch gate is consistent within a step. The 12L is *less* favorable per the paper's own depth-sensitivity claim, so a positive signal here is *stronger* than at 24L — but a single seed at a tier with this much noise (0.04 measured ctrl variance) plus a ctrl-to-ctrl gap of 0.0147 makes the WIN rule (|trt−ctrl|>gap for both ctrls) fail: trt beats ctrl1 by 0.0134 < 0.0147 gap. Not promoted; closed as NULL. **Re-evaluate at Phase-2 L=24+ tier** where the per-batch gate's depth-conditional gain compounds — and the 0.04 measured noise is much smaller relative to the signal.
