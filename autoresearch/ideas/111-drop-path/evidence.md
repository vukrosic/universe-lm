# Evidence — 111 DropPath / Stochastic Depth

## Verdict: DRIFT (FAIL)
- tier: tiny1m3m, seed 42, box: vast-52649 (RTX 3060, sm_86)
- ctrl val: 6.3887   trt val: 6.4422   Δ vs ctrl: +0.0535
- ctrl2 val: 6.3953  trt vs ctrl2 Δ: +0.0469
- ctrl-to-ctrl gap: 0.0066
- bpb: n/a (pending harness — measured val loss only at this tier)
- pass/fail bar: ≤ 6.3837 (ctrl − 0.005) → not met. NULL band |Δ| < 0.005
  exceeded by ~10× → outside the null band, well past the DRIFT threshold
  (> +0.005). The trt is worse than *both* controls by ~0.05, an order of
  magnitude beyond the in-session ctrl-to-ctrl gap and ~10× the
  measured ~0.04 box variance on this tier.
- box check: ctrl 6.3887, ctrl2 6.3953 — within the ~0.04 in-bracket noise;
  no DRIFT, the box is healthy.
- raw: remote-results/2026-06-13-vast-tiny1m3m/111-drop-path/{results.json,
  ctrl_52649.log, 111-drop-path_52649.log, ctrl2_52649.log}
- date: 2026-06-13

## Transfer note
DropPath at `drop_path_max=0.1` over 12 blocks for ~3M tokens of pretraining
hurts at this scale by a clear margin (val +0.05 vs both ctrls). The
mechanism is a stochastic regularizer that works on 100L+ ResNets and
12-24L ViT/ConvNeXt in vision; on a 12L, 0.94M-parameter causal LM at
~3M training tokens (well below the depth and token-count where the
original paper's effect stabilizes), the regularizer appears to remove
useful signal more often than it forces redundant representations.
Two viable explanations:
1. **Depth is too shallow.** Stochastic depth at p_max=0.1 has the last
   block being skipped ~10% of steps and earlier blocks proportionally
   less. At 12 layers each block is doing load-bearing work the others
   can't fully compensate for, so the random skip directly costs loss
   the model has to re-learn from a weaker state.
2. **Token budget is too short.** The original paper trained for
   orders of magnitude more steps; the regularizer's "force every
   block to be droppable" benefit may need longer to amortize the
   short-term cost of a noisier gradient signal.

A clean DRIFT here is informative: drop-path is *not* a free regularizer
at this scale. The lever stays closed for tiny1m3m / 135M-class model
recipes. If a future hypothesis is "depth-scaling" (n_layers 24+) at the
same parameter budget, drop-path is worth re-testing as a depth
companion — but at 12L it costs more than it returns.