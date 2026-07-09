# Evidence — 154 Rebased Attention

## Verdict: WIN
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, 12GB)
- baseline: cached mean=6.4394 ±0.04 (box 5b8a7fea8963, 3 ctrls measured 2026-06-14T00:13:56Z)
- treatment val: 2.9628   Δ vs baseline: −3.4766
- train: 2.9611   val_acc: 0.4958
- bpb: n/a (pending harness)
- pass/fail bar (plan.md): PASS trt < cached_mean - 0.04; NULL inside band; DRIFT > +0.01 → **met**
- box check: cached mean 6.4394 vs leaderboard 6.4306 (within 0.009 of noise band, healthy)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log: 154-rebased-attn_52674.log)
- date: 2026-06-14

## Transfer note
Rebasing collapses K/V before softmax, so the gain is plausibly a real locality prior rather than a parameter-count trick. The margin is large enough that part of it may be a small-scale optimization shortcut, but the mechanism is structurally different from the closed attention-output levers, so it is worth carrying forward and remeasuring against a fresh baseline before trusting the size of the win at 135M.
