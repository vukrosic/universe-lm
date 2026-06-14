# Evidence — 157 Conv FFN

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, 12GB)
- baseline: cached mean=6.4394 ±0.04 (box 5b8a7fea8963, 3 ctrls measured 2026-06-14T00:13:56Z)
- treatment val: 6.4316   Δ vs baseline: −0.0078
- train: 6.4004   val_acc: 0.1429
- bpb: n/a (pending harness)
- pass/fail bar (plan.md): PASS trt < cached_mean - 0.04; NULL inside band; DRIFT > +0.01 → **not met** (Δ inside band)
- box check: cached mean 6.4394 vs leaderboard 6.4306 (within 0.009 of noise band, healthy)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log: 157-conv-ffn_52674.log)
- date: 2026-06-14

## Transfer note
A post-activation depthwise conv inside the FFN does not buy detectable value at 0.94M. The null is consistent with the attention path already absorbing the local-mixing signal at this tier; if the locality prior survives to 135M, it likely needs deeper or wider FFNs for the extra spatial mixing to compound.
