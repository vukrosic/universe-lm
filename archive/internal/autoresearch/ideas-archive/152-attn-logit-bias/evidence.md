# Evidence — 152 Per-Head Attention Logit Bias

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (5b8a7fea8963, RTX 3060, sm_86, driver 580.159.03)
- baseline: cached mean=6.4394 ±0.04 (box 5b8a7fea8963, measured 2026-06-14T00:13:56Z from 3 ctrls: 6.4366, 6.4287, 6.4528)
- treatment val: 6.4525  Δ vs baseline: +0.0131
- bpb: n/a (pending harness)
- pass/fail bar: Δ ≤ −0.005 (small/null band — predicted mathematical null)  → not met (Δ=+0.0131 is inside the ±0.04 cache band and above the bar)
- box check: cached baseline mean 6.4394 vs leaderboard 6.4306 — Δ=+0.0088 (within noise)
- raw: autoresearch/remote-results/2026-06-14-vast-tiny1m3m-2/results.json (log alongside)
- date: 2026-06-14

## Run details
- Tier: tiny1M3M (0.94M params · 3M tokens, seed 42)
- Box: vast-52674 RTX 3060, CUDA 13.0, PyTorch 2.12.0+cu130, Python 3.12.13
- Wall: 3m 55s, train loss 6.4270, val acc 0.1413
- Val progression: step 0 = 10.8125 (bit-identical to ctrl, as expected — `attn_logit_bias=0` ⇒ softmax unchanged); step 50 = 7.81; step 100 = 7.43; step 200 = 6.99; step 400 = 6.66; final = 6.4525.
- Run clean, no NaN, no timeout.

## Transfer note
A **mathematical null at tiny1m3m** for per-head attention logit bias. The lever is essentially a per-head QK bias that the optimizer can already learn implicitly through Q/K weight updates at 0.94M / 12 layers / 4 heads. At PaLM 2 / OLMo 2 (≥7B) source scale, where each head can specialize on a separate feature channel and the per-head QK-bias is a free parameter the head can pull on without fighting the Q/K gradient, the lever likely has a real axis to exploit. The null here at tiny1m3m is consistent with both "the axis is real but small at this tier" and "the axis is dominated by Q/K updates at this scale" — the two are statistically indistinguishable inside a single-seed A/B. No transfer risk flag change recommended: production-scale per-head bias is well-validated and orthogonal to this null. Closed for tiny1m3m.
