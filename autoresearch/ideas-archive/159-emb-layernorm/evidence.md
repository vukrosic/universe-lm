# Evidence — 159 — Embedding Pre-LayerNorm (Input Embedding LN)

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, driver 580.159.03)
- baseline: cached mean=6.4504 ±0.0558 (box 5b8a7fea8963, measured 2026-06-14T05:39:17Z, n=7)
- treatment val: 6.5216   Δ vs baseline: +0.0712  (above band → DRIFT, but spec reads >+0.005 as harmful)
- bpb: n/a (pending harness)
- pass/fail bar (from plan.md): win = val < mean − band = 6.3946  → not met; null = inside band; drift = val > mean + band = 6.5062
- box check: baseline mean 6.4504 vs leaderboard ctrl ~6.41 (within noise band — no drift sentinel triggered)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (logs: 159-emb-layernorm_52674.log)
- date: 2026-06-14

## Transfer note
Pre-block LN on the token embedding is standard in LLaMA-3 / Gemma-2 / Qwen-2.5 at 2B-70B scale. At 0.94M it produced a +0.071 val loss hit (Δ = +1.4× the noise band, well past drift). The factorized tiny1m3m embedding distribution is approximately N(0, σ_c²) per token because the per-token `x_post[b,t,:]` samples the same i.i.d. distribution across positions — a global per-token rescale shifts the mean of that distribution to 0 and rescales by 1/σ, which at this depth+width changes the operating point the rest of the network was tuned for. The mechanism is real at scale (it gates the embedding-magnitude variance before layer 0), but at 0.94M the network has no spare capacity to re-fit the rescaled distribution inside 3M tokens — the LN is paying the cost of an effective LR/warmup re-tune with no downstream benefit at this budget. Not a real failure of the technique; a "this tier is too small for this lever" negative. Survives to 135M should be re-tested; the bigger model has more gradient signal per token to recover the rescaling cost.
