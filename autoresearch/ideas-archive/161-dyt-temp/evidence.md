# Evidence — 161 dyt-temp

## Verdict: NULL (DRIFT, Δ=+0.0830)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060 sm_86)
- baseline: fresh N=3 ctrls mean=6.4320 ±0.04 (box 5b8a7fea8963, measured 2026-06-14T06:12:27Z)
- treatment val: 6.5150   Δ vs baseline: +0.0830
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ ctrl − 0.005; NULL |Δ| < 0.005; DRIFT > +0.005  → **DRIFT** (Δ=+0.0830 ≫ +0.005)
- box check: fresh ctrl mean 6.4320 vs leaderboard 6.4306 (Δ=+0.0014, within noise)
- raw: autoresearch/remote-results/2026-06-14-vast-tiny1m3m-4/results.json
- date: 2026-06-14

## Transfer note
The per-layer learnable τ_l (one shared `[n_layers]` parameter on the model, init `1/sqrt(d_k)`) is a hostile lever at 0.94M / 12L: the gradient on τ_l fights the canonical `1/sqrt(d_k)` prior the model already uses, and there's no scale-depth for the lever to find a useful different-per-layer schedule — it just drifts. Closed 155 (per-head τ_h, NULL inside band) and 161 (per-layer τ_l, this — DRIFT) jointly say: **attention temperature is over-fit to the canonical constant at this scale**, neither within-layer nor cross-layer learnable variants earn their keep. The two-axis closed result also dovetails with the closed per-channel-temp siblings (no related axis in flight). Survival to 135M is unlikely: the canonical default is a strong prior that the optimizer does not need to re-discover via a per-layer scalar when the FFN residual stream already absorbs cross-layer variance.
