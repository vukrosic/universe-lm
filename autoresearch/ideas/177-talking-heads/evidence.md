# Evidence — 177-talking-heads

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52646
- baseline: mean=6.3988 ±0.04 (box-keyed cache; 3 ctrls: 6.4112/6.3934/6.3919)
- treatment val: 7.3497   Δ vs baseline: +0.9509
- bpb: n/a (pending harness)
- pass/fail bar: noise-band rule — WIN iff val < mean − band (6.3588); 7.3497 is 24× above the 0.04 noise band → massive DRIFT
- box check: ctrl mean 6.3988 vs leaderboard 6.3988 — within noise
- raw: remote-results/2026-06-15-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-15
- judged-by: queue-daemon.sh (deterministic)

## Transfer note
(auto) deferred — see idea.md `## Scale evidence`. Written by the analyzer pass, not the daemon. Train loss 7.2287 ≈ val loss 7.3497 (both ~0.95 worse than baseline) — talking-heads attention mixing on H=4 heads at d_k=16 has so few signal channels that the learned H×H mixing matrix dominates the softmax distribution and prevents the model from learning useful attention patterns. Mechanism likely transfers worse at larger H but plan bar was met, so result stands.
