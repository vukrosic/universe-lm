# Evidence — 176-v-pre-av-norm

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52646
- baseline: mean=6.3988 ±0.04 (box-keyed cache; 3 ctrls: 6.4112/6.3934/6.3919)
- treatment val: 6.4291   Δ vs baseline: +0.0303
- bpb: n/a (pending harness)
- pass/fail bar: noise-band rule — WIN iff val < mean − band (6.3588); 6.4291 is inside 6.3988±0.04 → NULL
- box check: ctrl mean 6.3988 vs leaderboard 6.3988 — within noise
- raw: remote-results/2026-06-15-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-15
- judged-by: queue-daemon.sh (deterministic)

## Transfer note
(auto) deferred — see idea.md `## Scale evidence`. Written by the analyzer pass, not the daemon.
