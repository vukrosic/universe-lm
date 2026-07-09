# Evidence — 339-v-mix-conv

## Verdict: LEAK
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52646
- baseline: mean=6.172 ±0.02 (box-keyed cache)
- treatment val: 0.4224   Δ vs baseline: 0
- bpb: n/a (pending harness)
- pass/fail bar: noise-band rule — WIN iff val < mean − band (see plan.md for the paper-level claim)
- raw: remote-results/2026-06-17-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-17
- judged-by: queue-daemon.sh (deterministic)

## Transfer note
(auto) deferred — see idea.md `## Scale evidence`. Written by the analyzer pass, not the daemon.
