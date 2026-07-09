# Evidence — 192-topk-attn

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52646
- baseline: mean=6.2539 ±0.04 (box-keyed cache)
- treatment val: 6.3800   Δ vs baseline: 0.1261
- bpb: n/a (pending harness)
- pass/fail bar: noise-band rule — WIN iff val < mean − band (see plan.md for the paper-level claim)
- raw: /Users/vukrosic/my-life/llm-research-kit-scaling/remote-results/2026-06-15-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-15
- judged-by: queue-daemon.sh (deterministic)

## Transfer note
(auto) deferred — see idea.md `## Scale evidence`. Written by the analyzer pass, not the daemon.
