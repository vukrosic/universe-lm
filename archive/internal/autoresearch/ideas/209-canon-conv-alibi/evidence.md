# Evidence — 209-canon-conv-alibi

## Verdict: NULL  (corrected from a false WIN)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52646
- champion bar: alibi mean=6.2403 ±0.04
- treatment val: 6.2519   Δ vs champion: +0.0116 (inside variance → NULL)
- bpb: n/a (pending harness)
- pass/fail bar: noise-band rule — WIN iff val < champion_val − band; 6.2519 does not clear 6.2303
- raw: remote-results/2026-06-15-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-15
- judged-by: queue-daemon.sh (deterministic)

## Correction note
The daemon first logged this as a WIN against a per-box BASE control mean (6.3988):
on a fresh box_key it re-measured a baseline whose controls ran `Tiny1M3MConfig`
(base), not the champion stub, so 6.2519 < 6.3988 read as a win. Judged against the
actual champion (alibi, 6.2403) it is a NULL. Champion reverted to alibi; daemon
fixed so `finalize_one` pins the judging bar to the champion's val.

## Transfer note
(auto) deferred — see idea.md `## Scale evidence`. Written by the analyzer pass, not the daemon.
