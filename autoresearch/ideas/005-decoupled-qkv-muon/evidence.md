# Evidence — 005 decoupled-qkv-muon

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast 220.82.52.202:34386 (RTX 3060, sm_86)
- treatment val: 6.3909 (r1) — n=1
- control bracket: ctrl=6.3875, ctrl2=6.4050 (gap 0.0175)
- Δ vs ctrl: +0.0034 (treatment is marginally *worse* than ctrl)
- Δ vs ctrl2: -0.0141 (treatment is better than ctrl2, but by < gap)
- pass/fail bar (idea.md): pass ≤ 6.4206 (target Δ = -0.0081 vs leaderboard ctrl).
  Bar is met in absolute terms (6.3909 < 6.4206) but the two-ctrl rule
  requires beating *both* ctrls by more than the gap (0.0175) → not a WIN.
- two-ctrl rule: 6.3909 sits between ctrl (6.3875) and ctrl2 (6.4050) → NULL
  (inside variance). Treatment does not beat *both* ctrls.
- box check: ctrl 6.3875 vs leaderboard 6.4287 = -0.0413 (within 0.04 noise band)
- raw: remote-results/2026-06-09-vast-tiny1m3m/arq-r1/{005-decoupled-qkv.log,ctrl.log,ctrl2.log}
- date: 2026-06-09

## Caveat — code-loop audit
The 005 folder has no `plan.md`, `review.md`, or `codereview.md`. The
log.jsonl shows a single entry — the runner claimed it from `needs-run` at
2026-06-09T06:05:01Z. This means the code loop (need → plan → codereview)
was *skipped* for 005 (likely a backfill of the status field without the
loop artifacts). The verdict above is data-driven and stands regardless of
the missing docs, but the absence of plan.md/review.md/codereview.md is a
pre-existing pipeline issue. The 005 idea ships as a NULL by the two-ctrl
rule; the missing artifacts are a separate concern for the code-implementer
/ pipeline audit.
