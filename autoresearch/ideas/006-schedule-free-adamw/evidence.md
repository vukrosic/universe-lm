# Evidence — 006 schedule-free-adamw

## Verdict: NULL (treatment worse)
- tier: tiny1m3m, seed 42, box: vast-34386 (220.82.52.202:34386, RTX 3060)
- control val: 6.5953   treatment val: 6.8056   Δ: **+0.2103** (worse)
- ctrl2: 6.6091 (two-ctrl bracket; ctrl-to-ctrl gap 0.0138)
- pass/fail bar: PASS ≤ −0.005 vs ctrl → **NOT met** — treatment is +0.21 worse,
  wrong sign, far outside the ~0.04 variance band. Schedule-Free AdamW does not
  help (clearly hurts) at this tier.
- code: optimizer verified bit-for-bit canonical SF-AdamW in code-review
  (no first-moment EMA; momentum emulated via y/z interpolation). The double-
  momentum bug from the first codereview was fixed before this run.
- ⚠️ box check: ctrl 6.5953 vs today's leaderboard ctrl ~6.39 → **DRIFT +0.19**
  (well beyond 0.04 noise). Both same-session ctrls agree (6.5953 / 6.6091), so
  the within-session A/B is valid, but absolute numbers are NOT comparable to
  prior days. Suspected cause: the runner scp'd the whole local working-tree
  `training/trainer.py` + `configs/llm_config.py` (carrying every in-flight idea's
  uncommitted changes) to the box rather than a targeted patch, shifting the
  baseline. Needs follow-up before trusting cross-day comparisons.
- raw: remote-results/2026-06-09-vast-tiny1m3m/logs/006-schedule-free-adamw.log
- date: 2026-06-09
