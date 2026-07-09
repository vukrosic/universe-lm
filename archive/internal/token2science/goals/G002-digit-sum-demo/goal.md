# G002 - Deterministic-score smoke goal

**Question:** does the goal -> task -> run -> verify loop work end to end on a
machine with no GPU and no API tokens, while using a higher-is-better metric?

**Metric:** `score` (higher is better)
**Pass-bar:** `score > 4.53`
**Baseline:** the deterministic toy experiment in `tasks/T101/experiment.py`.

This goal exists to exercise the plumbing. A real goal swaps `experiment.py` for
an actual training/eval script and sets a meaningful bar; everything else (claim,
submit, hash check, reproduce, confirm) stays identical.

## Tasks
- `T101` - run the toy experiment at `seed=42, n=2000` and submit the score.
