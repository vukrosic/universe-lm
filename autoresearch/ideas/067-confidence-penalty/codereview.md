# Code review log — 067 confidence-penalty

## r1 — 2026-06-11 — verdict: accept
- The lever is already pre-wired in `configs/output_head_ablations.py` and `training/trainer.py`, so the code path is present and the plan only needs to schedule it.
- The plan keeps the aux term train-only and the evaluation path unchanged, which matches the spec and preserves leaderboard comparability.
