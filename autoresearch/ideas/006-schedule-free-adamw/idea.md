---
id: 006-schedule-free-adamw
status: needs-codereview
round: 1
updated: 2026-06-09T10:19:18Z
---

# 006 — Schedule-Free AdamW

## Source
Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky — "The Road Less Scheduled" (arXiv:2405.15682, May 2024, v4 Oct 2024). Code: https://github.com/facebookresearch/schedule_free. Won the MLCommons 2024 AlgoPerf Algorithmic Efficiency Challenge (Self-Tuning track).

## Mechanism
Eliminates the LR schedule entirely. Replaces `torch.optim.AdamW` with a 3-point iterate (`x`, `z`, `y` = EMA of `z`) that the optimizer swaps between during eval vs training. Zero new hyperparameters over standard AdamW (the schedule constants in the LR schedule ARE the removed knobs). Drop-in replacement for the AdamW path. Implementation is ~50 LoC (the official PyTorch impl is small). Trains without warmup, without warmdown, without decay-to-zero — the iterate averaging handles the late-training stabilization.

## Pass / fail bar
- pass: tiny1m3m val ≤ 6.4206 (ctrl 6.4287, target Δ = −0.0081)
- fail: tiny1m3m val > 6.4287
- noise: |Δ| ≤ 0.005 — inconclusive at this scale
- expected Δ ≈ −0.005 to −0.02 (paper reports state-of-the-art across many tasks, but the gains are larger at scale; at tiny1m3m it may be in noise)

## Repo-fit concerns
- Must disable the current `schedule_type="warmup_decay_to_zero"` for tiny1m3m — schedule-free replaces it
- Muon path is unchanged (only AdamW swap)
- The `use_cautious_adamw` gate in `training/trainer.py:147-174` is orthogonal — can co-test on Schedule-Free AdamW if both win their individual A/Bs
