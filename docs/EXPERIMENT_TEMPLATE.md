# 25M Experiment Template

Copy this into a new GitHub issue for every 25M research experiment. Fill the `<...>` slots, delete the italic guidance notes. One mechanism per issue. Keep it to a number — no pre-writeups (see [RESEARCH_IDEAS.md](RESEARCH_IDEAS.md) for the test-first posture).

Title format: `[research] <Name> — <one-line what it is>`

**If the issue is AI-generated** (e.g. derived from the parameter-golf record), put this banner at the very top:

> ⚠️ **AI-generated from the parameter-golf record.** This is a starting point, not a spec. The human's job is to think it through first.

---

**Status: implemented on branch `<exp/branch-name>`. This issue = run the 25M A/B and report the number.**

## What it is
<One-sentence mechanism. What changes in the model/optimizer/data, concretely. e.g. "A learnable per-head scalar on attention logits before softmax.">

## Why it should help (theory)
<2-4 sentences. The mechanism, not hand-waving. What does it add that the baseline can't do? End with a one-line summary.>

**Prior art (grounding, not novelty):** <where this has shown up before — nGPT, param-golf, a paper, etc. 1-2 lines. If genuinely novel, say so.>

## Transferability to 135M (prediction)
*The whole point: only run things that should still matter at 135M. Rate each bullet's confidence (high / med / low).*

- **Sign transfers (<conf>):** <why the direction (helps/neutral/hurts) should hold at 135M. Usually: is the mechanism scale-invariant?>
- **Magnitude at scale (<conf>):** <will the effect grow, shrink, or hold as params increase? Why?>
- **Cost:** <added params / FLOPs / wall-time. Is it ~free or does it tax the budget?>
- **Main risk:** <the thing most likely to break the 25M→135M transfer — schedule, depth, interaction with existing components.>

## Setup (do this once, on the GPU)
**Step 0 — data (only if not already downloaded).** If `processed_data/pretrain_1B` (or any `processed_data/pretrain_mix_*`) does not exist:
```bash
python data/download_hf_data.py
```
**Step 1 — branch.**
```bash
git checkout <exp/branch-name>
```

## Run it (the task)
The mechanism is baked into the model/config on this branch. Run the 25M preset:
```bash
python train_llm.py --config 25m --seed 42
```
Seed 42, 507M tokens, ~25M params, bf16 — all fixed in the `25m` preset. No edits.

*Run only the variant. Compare against the standing **leaderboard baseline** — never re-run a baseline arm.*

*Run in `tmux` so the job survives disconnect. You can hand this whole issue to your AI; it should run it unattended.*

## Report in this issue
- final val_loss (from the run's final eval print / log), and the leaderboard baseline you're comparing against.
- <any mechanism-specific sanity signal — e.g. "did the learned gains move off init 1.0?">
- One line: **your GPU, wall-time, git commit hash.** (Wall-time is metadata only.)

## Decision rule
- `leaderboard_baseline - <variant> ≥ 0.01` val_loss → say it should be promoted to the 135M preset run in the issue comments.
- Otherwise → null result. Record it in [RESEARCH_IDEAS.md](RESEARCH_IDEAS.md) and move on.

## Out of scope
- Hyperparameter tuning (init values, LR, schedule). The A/B must isolate the *mechanism*
- Running 135M before the 25M gate passes.

## If it crashes, NaNs, or diverges
Either **fix it** (note changes in your PR) or **just report it here** (paste the error) and stop.