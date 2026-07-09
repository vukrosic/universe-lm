# Contributing to token2science

token2science is a bring-your-own-compute project.

- You donate AI tokens by running an agent on your own machine.
- Your agent spends those tokens to write code, run the experiment, and prepare the submission.
- The science is public. The repo, the tasks, the run artifacts, and the verification result are all visible.

## How contribution works

1. Propose a goal.
   - Open a goal issue that states the question and the measurable bar.
   - A maintainer or planner turns that goal into one or more tasks.
2. Claim a task.
   - Use `python worker/worker.py claim`.
   - The worker self-assigns an open task issue.
3. Do the work.
   - Your agent edits the task folder and runs the experiment locally.
   - Keep the output deterministic when the task requires it.
4. Submit the run.
   - Use `python worker/worker.py submit`.
   - The worker runs the experiment, records the result, and writes the run artifact.
5. Reproduce in CI.
   - GitHub Actions re-runs the experiment from the submitted config.
   - The submission is accepted only if the reproduced value matches the submitted value within tolerance.
6. Confirm independently.
   - A result becomes confirmed only after `K` independent workers reproduce it.

## Commands

Run these from the `token2science/` directory.

Claim the next open task:

```bash
python worker/worker.py claim
```

Submit a run:

```bash
python worker/worker.py submit \
  --goal G001-deterministic-demo \
  --task T001 \
  --worker <your-gh-handle>
```

The worker:

- runs the task's `experiment.py`
- reads the last `RESULT metric=<name> value=<float>` line
- hashes the config file
- writes `runs/<task>/<run>/result.json`
- writes `runs/<task>/<run>/run.log`
- prints the git and PR commands to run next

## result.json contract

`result.json` is the submission record. CI reads it first.

Required fields:

- `task_id`
- `goal_id`
- `worker`
- `metric`
- `value`
- `lower_is_better`
- `seed`
- `config_path`
- `config_hash`
- `command`

Common optional field:

- `tolerance` - max absolute difference allowed when CI re-runs the experiment

Field rules:

- `task_id` and `goal_id` must point at the exact task and goal being submitted.
- `worker` must be the contributor's GitHub handle.
- `metric` must match the goal metric.
- `value` must be the numeric result printed by the experiment.
- `lower_is_better` must match the goal.
- `seed` must come from the submitted config.
- `config_path` is relative to the task folder.
- `config_hash` must be `sha256:<hex>` for the exact config bytes that were run.
- `command` must be the exact command CI uses to reproduce the run, executed in the task folder.

## Acceptance rule

Hard rule: nothing is accepted unless CI re-runs the experiment and gets the same number.

- If the config hash does not match, the run is rejected.
- If CI cannot reproduce the submitted number within tolerance, the run is rejected.
- A submission that only looks correct in `result.json` is not enough.
- The reproduced value is the gate.

## Mental model

- Goal = the question
- Task = one claimable unit of work
- Run = one evidence artifact
- CI = the neutral machine that checks the claim
- K independent reproductions = confirmation
