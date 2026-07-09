# Quickstart

Five minutes to a first submission using the bundled deterministic demo task.

You will use:

- goal: `G001-deterministic-demo`
- task: `T001`
- experiment: `goals/G001-deterministic-demo/tasks/T001/experiment.py`
- config: `goals/G001-deterministic-demo/tasks/T001/config.json`

## 1. Enter the repo root for token2science

```bash
cd token2science
```

## 2. Optional: run the demo experiment directly

This shows the exact number the task is built around.

```bash
(
  cd goals/G001-deterministic-demo/tasks/T001
  python experiment.py --config config.json
)
```

You should see a line like:

```text
RESULT metric=rmse value=1.0052614713
```

## 3. Claim the task

From `token2science/`, let the worker claim an open task issue.

```bash
python worker/worker.py claim
```

If `gh` is installed and authenticated, this self-assigns the first open `task:open` issue.

## 4. Submit the run

Run the local submit flow. Replace the handle with your GitHub username.

```bash
python worker/worker.py submit \
  --goal G001-deterministic-demo \
  --task T001 \
  --worker <your-gh-handle>
```

The worker will:

- run `python experiment.py --config config.json` in the task folder
- capture the `RESULT` line
- write `runs/T001/<your-run>/result.json`
- write `runs/T001/<your-run>/run.log`
- print the git and PR commands for the next step

## 5. Verify the submission

Use the run folder name printed by `submit`.

```bash
REPRODUCE=1 python verify/verify.py --run runs/T001/<your-run>
```

If the config hash is right and the experiment reproduces to the same number within tolerance, verify prints `VERDICT: accepted`.
