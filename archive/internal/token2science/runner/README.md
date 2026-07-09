# Compute Donor Runner

`runner.py` is the compute-donor loop for `token2science`.

It is not the same as `worker/worker.py`:

- `worker.py` submits one chosen task.
- `runner.py` keeps scanning open goals and spends compute on the task that
  has the fewest distinct reproducing workers so far.

That makes it the "confirmation filler" role: it helps close K-replication
gaps instead of starting new work on tasks that are already well supported.

## What it looks at

The runner only considers tasks that satisfy all of these:

- the goal has `goal.json` with `"status": "open"`
- the task folder contains `experiment.py`
- the task folder contains `config.json`
- the worker has not already run that task

For each runnable task, it counts distinct workers who already submitted a
reproducing run for the same `config_hash` and a value within `1e-9`.

## How to run

Run it from `token2science/`:

```bash
python runner/runner.py --worker <name> [--rounds 1] [--k 2]
```

Examples:

```bash
python runner/runner.py --worker alice
python runner/runner.py --worker alice --rounds 5
python runner/runner.py --worker alice --rounds 10 --k 3
```

Each successful round:

- runs `python experiment.py --config config.json` in the selected task folder
- parses the last `RESULT metric=<name> value=<float>` line
- writes `runs/<task>/<worker>-<timestamp>/result.json`
- writes `runs/<task>/<worker>-<timestamp>/run.log`

The `result.json` schema matches `worker.py`:

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
- `tolerance`

The default tolerance is `1e-9`.

## Stop condition

The loop stops when either of these is true:

- it reaches the requested number of rounds
- nothing runnable still needs this worker to help fill a confirmation gap

This role is meant for donated compute, so it does only the narrow reproduce
job and nothing else.
