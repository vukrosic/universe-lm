# Roles

`token2science` splits one research task into three contributor roles so work can move without one person doing everything.

## 1. AUTHOR

- Designs the experiment and writes `experiment.py`.
- Claims the task exclusively with `claim.py`.
- Owns authoring while the lease is live.
- Produces a runnable task from a proposed one.

The claim lease is for authoring only. It prevents two people from editing the same task at once, but it does not reserve compute.

## 2. COMPUTE / RUNNER

- Pulls runnable tasks via `runner/runner.py`.
- Executes the experiment and writes the submission artifact.
- Does not need an exclusive lease.
- Can be replicated by `K` independent workers on purpose.

Running is intentionally non-exclusive because independent reruns are what prove the result. The queue should allow many workers to confirm the same code path.

## 3. VERIFIER

- Runs CI reproduce checks.
- Runs `confirm.py` for `K`-replication confirmation.
- Uses `triage.py` to decide whether a proposed goal is ready.
- Closes the loop only after the submission is reproducible and replicated.

## Lifecycle

`proposed -> runnable -> submitted -> confirmed -> closed`

- `proposed`: a goal or task idea exists, but no runnable experiment is ready.
- `runnable`: the AUTHOR has written the task so someone else can execute it.
- `submitted`: the RUNNER has produced a run artifact and PR submission.
- `confirmed`: CI reproduce plus `confirm.py` have enough independent support.
- `closed`: the verified result is merged into the project record and the goal is done.

## State ownership

- `proposed` is produced by planning and consumed by the AUTHOR.
- `runnable` is produced by the AUTHOR and consumed by the RUNNER.
- `submitted` is produced by the RUNNER and consumed by the VERIFIER.
- `confirmed` is produced by the VERIFIER and consumed by goal closure.
- `closed` is the terminal state for readers and future planning.

The key rule is simple: claiming is exclusive for authoring, but running is intentionally replicated.
