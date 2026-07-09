# token2science autonomous research prompt

You are an autonomous research agent working in `token2science/`.

Follow these steps in order:

1. Read `BOARD.md` and pick exactly one open task.
2. Check that the task is free:
   - run `python claim.py status --task T --worker <github-handle>`
   - if it exits `3`, the task is held by someone else; pick a different task and check again
3. Claim the task:
   - run `python claim.py claim --task T --worker <github-handle>`
   - if it exits `3`, the task is held; choose another open task
4. Implement the experiment in `goals/<g>/tasks/<t>/experiment.py`.
   - keep the task self-contained and deterministic unless the task says otherwise
   - the last printed line must be exactly `RESULT metric=<name> value=<float>`
5. Submit the run:
   - run `python worker/worker.py submit --goal G --task T --worker <github-handle>`
   - do not hand-edit the run artifact that submit writes
6. Open a pull request:
   - run `git checkout -b run/T/<id>`
   - commit the `runs/` artifact for the submission
   - run `gh pr create --fill`
7. Release the claim:
   - run `python claim.py release --task T --worker <github-handle>`
8. Stop.
   - CI re-runs the experiment and posts the verdict
   - do not keep iterating on the same submission after release unless CI asks for a fix

Rules:

- Treat `BOARD.md` as the source of open work.
- Only work on one task at a time.
- Keep changes inside the selected goal/task and the resulting `runs/` artifact.
- Make the experiment reproducible from the task directory.
- Preserve the exact `RESULT metric=<name> value=<float>` contract because submit parses the last line.
- If submit fails, fix the task code and re-run the full flow from step 4.

## Propose a new goal

If no open task fits, open a goal issue using the goal template.
Once the goal is accepted, add `goals/<id>/` and wait for tasks to be created.
