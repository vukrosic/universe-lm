Queue format matches plan 05: each spec is a self-contained `experiment.yaml` with `id`, `title`, `repo`, `commit`, `command`, `config`, `requires`, and `report`.
Status lives in the filename: `.queued.yaml`, `.claimed.yaml`, or `.done.yaml`.
Runner entrypoint: `python runner/runner.py queue/arq-030-unetskip.done.yaml`.
Dry run all queued specs with: `python runner/runner.py --dry-run`.
Dry run validates the pinned commit and spec shape, but does not execute training.
Real runs write `results/<spec-id>/<user>-<utc-timestamp>/result.json`.
If the command or artifacts expose a loss curve, the runner also writes `loss_curve.csv`.
Archived output tails are written to `log_tail.txt` when available.
