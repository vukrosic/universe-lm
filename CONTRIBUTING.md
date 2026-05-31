# Contributing to Universe

The first way to contribute is to pick an experiment from the issues. Many are
fully spelled out — a self-contained task with code or exact step-by-step
instructions attached — so you can hand one straight to your AI agent and let it
run. For these, you can point Claude Code, Codex, or any AI agent straight at the
issue link and it should handle the whole run end-to-end — pull, train, evaluate
— on its own.

## Reporting results

When the run finishes, the agent reports back **in the issue** — not as a pull
request. Post a comment with:

- A short summary of what was run and the final numbers (a metrics table).
- Plots / screenshots — loss curve, eval results — so the outcome can be
  eyeballed at a glance.
- The seed, the resolved config, and the exact command used, so the run is
  reproducible.

For the raw evidence, commit the full run output — config, logs, metrics, plots — to a **branch in your own fork**, then leave it untouched.
Don't open a PR and don't push to `main`. If the numbers look interesting, we'll
pull up that branch in your repo and review the raw results there.

Keep `model.pt` out of git; if you want a reusable checkpoint, save it separately under `checkpoints/<version>/model.pt` or use the release pipeline.

The issue comment is the report; the frozen branch is the evidence. We curate
the winners into `main` ourselves.
