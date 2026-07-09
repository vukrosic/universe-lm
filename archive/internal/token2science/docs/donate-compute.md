# Donate Compute

This guide is for donors who want to contribute compute, not just run an agent.

The goal is simple: give `token2science` a machine that can reproduce submitted runs in CI.

Why this matters:

- Token donation is already cheap to scale.
- The bottleneck becomes verification.
- As submissions grow, GitHub-hosted CI minutes and queue time become the limiting resource.
- A donated runner gives the project more reproduce capacity without asking every donor to burn shared CI.

## What the runner does

The runner should execute the verify workflow's reproduce step.

That means:

- checkout the PR
- read the submitted `runs/**/result.json`
- rerun the experiment with `REPRODUCE=1`
- compare the reproduced value against the submission

It should do only that job.

## High-level setup

1. Open the repository in GitHub.
2. Go to `Settings` -> `Actions` -> `Runners`.
3. Click `New self-hosted runner`.
4. Pick the machine OS you will use.
5. Download and configure the runner on an isolated machine or VM.
6. Give it a clear label, such as `token2science-repro`.
7. Make sure the verify workflow targets that label in `runs-on`.

The practical rule is:

- the runner registers with GitHub
- the workflow selects that runner by label
- the reproduce job lands on your donated compute

## Recommended runner shape

- Ephemeral VM or container, not a long-lived personal workstation.
- Fresh filesystem for each job or each run.
- No access to private secrets.
- No broad network or repository write access.

## Security caveats

Treat reproduction jobs as untrusted.

They come from PRs and submitted artifacts, so the runner must assume the job can be hostile.

Use these constraints:

- Use ephemeral runners when possible.
- Run inside an isolated VM or container.
- Do not mount personal credentials, cloud keys, or password managers.
- Do not store repo deploy keys on the runner.
- Do not give the runner permission to deploy anything.
- Limit it to reproduce only, never release or publish.

If a reproduce job is compromised, the blast radius should be the runner instance itself, not your laptop or production account.

## Good operating model

- One runner label for reproduce work.
- One workflow path for verification.
- One narrow permission set.
- One clear rule: the runner verifies science, it does not ship code.

If you are donating hardware to the project, this is the highest-leverage use of it.
