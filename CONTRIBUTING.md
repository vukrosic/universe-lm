# Contributing to Universe

Everything lands as a **pull request**, and credit works one way: **accepted PRs get
named credit on the published report.** There are two ways in.

## Path 1 — Claim a paper task (the main path)

Open [`tasks/`](tasks/) and pick a task (P1–P8 and growing). Each one is a self-contained
1–2 day experiment from a recent paper: what to read, what to implement, what to train,
and what an accepted result looks like. You run it yourself on a cheap rented GPU —
most tasks cost **$2–25** at the 23M/52M configs.

The rules that make a result acceptable are in [`tasks/README.md`](tasks/README.md).
The short version:

1. **Baseline first.** Your first PR reproduces the pinned baseline
   (`--config_class configs.llm_config.Ladder23M469MConfig`, seed 42) and reports the
   loss curve, final val loss, wall-clock, and GPU. Every task compares against it.
2. **Read the actual paper PDF** before running anything.
3. **Equal token budgets between arms, control run in the same session,** config diff +
   logs + **at least one figure** in the PR. No figure, not accepted.
4. Data-axis arms are scored on the shared FineWeb-Edu held-out set
   (`scripts/bpb_fineweb_edu.py`). Different tokenizers or corpora → report
   **bits-per-byte**, never per-token loss.

The live task board is beads (`bd ready` — see [TASKS.md](TASKS.md)); the task files in
`tasks/papers/` are the briefs. Comment on the matching issue (or open one) to claim,
so two people don't burn GPU money on the same task.

## Path 2 — Take the speedrun record

Race for the **lowest val loss on the `10m` config** (~10M params, 200M tokens, ~33 min
on one consumer GPU). Pinned: `seed=42`, bf16. A record must beat the standing one by
**≥0.01**. Standings and tier details: [LEADERBOARD.md](LEADERBOARD.md).

Mechanism PRs follow one shape:

- **One structural mechanism per PR**, behind a new `use_<name>` config flag that is
  **OFF by default** (in `configs/`, `models/`). Default-OFF means merging your PR
  changes nothing until a run turns it on — it can never silently regress the champion.
- **Structural ideas only** — attention, positional encoding, normalization, FFN, loss,
  residual routing. **No hyperparameter sweeps**: learning rate, weight decay, batch
  size, schedule, and init are tuned and closed; a PR that only changes a number will
  be rejected.
- Evidence in the PR: your run's `metrics.json`, the command line to reproduce it, and
  the in-session control you compared against. Use `tiny1m3m` / `screen10m` to iterate
  cheaply, but only a `10m` win takes the record.

## Every PR, both paths

- Reproducible: exact command line, config diff, seed, and committed `runs/<name>/metrics.json`.
- Honest: negative results are welcome and get credited — a clean "this lever does
  nothing here" saves everyone else the GPU money.
- Clean: keep `model.pt`, checkpoints, and datasets out of git.

Questions → open an issue or reach [@vukrosic](https://github.com/vukrosic).
