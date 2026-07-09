# Contributing to Universe

You contribute **code, not compute.** You take a brief, your AI writes one
experiment, and you open a pull request. A maintainer reviews and merges it, then
runs it on a single reference GPU under a fixed integrity gate. You never run
training yourself and no API keys ever change hands — **GitHub is the gate.**

This keeps results trustworthy: every architecture is measured the same way, on
the same box, with paired same-seed controls, so a win is a real win and not GPU
noise. (Donating GPU compute is a separate path that opens later — for now the
maintainer's box runs everything.)

## The loop in one picture

```
pick a brief  ->  your AI writes one experiment  ->  open a PR
                                                        |
                                          maintainer reviews + merges
                                                        |
                            maintainer's daemon runs it on the reference box
                              (same-seed paired controls, confirm gate)
                                                        |
                                   result lands on the record timeline
```

## What you submit

Each experiment is **one structural mechanism** stacked on the current champion,
behind a feature flag that is **OFF by default**. A PR adds exactly three things:

1. **The mechanism**, behind a new `use_<name>` config flag (default `False`) in
   the model code (`configs/`, `models/`). Default-OFF means merging your PR
   changes nothing until the stub turns it on — so it can never silently regress
   the champion.
2. **An experiment stub** `_arq_<id>-<slug>.py` at the repo root that enables your
   flag on top of the champion config and launches training. Copy an existing
   `_arq_*.py` as a template — they all follow the same shape.
3. **A queue entry** under `archive/internal/autoresearch/ideas/<id>-<slug>/`:
   - `idea.md` — YAML frontmatter (`id`, `status: needs-run`, `plain:` one-line
     hypothesis) then a short body: the mechanism, the citation, and a
     **falsifiable claim** (e.g. "val < 6.1700 ⇒ screen-win").
   - `run.json` — `{ "name": "<id>-<slug>", "arq_file": "_arq_<id>-<slug>.py", "job_timeout": "12m" }`

That is the whole deliverable. No training run, no metrics, no plots — the
reference box produces those.

## The one rule: novel mechanisms, not hyperparameter search

Submit **structural** ideas — attention, positional encoding, normalization, FFN,
loss, residual routing. **Do not** sweep learning rate, weight decay, momentum,
batch size, schedule, or init scale. Those axes are already tuned and closed; a PR
that only changes a number will be rejected. Build your candidate the same way as
the champion so the only difference measured is your mechanism.

## How to start

1. Open a brief from the board (or an issue labeled `research`). It contains a
   self-contained prompt you can hand straight to Claude Code, Codex, or any agent.
2. Your agent forks/clones, implements the three pieces above, and opens a PR to
   `main` titled after the brief.
3. A maintainer reviews the code (the merge is the only trust boundary), merges,
   and the daemon picks it up — builds it on the box's GPU venv (smoke test), runs
   it, and confirms wins through the paired gate.
4. The outcome shows up on the public record timeline. The champion only ever
   moves through the confirm gate.

Keep `model.pt` and any checkpoints out of git.
