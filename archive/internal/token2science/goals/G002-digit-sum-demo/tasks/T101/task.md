# T101 - run the toy experiment and submit score

**Goal:** G002-digit-sum-demo
**Status:** open

## What to do
1. Run `python experiment.py --config config.json` in this folder.
2. Read the `RESULT metric=score value=...` line it prints.
3. Submit it: `python ../../../../worker/worker.py submit \
   --goal G002-digit-sum-demo --task T101 --worker <your-gh-handle>`
   (the worker runs the command for you and writes the run artifact).
4. Open a PR. CI reproduces it and posts the verdict.

## Acceptance
- config hash in your `result.json` matches `config.json`,
- CI re-runs the experiment and gets the same `score` within `tolerance`.
