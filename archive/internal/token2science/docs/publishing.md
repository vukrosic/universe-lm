# Publishing

This is how a contributor turns confirmed work into a paper with their name on it.

## Generate the draft

Run:

```bash
python paper.py --goal G --me <handle>
```

That writes `papers/<goal>.md`, which is the source draft for the paper or result writeup.

## What gets published

- `result card`: one confirmed finding, usually one task and one claim.
- `goal paper`: a closed goal with the confirmed results stitched into one narrative.

Use a result card when the work is still narrow. Use a goal paper when the goal is closed and the story is complete.

## Authorship rule

Authorship is earned from merged plus confirmed runs.

- Git history is the ground truth for who did what.
- `verify/reputation.py` orders contributors by score.
- `verify/confirm.py` is the gate that says a run was reproduced by `K` independent workers.

In other words, the paper is not authored by vibes. It is authored by merged code, confirmed evidence, and the history recorded in git.

## Recommended flow

1. Run the experiment and submit the run.
2. Wait for CI reproduce to pass.
3. Wait for `verify/confirm.py` to show `K`-replication support.
4. Generate the paper with `paper.py`.
5. Put the authors in `verify/reputation.py` order, then revise the draft.
6. Merge the final writeup once the goal is closed.

## Export options

- GitHub Pages for a public web copy.
- PDF for a stable archive or sharing.
- Zenodo DOI for a citable frozen release.
- arXiv for formal paper distribution.

Each export should point back to the confirmed run artifacts so readers can trace the claim to the evidence.
