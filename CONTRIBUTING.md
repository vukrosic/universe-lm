# Contributing to Universe

Universe is an open LLM lab, run in public. We release a new model every week
and document everything as we go. Contributions are welcome — code, ideas,
experiments, papers, bug reports.

## What we're working on right now

The active research target is a **~70M parameter model**. We're lifting
techniques from [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)
and adapting them to our training stack. See open issues labeled `research`.

Releases live in [releases/](releases/) and on the HuggingFace
[vukrosic](https://huggingface.co/vukrosic) profile.

## Ground rules

1. **Open work.** Pick an issue or open one before starting. Drop a comment so
   nobody duplicates effort.
2. **One change, one PR.** Don't bundle unrelated changes.
3. **Reproducible.** New results need a seed, a config, and a log. No
   cherry-picked single-seed numbers in the leaderboard.
4. **Show your math.** If you claim a speedup or quality gain, link the diff,
   the log, and the eval command.
5. **Be kind.** This is a learning lab as much as a leaderboard.

## How to contribute

### Bug fix or small improvement
1. Open an issue (or comment on an existing one) describing the problem.
2. Fork, branch, fix, test.
3. PR against `main`. Reference the issue.

### Research experiment
1. Pick (or open) an issue labeled `research`.
2. Run the baseline first. Note the seed and config.
3. Implement the change as a config flag or branch — don't break the baseline.
4. Run 3 seeds. Report mean + std.
5. PR with: code change, before/after numbers, training log, and a one-paragraph
   write-up of what you tried and what happened (including negative results).

### Documentation / good-first-issue
Look for issues labeled `good-first-issue`. README fixes, docstrings, small
refactors, eval script improvements are all fair game.

## Branches

We use **one branch per change**. Don't experiment on `main`.

### Naming

| Prefix | When to use | Example |
|--------|-------------|---------|
| `experiment/<slug>` | A single architectural or training change | `experiment/qk-norm` |
| `sweep/<slug>` | Run many variants, produce a CSV, die | `sweep/width-depth-70m` |
| `fix/<slug>` | Bug fix | `fix/generate-crash` |
| `docs/<slug>` | Docs-only | `docs/release-pipeline` |
| `release/v0.X` | Frozen at release time | `release/v0.3` |

### Lifecycle

| Branch | After it lands | Why |
|--------|----------------|-----|
| Merged to `main` | **Delete locally + remote** | History is in `main`, branch clutters the list |
| Abandoned (didn't work) | **Keep on remote, delete locally** | Negative results are useful — link from a "graveyard" doc |
| `sweep/*` after CSV written | **Delete both** | Results live in `experiments/results/` |
| `release/v0.X` | **Keep forever** | Reproducibility tag |

Code lives in `main`. Results live in `main`. Branches are scratch.

### Working on multiple experiments in parallel (git worktrees)

For AI agents or humans juggling several experiments, use worktrees instead of
cloning the repo multiple times:

```bash
git worktree add ../universe-worktrees/qk-norm experiment/qk-norm
git worktree add ../universe-worktrees/mup    experiment/mup
# ...work in each folder independently...
git worktree remove ../universe-worktrees/qk-norm
```

Each worktree has its own working tree and branch but shares one `.git`
directory. **GPU is the real bottleneck, not git** — agents can edit code in
parallel, but only one training run at a time on a single GPU. Queue runs.

### Releases come from `main`

When cutting a release: branch `release/v0.X` from `main` at the chosen commit,
train, tag, push. The `release/*` branch never gets new commits after the tag.

## Pull request checklist

- [ ] Linked to an issue
- [ ] Tests pass / training smoke run works
- [ ] No unrelated changes
- [ ] Logs and configs included if claiming a result
- [ ] Updated docs if behavior changed

## Code of conduct

Be respectful. Disagree on ideas, not on people. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## Questions

Open an issue with the `question` label, or DM
[@vukrosic](https://github.com/vukrosic).
