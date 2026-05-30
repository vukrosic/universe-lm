# Experiment Workflow

How experiments are tracked, frozen, and archived — so `main` stays clean, every
result is preserved forever, and contributors can pick up where someone stopped.

Applies to **every** experiment (any mechanism, any size). Not specific to one study.

---

## Principle: track the *claim*, discard the *evidence*

Evidence is regenerable from `(config + seed + commit)`. The claim isn't. So:

| | What | Where it lives |
|---|---|---|
| **TRACK** (permanent) | one leaderboard row: `Run · val_loss · author · summary · date · commit/evidence` | leaderboard text on `main` |
| **FREEZE** (per-experiment) | mechanism code + final `metrics.json` | the experiment's tag (see below) |
| **DISCARD** (regenerable) | plots, per-step logs, intermediate sweep JSONs, one-off runner scripts | gitignored — never on `main` |

**The honesty rule:** *no commit hash → not on the leaderboard.* If a number can't
name the code that produced it, it doesn't count. This single rule is what keeps
the repo clean: once the hash captures the run, the regenerable junk is worthless
to keep.

---

## The unit: one experiment = one branch + one issue

- **Branch** (`exp/<name>`) — holds the mechanism code. Where contributors run and append.
- **Issue** — the running log; contributors post their numbers as they run sizes.
- When resolved, **one distilled row** goes to the `main` leaderboard.

Commit experiments to their branch — **never directly to `main`.** `main` only ever
receives *promoted winners* (merged) plus the leaderboard text.

---

## Lifecycle: branches are temporary, tags are the archive

A branch and a tag both just point at a commit. The **commit** holds the files — the
label is only a sticky-note. A branch is "still moving"; a tag is "frozen forever."
So when an experiment resolves, tag the tip and delete the branch.

```text
ACTIVE      → branch  (exp/qk-gain)        ← contributors run / append here
   │
   ├── WON  → merge to main + tag it       → delete branch
   │           main now has the mechanism; leaderboard row links the merge
   │
   └── NULL → tag it (result/qk-gain-null) → delete branch
               main stays clean; tag preserves the evidence forever
```

- **Branches** are alive only while an experiment is open. The branch list stays short and meaningful.
- **Tags** are the permanent archive. Every experiment — won or null — leaves an immutable tag. Cheap, forever, invisible in normal branch view.
- **`main`** accumulates only: promoted mechanisms + leaderboard rows + configs/protocol/README.

### Why a deleted branch loses nothing

The tag keeps the commit reachable, so it's never garbage-collected. `git checkout <tag>`
loads the exact snapshot (all files + `metrics.json`) in **detached HEAD** — no branch
needed, no merge to `main` required. To *continue* working from a frozen result:

```bash
git checkout -b continue-qk-gain result/qk-gain-10M
```

That starts a fresh moving branch from the exact frozen state.

---

## The leaderboard is the record, not a queue

One race: **lowest val loss on the `10m` config.** Each row is a general improvement
(any mechanism), ranked by val loss; a challenger takes the record only by beating the
standing best by **≥0.01**. Smaller configs (screens) are for experimentation — tracked,
but nothing counts until it also beats `10m`.

```text
| Val loss | Run               | Author   | Commit |
| 5.01     | baseline          | vukrosic | a1b2c  |
| 4.95     | QK-gain init=2.2  | alice    | d4e5f  |
```

A contributor checks out a challenger branch (or a tag), runs `10m`, and if they beat
the record opens a PR — the win merges to `main` as the new champion architecture. The
135M model is the **mission** (beat SmolLM2-135M), trained once the recipe and compute
are there — it is *not* a leaderboard row.
