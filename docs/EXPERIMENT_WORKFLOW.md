# Experiment Workflow

How experiments are tracked, frozen, and archived — so `main` stays clean, every
result is preserved forever, and contributors can pick up where someone stopped.

Applies to **every** experiment (any mechanism, any size). Not specific to one study.

---

## Principle: track the *claim*, discard the *evidence*

Evidence is regenerable from `(config + seed + commit)`. The claim isn't. So:

| | What | Where it lives |
|---|---|---|
| **TRACK** (permanent) | one row: `config · seed · commit hash · final val_loss · who/GPU` | leaderboard text on `main` |
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

## The leaderboard is the queue

Filled rows = done. `OPEN` rows = jobs waiting for a donor's compute.

```text
| size | val_loss | seed | commit | who/GPU            |
| 10M  | 4.98     | 42   | a1b2c  | you/5060           |
| 25M  | OPEN — needs a donor                          |
| 135M | OPEN — needs a donor                          |
```

A contributor checks out the experiment branch (or a tag), runs the missing rung,
appends their row, and freezes their `metrics.json`. The branch + issue is the
handoff point — someone literally continues where the last person stopped.
