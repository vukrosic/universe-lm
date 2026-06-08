# Idea pipeline

Multi-agent loop that takes an idea from "scouted" to "ran on hardware".
Agents are triggered manually and occasionally; each one **polls** the idea
folders, claims work by reading + flipping a status field, and stops when its
queue is empty. No agent talks to another directly — the only channel is the
`status` field in each `idea.md` frontmatter and the `review.md` log.

## The one source of truth: `idea.md` frontmatter

```yaml
---
id: 001-cautious-muon
status: needs-review
round: 1
updated: 2026-06-08T16:50
---
```

- `status` — **routing key**. Says *where in the pipe* the idea is, never a verdict.
- `round` — review-cycle counter. Caps the reviewer↔reviser loop (see below).
- `updated` — ISO timestamp, bumped on every status flip. Crash-recovery handle:
  an `-ing` status with a stale `updated` = a dead agent; reset it to its `needs-*`.

There is **no `owner` field and no separate state file** (`feedback.md` is dead).
Status alone routes. Verdicts live only in `review.md`.

## Status vocabulary

Queued (any matching agent may claim):

| status | claimed by |
|---|---|
| `needs-review` | reviewer |
| `needs-revision` | reviser |
| `needs-plan` | code-implementer |
| `needs-run` | run scheduler (human / Kaggle harness) |

In-flight (acts as the lock — one agent holds it):

`reviewing` · `revising` · `planning` · `running`

Terminal:

| status | meaning |
|---|---|
| `done` | ran, `evidence.md` written, win or null logged |
| `rejected` | killed in review; folder moved to `autoresearch/ideas/_closed/` |

## State machine

```text
scout/miner ─► needs-review
                   │  reviewer claims → reviewing → appends review.md r_n with a verdict
                   ▼
            ┌──────┴───────┬───────────────┐
         approve         revise          reject
            │               │               │
            ▼               ▼               ▼
        needs-plan    needs-revision     rejected ─► move to _closed/
            │               │
         planning      reviser claims → revising → edits idea.md, round++
            │               │
            ▼               └─────► needs-review   (re-review)
        needs-run
            │
         running ─► done   (evidence.md written)
```

## The claim protocol (every agent, every run)

Never hand-edit the frontmatter. **All status changes go through `flip.sh`**,
which rewrites the frontmatter *and* appends the event log in one call:

```
autoresearch/bin/flip.sh <idea-slug> <new-status> <agent> "<note>" [round]
```

1. `grep -l "status: <my-queue-state>" autoresearch/ideas/*/idea.md` — find my work.
2. For each hit, **claim it**: `flip.sh <idea> <-ing-lock> <agent> "claimed"`.
3. Do the work.
4. **Release it**: `flip.sh <idea> <next-status> <agent> "<what I did>" [round]`
   (the reviser passes the bumped round as the 5th arg).
5. Repeat until no hits remain, then stop.

The reviewer's append-to-`review.md` and its `flip.sh` release happen in the
**same pass** — never one without the other.

## Event log (per-idea thread)

Each idea folder has its own `log.jsonl` — one line per status flip, written by
`flip.sh`. To see an idea's whole life, read its folder (`idea.md` +
`review.md` + `log.jsonl`). To analyze the system, glob
`autoresearch/ideas/*/log.jsonl`.

```json
{"ts":"…Z","agent":"reviser","idea":"002-…","from":"needs-revision","to":"needs-review","round":2,"note":"applied 4 findings"}
```

This is the substrate for health checks — loop detection (round climbing,
review↔revise oscillation), stuck locks (an `-ing` with no follow-up event),
and throughput. It records **transitions, not file contents** — use `git diff`
to inspect what the text actually changed to.

## review.md format (append-only, newest round on top)

```markdown
# Review log — NNN <name>

## r2 — 2026-06-08 — verdict: approve
- ...

## r1 — 2026-06-08 — verdict: revise
- finding 1
- finding 2
```

Verdict is exactly one of `approve` / `revise` / `reject`. It sets `status`
(`approve→needs-plan`, `revise→needs-revision`, `reject→rejected`).

## Hard rules

- **🔴 ONE SEED ONLY.** Every ablation runs at a **single fixed seed (42)** —
  never multi-seed. No `≥3 seeds`, no seed sweeps, no per-seed means. A single
  seed keeps each A/B cheap enough to actually run on the constrained hardware;
  multi-seed protocols are out of scope for this pipeline. Any idea, plan, or
  review that asks for more than one seed is **malformed** — strip it down to
  seed 42 instead. Read a sub-noise effect as **inconclusive, not real**; do
  not "add seeds to confirm" — log it inconclusive and move on.
- **3-round cap.** On `round: 3` the reviewer may only `approve` or `reject` —
  `revise` is forbidden. No idea cycles more than 3 times.
- **Cost-gate the loop.** The review loop exists to stop bad ideas *before they
  burn compute*. Only gate the expensive ones:
  - tiny1m3m ideas (~2 min on a T4): scout sets `status: needs-run` directly,
    skipping review.
  - screen20m+ ideas: full review loop (`needs-review`).
- **Rejects leave the scan path.** `rejected` → move the folder to
  `autoresearch/ideas/_closed/` and append a line to `autoresearch/closed.md`.
  Active greps stay clean.
- **One verdict per review pass.** A review that ends without exactly one
  verdict is malformed.
- **Only the reviewer closes.** The dedup list `autoresearch/closed.md` has one
  agent on its write path: the reviewer, on `reject`. The code-implementer never
  closes — if blocked it bounces the idea back to `needs-review`. Post-run null
  results are appended to `closed.md` by the evidence/run step (human, for now).
  The miner/scout **read** `closed.md` before filing; never write it.

## Agent → prompt map

| Agent | Prompt | Greps | Writes |
|---|---|---|---|
| scout (in-repo) | `autoresearch/prompts/idea-scout.md` | — | `idea.md` (`needs-review` or `needs-run`) |
| miner (external) | `autoresearch/prompts/idea-miner.md` | Step 0 WIP gate, then mines | `idea.md` (`needs-review` or `needs-run`); skips if `active≥6` or `needs-run≥4` |
| reviewer | `autoresearch/prompts/idea-reviewer.md` | `needs-review` | appends `review.md`, flips status |
| reviser | `autoresearch/prompts/idea-reviser.md` | `needs-revision` | edits `idea.md`, `round++` |
| code-implementer | `autoresearch/prompts/code-implementer.md` | `needs-plan` | `plan.md` + code, → `needs-run` |
