# Idea pipeline

Multi-agent loop that takes an idea from "mined" to "ran on hardware".
Agents are triggered manually and occasionally; each one **polls** the idea
folders, claims work by reading + flipping a status field, and stops when its
queue is empty. No agent talks to another directly — the only channel is the
`status` field in each `idea.md` frontmatter and the per-gate review log.

## Shape: three doer ↔ critic gates

Every stage is a **doer** paired with an adversarial **critic**. The doer
produces an artifact; the critic, whose default is skeptical, issues one verdict
(`accept` / `revise` / `reject`); on `revise` the doer revises and they loop, up
to a **3-round cap** after which the critic must `accept` or `reject` — no idea
cycles forever. An idea must clear all three gates before it runs:

| Gate | Doer | Critic | Question |
|---|---|---|---|
| **Taste** | miner | taste-reviewer | Is this idea worth a slot *at all*? |
| **Definition** | reviser | reviewer | Is the idea *fully & soundly specified*? |
| **Code** | code-implementer | code-reviewer | Does the code *match the spec & run correctly*? |

In the taste and code loops the **doer revises itself**; the definition loop uses
a separate reviser. **Every idea runs at one tier only — `tiny1m3m` (0.94M params
· 3M tokens), seed 42** — and takes the full path through all three gates. There
is no larger tier and no cost-gate shortcut.

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

| status | claimed by | gate |
|---|---|---|
| `needs-taste` | taste-reviewer | taste (critic) |
| `needs-repitch` | miner | taste (doer revises) |
| `needs-review` | reviewer | definition (critic) |
| `needs-revision` | reviser | definition (doer revises) |
| `needs-plan` | code-implementer | code (doer) |
| `needs-codereview` | code-reviewer | code (critic) |
| `needs-recode` | code-implementer | code (doer revises) |
| `needs-run` | run scheduler (human / Kaggle harness) | — |

In-flight (acts as the lock — one agent holds it):

`tasting` · `repitching` · `reviewing` · `revising` · `planning` ·
`codereviewing` · `recoding` · `running`

Terminal:

| status | meaning |
|---|---|
| `done` | ran, `evidence.md` written, win or null logged |
| `rejected` | killed in review; folder moved to `autoresearch/ideas/_closed/` |

## State machine

Each gate has the same shape: doer → `needs-<critic>` → critic verdict →
{accept: next gate · revise: back to doer, round++ · reject: `_closed/`}.

```text
miner ─► needs-taste                                    ┌─ GATE 1: TASTE ─┐
                │  taste-reviewer → tasting → taste.md r_n verdict
                ▼
         ┌──────┴───────┬────────────────┐
      accept          revise           reject ─► _closed/
         │               │
         │          needs-repitch → miner re-pitches → needs-taste (round++)
         │
         ▼  (round reset to 1)                             ┌─ GATE 2: DEFINITION ─┐
      needs-review
                │  reviewer → reviewing → review.md r_n verdict
                ▼
         ┌──────┴───────┬────────────────┐
      approve         revise           reject ─► _closed/
         │               │
         │          needs-revision → reviser → revising → idea.md, round++ → needs-review
         ▼
      needs-plan                                          ┌─ GATE 3: CODE ─┐
                │  code-implementer → planning → plan.md + code
                ▼
      needs-codereview
                │  code-reviewer → codereviewing → codereview.md r_n verdict
                ▼
         ┌──────┴───────┬────────────────┐
      accept          revise           reject ─► _closed/
         │               │
         │          needs-recode → code-implementer re-codes → needs-codereview (round++)
         ▼
      needs-run
         │
      running ─► done   (evidence.md written)
```

Each gate runs its **own** 3-round budget: on `accept` into the next gate, the
critic resets `round` to 1.

## Run + evidence (the last mile, single-pass)

`needs-run` is owned by the **runner** (`prompts/runner.md`) — run + pull +
analyze in one pass; *not* "human, for now." Raw run data is the durable
`results.json` under `remote-results/<date>-vast-<tier>/` (logs alongside). The
pipeline-side record is **`evidence.md`** in the idea folder:

```markdown
# Evidence — NNN <name>

## Verdict: <WIN | NULL>
- tier, seed 42, box host
- control val · treatment val · Δ
- pass/fail bar (from plan.md) → met | not met
- box check: ctrl vs leaderboard (within noise | DRIFT)
- raw: remote-results/<dir>/results.json
- date
```

Both outcomes flip the idea to `done` (= ran, evidence written, win-or-null
logged). A **NULL** also gets one line in `closed.md` so it's never re-mined. The
runner never `reject`s — a clean null is a result. Box-validation rule: a control
that drifts > ~0.01 val loss from `LEADERBOARD.md` means the box is bad; its
results aren't trusted and the idea stays `needs-run`.

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

## Critic log format (append-only, newest round on top)

Each gate's critic keeps its own log in the idea folder — `taste.md` (taste),
`review.md` (definition), `codereview.md` (code). Same shape for all three:

```markdown
# <Taste|Review|Code-review> log — NNN <name>

## r2 — 2026-06-08 — verdict: accept
- ...

## r1 — 2026-06-08 — verdict: revise
- finding 1
- finding 2
```

Verdict is exactly one of `accept` (definition gate calls it `approve`) /
`revise` / `reject`. It sets `status`: `accept→` next gate, `revise→` the doer's
`needs-*` queue, `reject→rejected`.

## Hard rules

- **🔴 ONE SEED ONLY.** Every ablation runs at a **single fixed seed (42)** —
  never multi-seed. No `≥3 seeds`, no seed sweeps, no per-seed means. A single
  seed keeps each A/B cheap enough to actually run on the constrained hardware;
  multi-seed protocols are out of scope for this pipeline. Any idea, plan, or
  review that asks for more than one seed is **malformed** — strip it down to
  seed 42 instead. Read a sub-noise effect as **inconclusive, not real**; do
  not "add seeds to confirm" — log it inconclusive and move on.
- **3-round cap — every gate.** Each gate runs its own `round` budget. On
  `round: 3` the critic may only `accept` or `reject` — `revise` is forbidden,
  forcing the call. No idea cycles more than 3 times *within a gate*; on `accept`
  into the next gate the critic resets `round` to 1.
- **🔴 ONE TIER ONLY — `tiny1m3m`.** Every experiment runs at tiny1m3m (0.94M
  params · 3M tokens, seed 42) and nothing else. No `screen20m`, no full ladder,
  no multi-tier promotion — that scope is out. An idea whose payoff only appears
  at larger scale is a `reject` at the taste gate. Because there's a single tier,
  there is **no cost-gate**: every accepted idea takes the full path
  `needs-taste → needs-review → needs-code → needs-codereview → needs-run`.
- **Rejects leave the scan path.** `rejected` → move the folder to
  `autoresearch/ideas/_closed/` and append a line to `autoresearch/closed.md`.
  Active greps stay clean.
- **One verdict per critic pass.** A review that ends without exactly one verdict
  is malformed.
- **Critics close, doers don't.** Each gate's critic is on the `closed.md` write
  path, on its own `reject` (taste-reviewer, reviewer, code-reviewer). Tag the
  reason by gate: `taste-reject` / `reject` / `code-reject`. Doers never close —
  if a doer is blocked it bounces the idea back to a `needs-*` queue, not to
  `rejected`. Post-run null results are appended to `closed.md` by the
  evidence/run step (human, for now). The miner **reads** `closed.md` before
  filing; never writes it.

## Agent → prompt map

| Agent | Prompt | Greps | Writes |
|---|---|---|---|
| miner | `autoresearch/prompts/idea-miner.md` | `needs-repitch`, then Step 0 WIP gate + mine | `idea.md` (`needs-taste`); re-pitches `needs-repitch`; skips mining if `upstream≥6` (GPU-queued ideas don't count) |
| taste-reviewer | `autoresearch/prompts/idea-taste.md` | `needs-taste` | appends `taste.md`, → `needs-review`/`needs-run`/`needs-repitch`/`rejected` |
| reviewer | `autoresearch/prompts/idea-reviewer.md` | `needs-review` | appends `review.md`, flips status |
| reviser | `autoresearch/prompts/idea-reviser.md` | `needs-revision` | edits `idea.md`, `round++` |
| code-implementer | `autoresearch/prompts/code-implementer.md` | `needs-recode`, then `needs-plan` | `plan.md` + code, → `needs-codereview` |
| code-reviewer | `autoresearch/prompts/code-reviewer.md` | `needs-codereview` | appends `codereview.md`, → `needs-run`/`needs-recode`/`rejected` |
| runner | `autoresearch/prompts/runner.md` | `needs-run` | `remote-results/<date>-vast-<tier>/{*.log,results.json}` + `evidence.md`, → `done` (and null line in `closed.md`) |
