# Literature-review pipeline

Parallel to [`../autoresearch/PIPELINE.md`](../autoresearch/PIPELINE.md) — same
shape (file state machine, `flip.sh`, cmf workers via `bin/orchestrate.sh`), but
the artifact is **understanding papers**, not running GPU ablations.

**Prime directive:** keep the digest queue fed. An empty `needs-digest` queue with
papers stuck at `needs-screen` means the screener is behind; an empty screen queue
means the scout is behind.

## Shape: three gates

| Gate | Doer | Critic | Question |
|---|---|---|---|
| **Find** | scout | screener | Is this paper worth a deep read? |
| **Screen** | scout (re-scout) | screener | (re-file loop on `needs-rescout`) |
| **Digest** | digester | digest-reviewer | Is the digest accurate and actionable? |

Optional fourth pass (batch, not per-paper):

| Pass | Agent | Input | Output |
|---|---|---|---|
| **Synthesize** | synthesizer | `brief.md` + done `digest.md` files | `synthesis.md` in `litreview/` |

## Source of truth: `paper.md` frontmatter

```yaml
---
id: 001-forgetting-transformer
status: needs-screen
round: 1
updated: 2026-06-09T15:00:00Z
arxiv: "2503.02130"
theme: attention
---
```

- `status` — routing key only.
- `round` — per-gate revise cap (3 rounds, same rule as autoresearch).
- `arxiv` / `doi` / `url` — dedup keys; scout must fill at least one.
- `theme` — tag for synthesis grouping (e.g. `attention`, `optimizer`, `position`).

## Status vocabulary

Queued:

| status | claimed by | gate |
|---|---|---|
| `needs-scout` | scout | find (doer — re-scout sent-backs) |
| `needs-screen` | screener | find (critic) |
| `needs-rescout` | scout | find (doer revises `paper.md`) |
| `needs-digest` | digester | digest (doer) |
| `needs-digestreview` | digest-reviewer | digest (critic) |
| `needs-redigest` | digester | digest (doer revises) |

In-flight locks:

`scouting` · `screening` · `digesting` · `digestreviewing` · `redigesting`

Terminal:

| status | meaning |
|---|---|
| `done` | `digest.md` written, actionable verdict recorded |
| `rejected` | killed at screen or digest review; folder → `papers/_closed/` |

Synthesis (repo-level, not per-paper):

| status | file | meaning |
|---|---|---|
| `needs-synth` | `litreview/synthesis.md` frontmatter or `synth-request.md` | synthesizer should run |
| `synth-done` | `litreview/synthesis.md` | lit review section written |

## State machine

```text
scout files paper.md ─► needs-screen
         │  screener → screening → screen.md verdict
         ▼
    accept ─► needs-digest
    revise ─► needs-rescout → scout → needs-screen (round++)
    reject ─► _closed/

needs-digest
         │  digester → digesting → digest.md
         ▼
      needs-digestreview
         │  digest-reviewer → digestreviewing → digest-review.md verdict
         ▼
    accept ─► done
    revise ─► needs-redigest → digester (round++)
    reject ─► _closed/
```

## Per-paper artifacts

| File | Writer | Purpose |
|---|---|---|
| `paper.md` | scout | metadata + source + why filed |
| `screen.md` | screener | triage log (`accept` / `revise` / `reject`) |
| `digest.md` | digester | deep read: mechanism, claims, relevance |
| `digest-review.md` | digest-reviewer | accuracy / actionability verdict |
| `log.jsonl` | `flip.sh` | event log |

## Dedup: `seen.md`

Scout reads `litreview/seen.md` before filing. Screener/digest-reviewer append on
`reject`. Format: one line per closed paper — `arxiv:ID — reject: reason — date`.

## Claim protocol

Same as autoresearch — never hand-edit frontmatter:

```bash
litreview/bin/flip.sh <paper-slug> <new-status> <agent> "<note>" [round]
```

## Agent → prompt map

| Agent | Prompt | Greps | Writes |
|---|---|---|---|
| scout | `prompts/scout.md` | `needs-rescout`, then WIP gate + search | `paper.md` (`needs-screen`); re-scouts `needs-rescout` |
| screener | `prompts/screener.md` | `needs-screen` | `screen.md` → `needs-digest` / `needs-rescout` / `rejected` |
| digester | `prompts/digester.md` | `needs-redigest`, then `needs-digest` | `digest.md` → `needs-digestreview` |
| digest-reviewer | `prompts/digest-reviewer.md` | `needs-digestreview` | `digest-review.md` → `done` / `needs-redigest` / `rejected` |
| synthesizer | `prompts/synthesizer.md` | `synth-request.md` flag or manual invoke | `synthesis.md` |

## Orchestration

```bash
litreview/bin/orchestrate.sh          # reclaim stale locks, fan out cmf workers
litreview/bin/orchestrate.sh --dry-run
```

Cron suggestion (same host as autoresearch):

```bash
*/5 * * * * /path/to/litreview/bin/orchestrate.sh >> /tmp/litreview-orch.log 2>&1
```

Launch a one-off scout when the screen queue is empty:

```bash
# orchestrate handles per-paper workers; for a cold start:
scripts/launch_minimax.sh lit-scout "$(cat litreview/prompts/scout.md)"
```

## Link to autoresearch

When `digest.md` ends with **Suggested action: file-idea**, a human or a follow-up
agent copies the mechanism into `autoresearch/ideas/` via the idea-miner shape.
Litreview **never** flips autoresearch statuses — it only produces digests.
