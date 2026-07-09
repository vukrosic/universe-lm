# Brief pipeline (topic / scope selection)

Upstream of [`../PIPELINE.md`](../PIPELINE.md). The idea pipeline answers
"which mechanism wins?"; this pipeline answers **"which question is the next
paper about?"** Same shape — frontmatter state machine, `flip-brief.sh`,
doer ↔ critic gate, 3-round cap — but the unit is a **brief** (a research
campaign), not an idea.

**One active brief at a time.** `autoresearch/brief.md` is always a copy of
the active brief's body (the dashboard and the miner read it there — never
break that path). One brief = one campaign = one eventual paper.

## Lifecycle

```text
proposer files briefs/NNN-slug/brief.md ─► needs-scope
         │  brief-reviewer → scoping → scope.md verdict
         ▼
    accept ─► needs-blessing        (ranked; waits for HUMAN)
    revise ─► needs-repitch → proposer → needs-scope (round++)
    reject ─► _closed/
         │
   HUMAN blesses exactly one ─► active
         │  (its body is copied over autoresearch/brief.md;
         │   losing candidates flip to shelved)
         ▼
      active ── exit criteria met ──► retired
                                        │
                                        ├─► paper-request (paper pipeline)
                                        └─► proposer's step-0 sees no active
                                            brief → proposes next candidates
```

## Status vocabulary

| status | claimed by | meaning |
|---|---|---|
| `needs-scope` | brief-reviewer | candidate filed, awaiting critique |
| `needs-repitch` | proposer | critic said revise |
| `needs-blessing` | **human only** | critic accepted; ranked, awaiting Vuk |
| `active` | — | the live campaign; miner mines under it |
| `shelved` | proposer (may re-file later) | lost the blessing round; kept, not closed |
| `retired` | — | exit criteria met; campaign done → paper |
| `rejected` | — | folder → `briefs/_closed/` + line in `closed-briefs.md` |

In-flight locks: `proposing` · `scoping` · `repitching`.

**Human gate:** `needs-blessing → active` is the only transition agents may
NOT make. Topic choice is brand-level; Vuk flips it:

```bash
autoresearch/bin/flip-brief.sh <slug> active vuk "blessed"
cp autoresearch/briefs/<slug>/brief.md autoresearch/brief.md   # strip frontmatter if desired
```

## Brief frontmatter

```yaml
---
id: 002-positional-encoding-zoo
status: needs-scope
round: 1
updated: 2026-06-10T12:00:00Z
exit: "12 done ideas OR 3 WINs OR 2026-06-24"
---
```

- `exit` — **mandatory.** Machine-checkable retirement condition: count of
  `done` ideas filed under this brief, count of WINs, or a hard date —
  whichever hits first. A brief with no exit criteria is malformed.
- `venue_ceiling` — **mandatory.** Honest best-venue call for the finished
  paper: `tmlr` (needs breadth — ≳30 mechanisms or cross-campaign
  meta-analysis), `workshop` (one clean campaign), or `arxiv` (exploratory).
  Calibrated against `autoresearch/skills/paper-writing/` at the gate
  (reviewer rubric line 7 runs a peer-review pre-mortem). Main-conference is
  never claimable: seed-42-only evidence can't pass a main-track experimental
  bar — breadth, not seeds, raises the ceiling.

## Brief body (sections, all mandatory)

Same shape as the current `autoresearch/brief.md`, plus two new sections that
make it paper-ready:

1. **Topic** — one paragraph.
2. **Research question** — one bold sentence, falsifiable at tiny1m3m/seed-42.
3. **Paper claim** — the single sentence the finished paper will argue. If the
   campaign's WIN/NULL table can't support a one-sentence claim, the brief is
   too diffuse.
4. **Mineability seed list** — ≥10 distinct candidate idea directions with a
   source each (arxiv id / repo). This is proof the miner won't starve; the
   GPU-never-idle rule starts here.
5. **Scope & constraints** — inherits the fixed block verbatim: tier
   `tiny1m3m` only, seed 42 only, mechanisms not HP sweeps, <200 LoC,
   identity/zero-init, dedup vs `closed.md`.
6. **Success criteria** — WIN/NULL bars (ctrl-bracket protocol, unchanged).
7. **Venue case** — justification for `venue_ceiling`: which paper-writing
   quality gates the finished paper passes (scaled per the skill's Adaptation
   Notes), and the one change that would raise the ceiling a tier.

## Gate rubric (brief-reviewer)

One verdict per pass, newest round on top in `scope.md`:

- **Constraint fit** — every seed idea runnable at tiny1m3m / seed 42 / <200 LoC?
  A topic whose effects only appear at scale is a `reject`.
- **Mineability** — does the seed list plausibly extend to 10+ ideas the
  closed.md doesn't already cover? Thin list = `revise`.
- **Falsifiability** — does each direction produce a clean WIN or NULL under
  the ctrl-bracket? "Interesting to explore" without a bar = `revise`.
- **Paper coherence** — do the expected results sum to the Paper claim?
- **Novelty** — overlap with `retired`/`rejected` briefs and `closed.md` ideas?

## Proposer inputs (read before proposing)

- `autoresearch/ideas/*/evidence.md` — what won, what nulled, and the Notes
  sections (follow-up threads live there).
- `autoresearch/closed.md` — exhausted ground.
- `litreview/synthesis.md` + `litreview/papers/*/digest.md` — what the field
  offers; digests ending in "Suggested action: file-idea" are topic fuel.
- `autoresearch/briefs/*/brief.md` — prior campaigns (don't re-pitch retired
  scopes; `shelved` candidates may be refreshed and re-filed).
- The fixed constraint block — never propose around it.

Proposer files **3 candidates per pass** so the blessing is a choice, not a
rubber stamp.

## Step 0 — campaign gate (proposer, every pass)

```bash
grep -l "status: active" autoresearch/briefs/*/brief.md
```

- Active brief exists and exit criteria unmet → `SKIP` (nothing to do).
- Active brief's exit criteria met → flip it `retired`, touch
  `paper-request` (paper pipeline), then propose the next 3 candidates.
- No active brief, candidates at `needs-blessing` → `SKIP` (waiting on human).
- No active brief, no candidates → propose 3.

## Claim protocol

Same as ideas — never hand-edit frontmatter:

```bash
autoresearch/bin/flip-brief.sh <brief-slug> <new-status> <agent> "<note>" [round]
```

## Agent → prompt map

| Agent | Prompt | Greps | Writes |
|---|---|---|---|
| proposer | `autoresearch/prompts/brief-proposer.md` | `needs-repitch`, then step-0 gate | `brief.md` ×3 (`needs-scope`) |
| brief-reviewer | `autoresearch/prompts/brief-reviewer.md` | `needs-scope` | appends `scope.md`, ranks accepted set, flips status |
| human (Vuk) | — | `needs-blessing` | flips one to `active`, copies body to `autoresearch/brief.md`, flips rest to `shelved` |
