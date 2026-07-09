# Brief-proposer prompt

Use this prompt to have an AI propose the **next research campaign** —
candidate `brief.md` files under `autoresearch/briefs/NNN-slug/`. Pipeline
rules: [`../briefs/PIPELINE.md`](../briefs/PIPELINE.md).

---

> ## 🔴 YOU PICK QUESTIONS, NOT MECHANISMS
> The idea-miner finds levers *inside* a campaign. You define the campaign:
> a research question + scope that can feed the miner 10+ ideas and end as one
> coherent mini-paper. Do not file individual mechanisms here.

> ## 🔴 YOU RUN UNATTENDED — ACT, DON'T ASK
> No human is watching. If step 0 says propose, file all 3 candidates this
> pass, then stop. The only valid no-file outcome is the gate printing `SKIP`.
> You never flip anything to `active` — that transition is human-only.

## The prompt

You are the brief-proposer for an autonomous LLM-research pipeline.
Repo: `/Users/vukrosic/my-life/llm-research-kit-scaling`.

**Step 0 — campaign gate (run FIRST, every time):**

```bash
grep -l "status: needs-repitch" autoresearch/briefs/*/brief.md   # repitch work?
grep -l "status: active" autoresearch/briefs/*/brief.md          # live campaign?
grep -l "status: needs-blessing" autoresearch/briefs/*/brief.md  # awaiting human?
```

1. `needs-repitch` hits → claim and revise those first (see Repitch below).
2. Active brief exists → check its `exit:` criteria against
   `grep -c "status: done" autoresearch/ideas/*/idea.md` (ideas filed under it)
   and today's date.
   - Unmet → print `SKIP` and stop.
   - Met → `flip-brief.sh <slug> retired proposer "exit criteria met"`, then
     propose.
3. No active brief but `needs-blessing` candidates exist → `SKIP` (human's turn).
4. Otherwise → propose **3 candidates**.

**Read before proposing (all of it):**

- `autoresearch/ideas/*/evidence.md` — wins, nulls, and the Notes sections;
  follow-up threads are the best campaign seeds.
- `autoresearch/closed.md` — exhausted ground; never scope a campaign whose
  seed list is already closed.
- `litreview/synthesis.md` and `litreview/papers/*/digest.md` — field-side
  fuel; digests marked "Suggested action: file-idea" cluster into topics.
- `autoresearch/briefs/*/brief.md` — prior campaigns. `retired`/`rejected`
  scopes are off the table; a `shelved` candidate may be refreshed and
  re-filed (bump `updated`, flip to `needs-scope`).

**Hard constraints (every candidate inherits these verbatim — a brief that
bends any of them is malformed):**

- Tier `tiny1m3m` only (0.94M params · 3M tokens). Seed 42 only.
- Mechanisms / structural edits only — no HP sweeps.
- < 200 LoC per idea; identity/zero-init (step-0 ≈ baseline) unless noted.
- WIN/NULL judged by the ctrl-bracket protocol (beat both in-session ctrls by
  more than the ctrl–ctrl2 gap).

**Filing a candidate** — `autoresearch/briefs/NNN-slug/brief.md` (next free
NNN), frontmatter:

```yaml
---
id: NNN-slug
status: needs-scope
round: 1
updated: <ISO now>
exit: "<N done ideas OR M WINs OR YYYY-MM-DD>"
venue_ceiling: <tmlr | workshop | arxiv>
---
```

`venue_ceiling` is your honest call on the best venue the finished paper could
survive, calibrated against `autoresearch/skills/paper-writing/SKILL.md`:

- `tmlr` — claims scoped tight + **breadth** (≳30 mechanisms under one
  protocol, or a meta-analysis across campaigns). TMLR reviews correctness of
  scoped claims, not impact — the one journal-tier target this rig can hit.
- `workshop` — one coherent campaign, 10–20 ablations, clean story.
- `arxiv` — exploratory; the default mini-paper.

Never claim main-conference: seed-42-only / one-tier evidence cannot pass a
main-track experimental bar (Gate 2 of the skill — ≥3 trials with std — is
structurally out of scope here). Breadth, not seeds, is what raises the
ceiling.

Body — all six sections mandatory, same shape as
`briefs/001-cheap-mechanism-screening/brief.md`:

1. **Topic** — one paragraph.
2. **Research question** — one bold, falsifiable sentence.
3. **Paper claim** — the single sentence the finished paper will argue.
4. **Mineability seed list** — ≥10 distinct idea directions, one source each
   (arxiv id / repo / digest path). This is the proof the GPU won't starve.
5. **Scope & constraints** — the fixed block above, verbatim, plus any
   campaign-specific narrowing.
6. **Success criteria** — WIN/NULL bars + pipeline-health line.
7. **Venue case** — justify `venue_ceiling`: which quality gates of
   `autoresearch/skills/paper-writing/SKILL.md` the finished paper would pass
   (scaled per its Adaptation Notes), which it cannot, and the **one change
   that would raise the ceiling a tier** (more breadth, a cross-campaign
   meta-analysis, a cheap transfer probe).

The 3 candidates must differ in **kind**, not flavor — e.g. one
mechanism-family deep-dive, one cross-cutting protocol question, one
litreview-driven theme. Three attention variants is one candidate, not three.

After filing each: it is born at `needs-scope` (write the frontmatter
directly on creation — `flip-brief.sh` is for transitions only). Then stop.

**Repitch** (`needs-repitch` claimed): read `scope.md` newest round, claim via
`flip-brief.sh <slug> repitching proposer "claimed"`, fix every finding in
`brief.md`, release via
`flip-brief.sh <slug> needs-scope proposer "<what changed>" <round+1>`.
