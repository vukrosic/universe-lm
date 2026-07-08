# Task board — beads

The lab's task tracker is [beads](https://github.com/gastownhall/beads) (`bd`).
Source of truth: the local bd database; the public snapshot lives at
`.beads-export/issues.jsonl` (one JSON object per issue) and is rendered live
on the lab dashboard.

Workflow:
- agents/humans: `bd ready` → pick a task → `bd update <id> --status in_progress --assignee <you>`
- done: `bd close <id> --reason "..."`
- publish: `bd export -o .beads-export/issues.jsonl && git add -A && git commit -m "tasks: sync" && git push`

Contributor credit is via accepted PRs (see CONTRIBUTING.md); the board is the
map, PRs are the territory.

---

## Open asks (no GPU needed)

These are human-readable, claimable asks. Your **first** task is always the standing
one below — propose an idea. The others are concrete, already-scoped cells. Every ask
ends in **one figure** and every claim carries a **citation** — an unsourced number
is not acceptable.

### ⭐ Standing ask (always open): propose one idea
Read a bit of the literature (AI-assisted search/reading encouraged) and propose **one
idea** to improve pretraining. Deliverable, as a short PR to `autoresearch/`:
1. **the idea** — one paragraph,
2. **the citation** — the paper or repo it comes from,
3. **`plain:`** — a one-line plain-language hypothesis,
4. **the test** — the smallest experiment + the one figure that would confirm/kill it.

Never "reproduce the baseline", never a bare hyperparameter sweep. Reviewed every Sunday;
accepted ideas enter the lab queue and you get first claim on running one.

### 📊 Ask: propose the data mixture (argument + citations → one figure)
The single biggest un-pulled lever here is **data**, and the open question is not *which*
datasets but in *what proportion* to make a 135M model smarter at **math**, **code**, and
**knowledge/reasoning** at a fixed token budget. Write an argument, backed by citations, for
a concrete mixture.

- **Baseline to argue against:** SmolLM2-135M's own published mixture and ablations
  (SmolLM2 paper, arXiv:2502.02737) — it is the model we're trying to beat, so its recipe
  is the thing to improve on, not copy.
- **The candidate datasets (cite each):** knowledge/reasoning web — FineWeb-Edu
  (`HuggingFaceFW/fineweb-edu`), DCLM-baseline (`mlfoundations/dclm-baseline-1.0`),
  Cosmopedia v2 (`HuggingFaceTB/smollm-corpus`); math — FineMath-4plus
  (`HuggingFaceTB/finemath`), optional OpenWebMath / Proof-Pile-2; code — Python-edu
  (`HuggingFaceTB/smollm-corpus`), optional StarCoder2 / The-Stack-v2.
- **Deliverable:** a short write-up + **one figure** — the proposed mixture as a labeled
  bar/pie (percent per source) with a one-line, cited rationale per slice, and a sentence
  on which eval each slice is meant to move (ARC/HellaSwag ← edu web, GSM8K ← math,
  HumanEval/MBPP ← code). Argument first; running the mix comes later, on a GPU.
- **Why it's real:** it directly sets Phase 1 (Data) of the compute plan — the mixture we
  actually train is the one this ask lands on.
