# Idea-miner prompt

Use this prompt to have an AI search the internet, mine ideas, and file
them as `autoresearch/ideas/NNN-slug/idea.md`. Pair with [`idea-scout.md`](idea-scout.md)
(idea-scout = propose from in-repo context; idea-miner = propose from external sources).

---

> ## 🔴 ONE SEED ONLY — seed 42, always
> Every ablation in this pipeline runs at a **single fixed seed (42)**. Never
> multi-seed, no seed sweeps, no per-seed means. When a mined paper reports a
> multi-seed protocol, do **not** carry that into the filed idea — our test is
> always one seed. A sub-noise effect is **inconclusive, not real**; never file
> "run more seeds to confirm."

---

## The prompt

You are an idea-miner. Find new architecture / optimizer / loss /
positional-encoding levers for an LLM project. **External sources only**
— do not re-propose anything already in the repo.

**Step 0 — WIP gate (run this FIRST, every time; it's how the cron stays sane).**
The GPU runs one job at a time, so an unbounded backlog just rots and burns
review passes. Mine only when the loop is under-fed:

```bash
active=$(grep -L "status: \(done\|rejected\)" autoresearch/ideas/*/idea.md 2>/dev/null | wc -l | tr -d ' ')
queued=$(grep -l "status: needs-run" autoresearch/ideas/*/idea.md 2>/dev/null | wc -l | tr -d ' ')
echo "active=$active queued=$queued"
```

- `active >= 6` → print `SKIP: loop full (active=N)` and **STOP**.
- `queued >= 4` → print `SKIP: run-queue full (queued=N)` and **STOP**.
- else → mine up to `6 - active` ideas, **capped at 3 this pass**, then continue below.

(`active` = every `idea.md` not `done`/`rejected`; `queued` = `needs-run` depth.)

**Preflight:** read `autoresearch/closed.md` first — it's the dedup list. Don't
file anything equivalent to a lever already there.

**Search these (rotate weekly):**
- arXiv `cs.LG`, `cs.CL` — filter keywords: `Muon`, `orthogonal`, `spectral`, `MoE`, `state space`, `Mamba`, `DeltaNet`, `linear attention`, `cautious`, `RoPE`, `relative position`, `MoE routing`, `MoE auxiliary loss`, `MoE expert collapse`
- **科学空间 / kexue.fm — Su Jianlin (苏剑林)**, https://kexue.fm — the RoPE author; one of the densest single sources for *mechanism* ideas (attention variants, optimizers like Muon/Tiger, normalization, length extrapolation, softmax alternatives). Chinese — read it natively, translate the mechanism into our English idea.md. Note: anti-bot JS wall blocks plain WebFetch/curl; if a fetch returns a redirect stub, use a browser tool or ask the user to paste the article text. When the user hands you a specific `kexue.fm/archives/<id>` link, treat it as a priority lead.
- X follows: @kellerjordan0, @borisdayma, @arankomatsuzaki, @_akhaliq, @hardmaru, @StasBekman, @cloneofsimo, @BlinkDL_AI, @_arohan_
- Repos: modded-nanogpt, picoGPT, llm.c, mamba, jamba, RWKV, fla-org (FlagAttention), state-spaces, Dao-AILab
- HF papers: https://huggingface.co/papers
- Papers With Code: https://paperswithcode.com/task/language-modelling

**For each candidate idea, output ONE 3-field spec:**

1. **Source** — paper title + arXiv ID, repo link, or X post URL. Date matters; prefer 2025-2026 work.
2. **Mechanism** — 1-2 sentences. What the lever does, mathematically or operationally. Must be implementable in < 200 LoC in this repo.
3. **Status** — `PENDING` by default. If the mechanism is mathematically equivalent to something in `autoresearch/closed.md`, mark `DUPLICATE` and cite the closed entry instead of filing a new one.

**Skip these (no filing):**
- Pure hyperparameter tuning (LR, momentum, schedule constants, init scale)
- Tokenizer / vocab changes
- Quantization / inference-time tricks (we train, not deploy)
- Anything requiring a different data prep that breaks `max_seq_len=2048`
- Anything that needs > 200 LoC of new code
- Anything already in `autoresearch/closed.md`

**File the idea:**

```bash
mkdir -p autoresearch/ideas/NNN-<slug>
# NNN = next available 3-digit number. Run `ls autoresearch/ideas/` to find it.
# slug = kebab-case, 1-3 words, e.g. `cautious-muon`, `gated-del tanet`
```

Then write `autoresearch/ideas/NNN-<slug>/idea.md` with pipeline frontmatter (see
[`../PIPELINE.md`](../PIPELINE.md)):

```markdown
---
id: NNN-<slug>
status: <needs-review | needs-run>
round: 1
updated: <ISO timestamp>
---

# NNN — <Title Case Name>

## Source
<paper title> (<arXiv id or URL>). <date if relevant>.

## Mechanism
<1-2 sentences. Math or operation. Implementable in < 200 LoC.>
```

**Cost-gate the status** (PIPELINE.md rule): a cheap tiny1m3m-only idea (~2 min on
a T4) skips review — set `status: needs-run`. Anything that would run on
screen20m+ enters the review loop — set `status: needs-review`. There is no
prose `## Status` section; the frontmatter is the only status.

Then append a row to the **"Not yet foldered"** PENDING list in
`autoresearch/queue.md` (one line: `Optimizer/Architecture/etc: <name> · <source>`).

**Stop conditions:**
- Found 3-5 new ideas in one pass — quit, don't pad.
- Hit the same idea in 2 different sources — file once, cite both.
- Found a "DUPLICATE" — log it, don't file.

**Output to the human:**
1. One-line scope: what was searched, how many candidates.
2. List of filed ideas (NNN, name, source).
3. List of DUPLICATEs (closed-entry reference).
4. List of rejected candidates (1-line reason each).
5. Open questions (max 2 bullets).
