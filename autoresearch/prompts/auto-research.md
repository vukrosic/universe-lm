# Auto-research prompt (single pass, end to end)

One prompt that does the whole loop: **find one idea → review it → plan it →
implement it → run it → judge it → log it.** No separate stages, no handoffs.

---

> ## 🔴 YOU RUN UNATTENDED — ACT, DON'T ASK
> This fires from a cron with no human watching. Do the work and write the
> result down. Never end by asking "want me to continue?" — the answer is always
> yes. The only valid stop is when the pass is finished or a hard rule says stop.

> ## 🔴 THE FIXED TEST — never change it
> Every experiment runs at **tiny1m3m (0.94M params · 3M tokens), one seed: 42.**
> No seed sweeps, no other scale. A sub-noise result is **inconclusive, not a
> win.** Never propose "run more seeds to confirm."

**Repo:** `/Users/vukrosic/my-life/llm-research-kit-scaling`
**Goal:** find a mechanism (architecture / optimizer / loss / positional
encoding) that lowers val loss vs the baseline. Mechanisms only — no
hyperparameter tuning, no tokenizer/data changes. Must be implementable in
**< 200 lines** behind a config flag.

Before you start: `git status` and `git diff` (another agent may be editing the
same files). Never rebase, never push.

---

## Step 1 — Search (find ONE idea)

Read `autoresearch/closed.md` first — it's the dedup list. Don't repeat anything
in it.

Search the internet for one new lever. Use the `mcp__bing-search__bing_web_search`
tool plus these sources:
- arXiv `cs.LG` / `cs.CL` — keywords: Muon, orthogonal, spectral, RoPE, relative
  position, linear attention, DeltaNet, normalization, softmax alternatives,
  cautious, MoE routing.
- kexue.fm (Su Jianlin) — dense source of mechanism ideas; read Chinese natively.
- X: @kellerjordan0, @borisdayma, @hardmaru, @cloneofsimo, @BlinkDL_AI.
- Repos: modded-nanogpt, llm.c, fla-org, state-spaces, Dao-AILab.

Prefer levers with **published gains at ≥100M scale** (lower transfer risk).
Pick the single best candidate. If you find a duplicate of something in
`closed.md`, skip it and pick another.

## Step 2 — Review the idea once

Sanity-check your pick in a few sentences. Kill it now if any of these fail:
- Is it actually a mechanism, not a hyperparameter in disguise?
- Is it < 200 LoC and identity/zero-init (step-0 ≈ baseline)?
- Is it genuinely new vs `closed.md`?
- Would a **null** result still teach us something?

If it fails, go back to Step 1 and pick another. If it passes, write it down as
`autoresearch/ideas/NNN-<slug>/idea.md` (NNN = next free 3-digit number):

```markdown
---
id: NNN-<slug>
status: planning
round: 1
updated: <ISO timestamp>
transfer-risk: <low|med|high>
---

# NNN — <Name>

## Source
<paper/repo/post + link>

## Mechanism
<1-2 sentences: the math/operation. < 200 LoC.>

## Why it's worth a slot
<the bet in one sentence: we expect X because Y; a null still tells us Z.>
```

## Step 3 — Plan it

Write `autoresearch/ideas/NNN-<slug>/plan.md`: exactly which files and functions
change (usually `models/layers.py` and/or `configs/llm_config.py`), the config
flag name (`use_<feature>`), how it stays zero-init at step 0, and how you'll
read the final val loss.

## Step 4 — Review the plan (only if it's non-trivial)

If the change is simple and obvious, skip this. If it touches the attention core,
the optimizer, or more than ~100 LoC, read the plan back once and check: does it
break the baseline path when the flag is off? does it stay within 200 LoC? Fix
the plan if not. Then move on — one review, no loop.

## Step 5 — Implement

Make the change behind the `use_<feature>` flag, off by default so the baseline
is untouched. Keep it minimal. Confirm it imports and the flag toggles the
behavior.

## Step 6 — Run it

Run the standard tiny1m3m / seed-42 training for the variant **and** the baseline
(follow the exact run convention in `prompts/runner.md` / `PIPELINE.md`). Read the
final val loss from each.

## Step 7 — Judge

```
delta = variant_val_loss − baseline_val_loss
|delta| ≤ 0.005   → NULL (inconclusive, within noise)
delta ≤ −0.01     → WIN
otherwise          → NULL
```

Write `autoresearch/ideas/NNN-<slug>/evidence.md` with the verdict, both numbers,
the delta, and one line on what it means. If WIN, note it as a candidate for the
recipe. If NULL, append a one-line entry to `autoresearch/closed.md` so it's never
re-tried.

## Step 8 — Log and stop

Set the idea's frontmatter `status:` to `done`. Print a short log: idea name,
source, verdict, the two numbers. Then stop — don't start a second idea.

---

**One pass = one idea, start to finish.** The cron runs you again for the next.
