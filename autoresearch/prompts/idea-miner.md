# Idea-miner prompt

Use this prompt to have an AI search the internet, mine ideas, and file
them as `autoresearch/ideas/NNN-slug/idea.md`.

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

> ## 🔴 THE GPU MUST NEVER BE IDLE — you are the queue's water source
> You are the head of the pipe. If the GPU ever idles for lack of `needs-run`
> ideas, the root cause traces back to the upstream loop running dry — and you
> feed it. Keep enough ideas flowing that the downstream gates always have
> something to advance toward `needs-run`. When the WIP gate says mine, an idle
> or thin GPU queue means mine the **full** allowance — never under-fill. See the
> prime directive in [`../PIPELINE.md`](../PIPELINE.md).

> ## 🔴 YOU RUN UNATTENDED — ACT, DON'T ASK
> This prompt fires from a cron with no human watching. Your job is to **file
> ideas**, not to report a plan and wait for approval. Never end a pass with a
> question like "Want me to mine N more?" — if the gate says mine, mine the full
> allowance this pass, then stop. The only valid no-file outcome is the gate
> printing `SKIP`. If you find yourself about to ask the human whether to
> proceed: the answer is always yes, so proceed.

**Project context (read first):**
- Repo: `/Users/vukrosic/my-life/llm-research-kit-scaling`.
- Goal: find mechanisms that lower val loss, screened cheaply — **in service of
  the program in `plans/beat-smollm2-135m.md`**: levers screened here feed a
  recipe that must survive the 10M → 37M → 135M ladder and beat SmolLM2-135M's
  released checkpoint. Prefer levers with **published evidence at 100M+ scale**;
  a lever only ever demonstrated at toy scale is a worse bet at equal cost.
- **🔴 ONE TIER ONLY — `tiny1m3m` (0.94M params · 3M tokens), seed 42.** Every
  idea in this pipeline is run *only* at tiny1m3m. Do not propose, scope, or
  reference `screen20m`, the full ladder, or any larger tier — those are out of
  scope. An idea that only pays off at larger scale doesn't belong here.
- Only **mechanisms / structural changes**, no hyperparameter tuning. Must be
  **transferable** across scale and **identity/zero-init** (step-0 ≈ baseline)
  unless explicitly noted.
- **Coordination:** another Claude implements other research in parallel.
  Before editing `models/layers.py` or `configs/llm_config.py`, `git diff` and
  `git status` for conflicts. Never rebase or push.

**Step 0 — WIP gate (run this FIRST, every time; it's how the cron stays sane).**
The cap is on the **upstream** pipeline — ideas still being *worked on* before the
GPU (taste → definition → code). Once an idea reaches the GPU queue
(`needs-run`/`running`) it's effectively shipped; it no longer counts, so the GPU
backlog can be arbitrarily deep without ever blocking mining. Mine only when the
*upstream* loop is under-fed:

```bash
# upstream = not terminal AND not yet handed to the GPU
upstream=$(grep -L "status: \(done\|rejected\|needs-run\|running\)" autoresearch/ideas/*/idea.md 2>/dev/null | wc -l | tr -d ' ')
gpu=$(grep -l "status: \(needs-run\|running\)" autoresearch/ideas/*/idea.md 2>/dev/null | wc -l | tr -d ' ')
echo "upstream=$upstream gpu=$gpu"
```

Gate on `upstream` only. Compute your allowance, then mine **exactly that many** —
no asking, no stopping early:

- If `upstream >= 6` → print `SKIP: upstream full (upstream=N, gpu=M)` and **STOP**.
- Otherwise mine `N = min(3, 6 - upstream)` ideas this pass — capped at 3 so one
  tick stays bounded; the cron runs again to top up. File all `N` at
  `needs-taste`; do not file fewer and ask whether to continue.

(`upstream` = every `idea.md` whose status is *not* `done`/`rejected`/`needs-run`/
`running` — i.e. `needs-taste`, `needs-repitch`, `needs-review`, `needs-revision`,
`needs-plan`, `needs-codereview`, `needs-recode`, and their `-ing` locks. Ideas on
the GPU don't count.)

Run the **re-pitch queue** (below) before mining — a sent-back idea is already an
upstream slot and is closer to shipping than a cold one.

**Preflight:** read `autoresearch/closed.md` first — it's the dedup list. Don't
file anything equivalent to a lever already there.

**Search these (rotate weekly):**
- **`plans/litrev-sub200m.md` § "recipe levers that mattered"** — the standing
  internal source (sole exception to the external-only rule): levers that
  repeatedly showed up in winning sub-400M recipes. Check it every pass; an
  unfiled lever from this list outranks a cold external find.
- **USE the `mcp__bing-search__bing_web_search` tool** for every query — do not stop at zero searches. If a tick returns 0 candidates, the next tick queries more keywords. Multiple queries per pass are fine; do not cap yourself at 1.
- arXiv `cs.LG`, `cs.CL` — filter keywords: `Muon`, `orthogonal`, `spectral`, `MoE`, `state space`, `Mamba`, `DeltaNet`, `linear attention`, `cautious`, `RoPE`, `relative position`, `MoE routing`, `MoE auxiliary loss`, `MoE expert collapse`
- **科学空间 / kexue.fm — Su Jianlin (苏剑林)**, https://kexue.fm — the RoPE author; one of the densest single sources for *mechanism* ideas (attention variants, optimizers like Muon/Tiger, normalization, length extrapolation, softmax alternatives). Chinese — read it natively, translate the mechanism into our English idea.md. Note: anti-bot JS wall blocks plain WebFetch/curl; if a fetch returns a redirect stub, use a browser tool or ask the user to paste the article text. When the user hands you a specific `kexue.fm/archives/<id>` link, treat it as a priority lead.
- X follows: @kellerjordan0, @borisdayma, @arankomatsuzaki, @_akhaliq, @hardmaru, @StasBekman, @cloneofsimo, @BlinkDL_AI, @_arohan_
- Repos: modded-nanogpt, picoGPT, llm.c, mamba, jamba, RWKV, fla-org (FlagAttention), state-spaces, Dao-AILab
- HF papers: https://huggingface.co/papers
- Papers With Code: https://paperswithcode.com/task/language-modelling

**For each candidate idea, output ONE 4-field spec:**

1. **Source** — paper title + arXiv ID, repo link, or X post URL. Date matters; prefer 2025-2026 work.
2. **Mechanism** — 1-2 sentences. What the lever does, mathematically or operationally. Must be implementable in < 200 LoC in this repo.
3. **Scale evidence** — the largest scale the source demonstrated gains at
   ("showed +X at 125M/1B/7B" or "toy-scale only"), and a `transfer-risk`
   call: `low` (gains shown ≥100M), `med` (gains shown 10–100M or strong
   mechanistic argument), `high` (toy-scale only or mechanism plausibly
   exploits tiny-model artifacts). One-line justification required.
4. **Status** — `PENDING` by default. If the mechanism is mathematically equivalent to something in `autoresearch/closed.md`, mark `DUPLICATE` and cite the closed entry instead of filing a new one.

**Skip these (no filing):**
- Pure hyperparameter tuning (LR, momentum, schedule constants, init scale)
- Tokenizer / vocab changes
- **Data-mix / LR-schedule / tokenizer levers are NOT tiny1m3m ideas** — but
  don't drop them: append one line to the "Scale-tier backlog" section of
  `autoresearch/queue.md` (`<lever> · <source> · needs ≥10M tier`). They get
  tested on the 10M+ ladder by the program, not by this pipeline.
- Quantization / inference-time tricks (we train, not deploy)
- Anything requiring a different data prep that breaks `max_seq_len=2048`
- Anything that needs > 200 LoC of new code
- Anything already in `autoresearch/closed.md`
- A model-shape change (depth/width swap, tied-embeddings removal) **without** an
  explicit transfer argument — those are usually HP re-allocations, not mechanisms

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
status: needs-taste
round: 1
updated: <ISO timestamp>
transfer-risk: <low|med|high>
---

# NNN — <Title Case Name>

## Source
<paper title> (<arXiv id or URL>). <date if relevant>.

## Mechanism
<1-2 sentences. Math or operation. Implementable in < 200 LoC.>

## Scale evidence
<largest scale the source showed gains at, and the one-line justification for
the frontmatter transfer-risk tag. "Toy-scale only" is a valid entry but makes
the taste gate's transfer criterion hard to pass.>

## Why it's worth a slot
<the bet, in one sharp sentence: we expect X because Y. The leverage, and why a
null result would still be informative. This is what the taste gate judges.>
```

**Always file at `status: needs-taste`** — the taste gate is the first stop for
*every* mined idea (see [`idea-taste.md`](idea-taste.md)). Don't pre-route to
`needs-review`/`needs-run`. Every idea runs at **tiny1m3m, seed 42** — there is no
tier to choose. There is no prose `## Status` section; the frontmatter is the only
status.

Then append a row to the **"Not yet foldered"** PENDING list in
`autoresearch/queue.md` (one line: `Optimizer/Architecture/etc: <name> · <source>`).

**Stop conditions:**
- Filed your `N`-idea allowance — quit, don't pad.
- Hit the same idea in 2 different sources — file once, cite both.
- Found a "DUPLICATE" — log it, don't file.
- A source is dry — move to the next; don't stop to ask. If sources run out
  before `N` is hit, file what you found and note the shortfall in the log.

**Output (a log, not a conversation — no questions, no approval requests):**
1. One-line scope: what was searched, how many candidates.
2. List of filed ideas (NNN, name, source, status).
3. List of DUPLICATEs (closed-entry reference).
4. List of rejected candidates (1-line reason each).

---

## Re-pitch mode (you are also the doer in the taste loop)

The taste-reviewer ([`idea-taste.md`](idea-taste.md)) sends ideas back when the
*bet* is dull, crowded, or vague. You revise them — same agent, no separate
reviser. **Run this queue at the start of every pass, before mining anything new**
(sharpening a live idea beats adding a cold one):

```bash
grep -l "status: needs-repitch" autoresearch/ideas/*/idea.md
```

For each hit, in order:

1. Read `taste.md` — the **latest** round's findings (top of file). Read the whole
   `idea.md` (note its `round`).
2. **Claim it**: `autoresearch/bin/flip.sh <idea> repitching miner "claimed"`.
3. Close **every** finding in the latest round: sharpen the `## Why it's worth a
   slot` bet, swap to a less-crowded family, name the leverage in one sentence, or
   pick a tier where a null result still teaches us. You may rewrite the
   mechanism/source if that's what the finding demands — but if the idea simply
   can't be made interesting, say so in the doc and let round 3 auto-reject it;
   don't pad it with a worse pitch.
4. **Release** with the bumped round as the 5th arg:
   `autoresearch/bin/flip.sh <idea> needs-taste miner "<k findings applied>" <round+1>`.
5. Next hit. Then mine new ideas only if Step 0's gate still says to.

Never hand-edit the frontmatter. One pass per claim — apply, bump, release; don't
re-judge your own pitch (that's the taste-reviewer's call).
