# The LLM Manual — the rules for training a large language model

> **Goal.** One structured, cited reference for *everything known about how to train an LLM* —
> organized by the decisions you actually make, in the order they matter, each rule carrying
> its evidence, its scope, and what would kill it. Built so a city, a government, a company,
> or one person could read it top to bottom and train a good model on purpose rather than by
> guess. North star: enough verified rules, stacked, that a good LLM becomes *derivable*.

This is a single corpus with **two kinds of entry**, living side by side in each section:

| | **Field rules** (`FM-*`) | **Measured rules** (`L### / D### / C###`) |
|---|---|---|
| Source | The field — cited papers | **Us** — our own seed-42 experiments |
| Confidence axis | **E / C / F** (see below) | **D / L / C** (see below) |
| Who may write it | The assistant, freely (with a citation) | 🔒 **Operator approval only** (the hard rule) |
| Mutability | Edit freely; it's a living review | Append-only identity; never renumber |
| Answers | "What does the literature claim?" | "What did *we* actually reproduce?" |

The two are designed to be **in tension**. Field rules record what the literature claims;
measured rules record what *our box* confirmed or contradicted. Where they meet — a published
rule we tested ourselves — is the most valuable thing this manual can hold. Every `FM-*` rule
carries a **`Reproduced by us?`** line that points at the measured entry (or names the open
question); every measured entry links back to the field rule it tests.

## The two gates (both still apply)

1. **Field rules** need a **citation + a confidence tier**. The assistant may add/edit them.
2. **Measured rules** need **operator approval** to become numbered. This is the 🔒 HARD RULE
   (set by Vuk 2026-06-17), preserved verbatim at the top of [PIPELINE.md](PIPELINE.md):
   the assistant may write to [`drafts/`](drafts/) only; a numbered `L###/D###/C###` file is
   created **only after Vuk approves that specific promotion.** Folding the ledger into this
   manual did **not** weaken this. Eligible ≠ approved.

So: the assistant writes literature freely, and writes our experiments only as far as
`drafts/` — exactly as before, now under one roof.

## Confidence tiers

**Field rules — how solid is the *field's* claim:**
- **[E] Established** — replicated across labs *and* load-bearing in shipped models. A default.
- **[C] Contested** — real signal, but magnitude/optimum/scope genuinely disputed.
- **[F] Frontier** — one strong source, not yet broadly replicated. A bet, not a default.

**Measured rules — how solid is *our* evidence** (full gate in [PIPELINE.md](PIPELINE.md)):
- **D — Derivation** — deductive from stated assumptions.
- **L — Empirical law** — measured; `L!` strong (paired, ≥3 seeds, clears noise) or `L?` tentative.
- **C — Conjecture** — believed, untested in-scope; a citable to-do.

## What every entry states

Field rule: **Rule · Confidence · Scope · Evidence · Reproduced by us? · Falsifier · Sources.**
Measured rule: **Statement · Status · Scope · Evidence/Derivation · Falsifier · Links.**

No entry is "true." Each is "survived testing so far, here, under these conditions." Scope is
sacred — a rule quoted outside the regime it was shown in is folklore. Demote anything a new
result breaks. Templates: [`_TEMPLATE-field-rule.md`](_TEMPLATE-field-rule.md),
[`_TEMPLATE-measured.md`](_TEMPLATE-measured.md).

## Citation discipline (the rigor rule)

A citation pointing to the wrong paper is a scientific error, not a typo. So:

- Every `FM-*` rule ends with a **`_Reviewed: <date>_`** line. Sources whose title/authors/key
  number were checked against the primary source carry `— verified <date>`.
- A claim taken from a secondary summary (blog, search digest) but **not** confirmed against the
  primary source is marked as such inline (e.g. "the paper's claim, not re-verified here"). Do
  not launder a secondary summary into an apparent primary fact.
- Quantitative claims state the regime they were measured in. A number without a scope is
  folklore. When the manual repeats a *popular but unverified* explanation, it says so and
  names the verified alternative (see FM-01.1 on the Kaplan↔Chinchilla cause).

Entries created 2026-06-17 had their **primary, load-bearing** citations verified against arXiv
abstracts that day; secondary citations and exact in-paper numbers are flagged where not
re-checked. Re-verify before relying on any number for a real run.

## How it's organized

Markdown, one rule per file, nested by **the decision you're making** when you train — not by
chronology of papers. **Start at [00-MAP.md](00-MAP.md).**

## Index

- **[00-MAP.md](00-MAP.md)** — the decision spine: every choice in training an LLM, in order.
- **[01-scaling-laws/](01-scaling-laws/)** — spending compute: params vs tokens, repetition, vocab, emergence.
- **[02-data/](02-data/)** — what to train on: quality, dedup, mixture.
- **[03-architecture/](03-architecture/)** — the convergent recipe and why each piece exists.
- **[04-optimization/](04-optimization/)** — optimizer, schedule, batch/wd, μP transfer.
- **[05-stability/](05-stability/)** — not diverging: z-loss, qk-norm.
- **[06-knowledge-and-reasoning/](06-knowledge-and-reasoning/)** — the *Physics of LLMs* program.
- **[07-post-training/](07-post-training/)** — base model → assistant: SFT, DPO, RLHF.
- **[08-efficiency-and-scale/](08-efficiency-and-scale/)** — MoE, precision (BF16/FP8), long-context extension.
- **[drafts/](drafts/)** — the bench: our experiments awaiting measurement/approval (not citable).
- **[PIPELINE.md](PIPELINE.md)** — how a measured claim earns a number (the gate + status board).

## Our measured ledger — current state

The numbered ledger is **empty**: no measured entry has been operator-approved yet. Well-
evidenced candidates sit in [`drafts/`](drafts/) (paired-variance gate, noise floor,
concave-penalty, ALiBi shape invariance, distance-penalty, slope-init, value-residual,
sub-threshold compounding) — eligible, awaiting Vuk's approval. When approved, each lands as
`L###/D###/C###` **inside the relevant section folder**, next to the field rule it tests.

## Naming

`FM-<section>.<n>` for field rules (e.g. `FM-03.1`). `D###/L###/C###` for measured rules,
per-tier counters in order of entry — never renumber, never re-letter (links are permanent).
One rule per file. Confidence/dates change in place; identity does not.
