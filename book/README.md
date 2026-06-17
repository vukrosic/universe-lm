# The Book — rules that (try to) solve LLMs

A growing, scientifically rigorous ledger of what we actually know about how LLMs
work and how to train them better. North star: enough verified rules, stacked, that
the design of a good LLM becomes *derivable* rather than guessed.

We are honest about a hard fact: **almost nothing here is a theorem.** Most entries are
*empirical laws measured at a stated scale*. Rigor comes not from pretending otherwise
but from making every claim carry its evidence, its scope, and its kill-switch.

## Epistemic tiers (every entry declares one)

- **D — Derivation.** Follows deductively from math/assumptions. As certain as its
  assumptions. (e.g. why paired seeds reduce variance.)
- **L — Empirical law.** Measured. Tagged **strong** (paired, multi-seed, clears the
  noise) or **tentative** (suggestive, not yet conclusive). Always scope-limited.
- **C — Conjecture.** Believed but untested here. A to-do with a hypothesis attached.

## The rule of the book

Every entry must state:
1. **Statement** — one sentence, falsifiable.
2. **Status** — tier + strength.
3. **Scope** — the exact regime it was shown in (model size, tokens, context, etc.).
   A law outside its scope is a conjecture until re-measured.
4. **Evidence / Derivation** — numbers with uncertainty, or the proof.
5. **Falsifier** — the single result that would kill or bound it.
6. **Links** — experiments / records that back it.

No entry is "true." Each is "survived testing so far, here, under these conditions."
Upgrade tentative→strong with more seeds; demote anything a new result breaks.

## How an entry is born

Ideas do not get written straight into the ledger. They start on the bench in
[`drafts/`](drafts/) — unnumbered, mutable, not citable — and cross over only by clearing
an objective gate **AND receiving explicit operator approval** (the 🔒 HARD RULE at the top of
[PIPELINE.md](PIPELINE.md) — the assistant may only write to `drafts/`; numbered entries
require Vuk's sign-off). The gates, the lanes (draft → C / L? / L! / D), the demotion rules,
and the live status board are in **[PIPELINE.md](PIPELINE.md)**. Templates:
[`_TEMPLATE-entry.md`](_TEMPLATE-entry.md), [`drafts/_TEMPLATE-draft.md`](drafts/_TEMPLATE-draft.md).

## Index

_The ledger is **empty** — no entry has been operator-approved yet._

Candidate claims live on the bench in [`drafts/`](drafts/). A set of well-evidenced
foundational candidates (the former D001 + L001–L007: paired-variance gate, noise floor,
concave-penalty-loss, ALiBi shape invariance, distance-penalty load-bearing, slope-init
reliability, value-residual non-stack, sub-threshold compounding) was drafted from existing
evidence and is **awaiting approval** before it can be numbered (text recoverable from git
history, commit 39ecf04).

## Naming

`D###` derivations · `L###` laws · `C###` conjectures. Number in order of entry; never
renumber (links must stay stable). One claim per file.
