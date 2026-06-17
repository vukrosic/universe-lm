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
an objective gate. The gates, the lanes (draft → C / L? / L! / D), the demotion rules, and
the live status board are in **[PIPELINE.md](PIPELINE.md)**. Templates:
[`_TEMPLATE-entry.md`](_TEMPLATE-entry.md), [`drafts/_TEMPLATE-draft.md`](drafts/_TEMPLATE-draft.md).

## Index

- [D001 — Paired seeds cancel common-mode noise](D001-paired-variance-reduction.md) — why same-seed differencing is the measurement gate
- [L001 — The seed-noise floor at tiny scale](L001-noise-floor.md) — ~0.015 val-loss SD; sub-0.02 unpaired diffs are invisible
- [L002 — Concave distance penalties underperform linear](L002-concave-penalty-hurts.md) — kerple-log loses by 0.056 (paired, conclusive)
- [L003 — ALiBi shape invariance](L003-alibi-shape-invariance.md) — among growing penalties, curvature is irrelevant
- [L004 — A growing distance penalty is load-bearing](L004-distance-penalty-is-load-bearing.md) — ALiBi vs nothing = +0.155 (paired, the dominant lever)
- [L005 — ALiBi slope init is a reliability knob](L005-alibi-init-is-a-reliability-knob.md) — geometric seeding buys steadiness (~3×), not score
- [L006 — Value-residual does not stack on a distance penalty](L006-value-residual-no-stack-alibi.md) — a stand-alone win that vanishes inside the ALiBi recipe (tentative null)
- [L007 — Sub-threshold levers compound](L007-sub-threshold-levers-compound.md) — DeepNet-α and poly-ALiBi each miss the floor alone, clear it stacked (the champion)

## Naming

`D###` derivations · `L###` laws · `C###` conjectures. Number in order of entry; never
renumber (links must stay stable). One claim per file.
