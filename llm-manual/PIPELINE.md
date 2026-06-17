# The Pipeline — how a claim earns its place

> ## 🔒 HARD RULE — operator approval is the ONLY gate into the measured ledger
> _Set by Vuk, 2026-06-17. No exceptions, no self-promotion. Unchanged when the ledger was
> folded into the LLM Manual — it governs **measured** `L###/D###/C###` entries only; the
> citation-gated field rules (`FM-*`) are a separate surface the assistant may write freely._
>
> 1. The assistant may create or edit measured-rule files **ONLY** in `drafts/`.
> 2. A numbered ledger file (`D###`, `L###`, `C###` — placed in its topical section folder)
>    may be created or
>    modified **only after Vuk explicitly approves that specific promotion.** Clearing the
>    numeric gates below makes a draft *eligible* for approval — it does **not** authorize
>    the write. Eligible ≠ approved.
> 3. "It was already in the README index," "the evidence clearly supports it," and
>    "I'm just materializing a planned entry" are **NOT** approval. Only Vuk saying so is.
> 4. Any ledger file that was not explicitly approved is to be **deleted** (recoverable
>    from git history if later approved).
>
> Workflow: assistant writes/updates a `DRAFT-*.md`, reports it as eligible → Vuk approves →
> *then* the assistant (or Vuk) creates the numbered entry.

Two surfaces, one-way gates. This file is the *process*; the entries are the *product*.

- **`drafts/`** — the bench. Mutable, unnumbered, **not citable**. Where an idea lives
  until measurement decides its tier. Rename or delete freely.
- **the section folders** (`01-…/` … `07-…/`) — the ledger. Approved measured entries
  (`L###/D###/C###`) live here beside the field rules they test. Numbered, citable,
  **append-only identity**: never renumber, never re-letter an existing file (links are
  permanent — see README).

An idea crosses from bench to ledger only by clearing an objective gate. That crossing is
the "strict system of entering" — it is the whole point of this file.

## Lanes — a claim's life

```
DRAFT ─┬─ admit as → C###   conjecture        believed, untested in-scope (a citable to-do)
       ├─ admit as → L###?  law (tentative)   measured, direction clear, not conclusive
       ├─ admit as → L###!  law (strong)      paired, ≥3 seeds, clears the noise floor
       └─ admit as → D###   derivation        deductive from stated assumptions

in-ledger moves:
   L###?  ── upgrade ──→ L###!     same file, same number; only the Status line changes
   any    ── demote  ──→ DEAD      its own Falsifier fired in-scope; mark, keep the file
   C###   ── confirm ──→ new L###  cross-letter never renames; C gets "→ promoted to L###"
```

A draft is normally admitted **straight to the tier its evidence already supports**. Don't
park something as `C` if you intend to measure it next week — keep it a draft. `C` is for
beliefs you are deliberately *not* scheduling yet but want a stable number to cite.

## Gates — objective, numeric

Noise constants (from [L001](L001-noise-floor.md)): unpaired val-loss SD ≈ **0.015**;
screen band = **0.02**. Measurement gate = paired differencing ([D001](D001-paired-variance-reduction.md)).

> **"Clears the noise"** := paired `|Δ| ≥ 0.02` **AND** 95% CI excludes 0.

| Promotion        | Gate — **all** must hold |
|------------------|--------------------------|
| draft → **C**    | falsifiable one-sentence statement · stated scope · named falsifier · a mechanism. *No data required.* |
| draft → **L?**   | ≥1 in-scope **paired** Δ, sign consistent across what was run, but NOT clearing noise (single seed, or CI straddles 0, or `|Δ|<0.02`). |
| draft → **L!**   | **paired, ≥3 shared seeds**, all same sign, **clears the noise**. |
| draft → **D**    | a proof from stated assumptions; any numbers only confirm an *assumption*, not the claim. |
| **L? → L!**      | re-measure to ≥3 seeds that clear the noise. |
| any → **DEAD**   | one in-scope result satisfies the entry's own **Falsifier**. Demote (not delete); note what killed it. |

## Admission checklist (the strict part)

A draft **cannot** enter the ledger until it carries all six fields the README mandates
(Statement · Status · Scope · Evidence/Derivation · Falsifier · Links). Practical bars:

- **No Falsifier → stays a draft.** A claim you can't kill isn't a law.
- **Out-of-scope evidence → at most a `C`.** Measured elsewhere ≠ measured here.
- **One claim per file.** A draft bundling two effects splits into two on admission.
- **Numbers carry uncertainty.** A Δ with no CI / seed count is not evidence yet.

## Scope discipline

A law is true **only inside its Scope line**. The same claim in a new regime (size,
tokens, context, or a different base recipe) is a **separate measurement**: either
re-measure and *widen* the Scope, or open a fresh draft `"does X still hold under Y"`.
**Never silently extend scope** — that is the most common way a ledger rots.

## Numbering

Per-tier counters in order of entry: `D###`, `L###`, `C###`. Never renumber, never
re-letter an existing file. Drafts are slug-only (`DRAFT-slug.md`) and carry no permanent
identity until admitted.

## Status board

Everything in flight. Update on every admission / upgrade / kill.

_Lane "**draft (await-approval)**" = the evidence clears its numeric gate and the claim is
eligible, but it is NOT in the ledger until Vuk approves the promotion (🔒 HARD RULE)._

| ID / draft | Claim (short) | Lane | Waiting on |
|---|---|---|---|
| _cand_ paired-variance | Paired seeds cancel common-mode noise | **draft (await-approval) → D** | Vuk's approval to number as D001 |
| _cand_ noise-floor | Seed-noise floor ≈ 0.015 SD, band 0.02 | **draft (await-approval) → L!** | Vuk's approval to number as L001 |
| _cand_ concave-penalty | Concave penalty loses to linear | **draft (await-approval) → L!** | Vuk's approval |
| _cand_ alibi-shape-invariance | ALiBi shape invariance (curvature irrelevant) | **draft (await-approval) → L!** | Vuk's approval; scope-extension note under deepnet |
| _cand_ distance-penalty-load-bearing | Growing distance penalty is load-bearing (+0.155) | **draft (await-approval) → L!** | Vuk's approval; longer-context re-test |
| _cand_ slope-init-reliability | ALiBi slope init = reliability knob | **draft (await-approval) → L!** (mean L?) | Vuk's approval; mean claim needs Δ clearing 0.02 w/ CI≠0 |
| _cand_ value-residual-no-stack | Value-residual gives no gain atop ALiBi | **draft (await-approval) → L?** | Vuk's approval; 3-seed paired confirm of the null |
| _cand_ sub-threshold-compound | DeepNet-α + poly-ALiBi compound past the floor | **draft (await-approval) → L!** (323 super-add = L?) | Vuk's approval; 323 n=3 CI straddles 0 |
| _draft_ single-seed-inflation | 1-seed screen-win over-estimates honest Δ by ≈1 SD | **draft → L?** | a 3rd inflation case (shrink-CI off 0) for L! |
| _draft_ tiny-update-starvation | tiny1m3m update-starved; LR×mom compound super-additively | **draft → L?** | ≥5-seed re-run of 323 to move the magnitude CI off 0 (n=3 CI straddles 0) |
| _draft_ identical-construction-differencing | paired Δ measures the lever only if both arms built identically | **draft → D** | a numbered slot (deductive; 0.0-nonsense-flag control passes) |
| _draft_ flag-inertness-on-champion | ~152/~200 `use_*` flags step-0 inert on ALiBi champion | **draft → L?** | firing×band-clearance table to quantify the inert⇒sub-band link |
