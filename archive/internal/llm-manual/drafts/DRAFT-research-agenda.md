# DRAFT — Research Agenda (the questions the ledger exists to answer)

> Status: **draft / bench**. This is the map, not a claim. It is eligible to become a
> top-level `book/AGENDA.md` (not a numbered D/L/C entry) only on operator approval.
> Questions are not falsifiable claims — they do not get D/L/C numbers. Each *answer*
> does, as an `L` (measured) or `D` (derived) entry. A question here is "open" until a
> ledger entry closes it; we record which entry closed it next to the question.

## Mission

Find **structural** mechanisms (attention, positional, normalization, FFN, loss, residual
routing — RULE 0: never optimizer/LR/batch/wd) that lower validation loss on a fixed tiny
LM (`Tiny1M3MAlibiConfig`, ~1–3M params, 3M-token budget), establish *why* they work, and
determine **which findings are real and which transfer to larger scale**. The deliverable is
not a model — it is a ledger of scoped, falsifiable rules that compound.

Every answer carries a **scope clause** (what model / data / champion / token-budget it was
measured under). A Δ without a scope is folklore, not science.

---

## G1 — Which structural loci yield real wins at tiny scale, and which are exhausted?

The mining question. ~190 `use_*` flags + string-keyed loci (`out_op`, `resid_mode`) exist;
most are **inert step-0 on the champion** (the candidate is built identically → measures a
construction artifact, not the mechanism). We want the short list that *fires* and *clears
the noise* (paired |Δ| ≥ 0.02, 95% CI excludes 0).

- **Q1.1** — Of all `use_*` / `out_op` / `resid_mode` levers, which are **non-inert at step 0**
  on the champion when base and candidate are built identically? *(Scope: champion, seed 42.
  Falsifier: a lever whose step-0 loss differs from champion's by > 1e-4 is a construction
  artifact, not a mechanism — exclude it.)* Related draft: `DRAFT-flag-inertness-on-champion`,
  `DRAFT-identical-construction-differencing`.
- **Q1.2** — **Attention-output locus** (`out_op`, the [B,H,T,D] choke point before W_O): does
  any zero-init cross-head/channel op (`headmix_lowrank1/2`, `head_gate_reparam`,
  `per_hd_affine`) beat the champion past the 0.02 screen? *(First experiment below targets
  this.)*
- **Q1.3** — **Residual-stream locus** (`resid_mode`, the `x = a·x + g·f(x)` add): is any
  identity-init gate (rezero / branch-gain / input-scale, scalar vs per-channel) a real win,
  or are they all sub-band? *(Hypothesis from memory: the model is update-starved, not
  routing-starved — predict NULL. A NULL here is itself a citable `L` rule.)*
- **Q1.4** — **Positional locus**: the champion's own lever is a learnable ALiBi slope +
  poly curvature. Does a *structurally different* decay (FIRE / KERPLE-log / CoPE / per-head
  window) beat it, or is geometric-init learnable ALiBi a local optimum at this scale?

## G2 — Are the wins independent levers or redundant? (the basis question)

A pile of wins is not knowledge until we know how they combine. We already have hard data
that combination is **not** additive-by-default.

- **Q2.1** — Given two confirmed wins, do they stack **additively**, **sub-additively
  (substitutes)**, or **super-additively (complements)**? *(Known points, in-scope:
  update-amount knobs ×2LR & bs=1 are **substitutes** (don't stack); muon-mom 0.90 + ×2LR
  are **super-additive** (−0.0085 + −0.0104 → −0.0278). These belong in the ledger as the
  first `L` entries on lever-composition.)*
- **Q2.2** — What is the **minimal generating set** of levers that reconstructs the champion's
  total gain over the bare base? *(Falsifier: if removing lever X from the set and re-measuring
  loses < 0.02, X is redundant — not in the basis.)*

## G3 — Do tiny-scale structural findings transfer across scale? (the valuable question)

The whole bet: that 3M-param experiments buy knowledge about real models. This is where the
science is worth the most and is currently **completely untested** in the ledger.

- **Q3.1** — Does a lever confirmed at 1–3M **hold its sign** at ≥135M (Phase-2)? *(Scope jump
  is the experiment. Falsifier: sign flips or |Δ| collapses below noise at 135M → the lever is
  scale-specific, not a law.)*
- **Q3.2** — Is transfer **predictable from a cheap signal** (step-0 fire magnitude, early-loss
  slope) without paying for the full large run?
- **Q3.3** — Which **loci are scale-invariant vs scale-specific**? *(Prior: step-0 conditioning
  / warm-start transfers; raw capacity/FFN levers saturate at tiny and may not. Untested.)*

## G4 — What measurement discipline makes a tiny Δ trustworthy? (the meta-rules)

Already partly drafted — these are the rules that protect every answer above.

- **Q4.1** — The noise floor & paired differencing. *(Closed-ish: `L001` noise floor,
  `D001` paired variance reduction.)*
- **Q4.2** — Identical-construction differencing: base and candidate must be built the same way
  or the Δ is a construction artifact. *(Draft: `DRAFT-identical-construction-differencing`.)*
- **Q4.3** — Single-seed inflation / lucky-seed guard: a 1-seed WIN parks in `needs-confirm`,
  never auto-promotes. *(Draft: `DRAFT-single-seed-inflation`.)*

---

## How a question closes

```
open question  ──run experiment(s)──▶  paired Δ in Neon  ──clears gate──▶  L###  (answer, scoped)
                                                          └─sub-band────▶  L###?  or  "NULL" rule
```

A NULL is a result, not a failure: "locus X does not move the champion at tiny scale (|Δ|<0.02,
3 seeds)" is a citable `L` that stops the loop and every future contributor from re-mining it.
This is how the agenda stays finite even though the lever space is large.
