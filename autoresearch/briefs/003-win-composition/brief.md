---
id: 003-win-composition
status: needs-scope
round: 1
updated: 2026-06-10T00:16:21Z
exit: "12 done ideas OR 3 WINs OR 2026-07-01"
venue_ceiling: tmlr
---

# Research brief — 003 do the WINs compose?

> Candidate filed by brief-proposer, 2026-06-10. Kind: **evidence-driven
> follow-up theme** — the campaign 001 evidence base directly poses this
> question; every seed below cites a repo evidence file.

## Topic

Campaign 001 produced a carry-forward WIN set — FIRE-PE (Δ −0.064/−0.082,
`ideas/009-fire-pe/evidence.md`), cautious-Lion (Δ −0.031,
`ideas/011-cautious-lion/evidence.md`), Moonlight Muon-RMS (Δ −0.014/−0.019,
`ideas/015-moonlight-muon-rms/evidence.md`), QK-Norm (Δ −0.014/−0.019,
`ideas/016-qk-norm/evidence.md`) — and one loud warning that wins do **not**
trivially add: CoPE stacked on FIRE was *worse than no positional encoding
at all* (+0.143 vs FIRE-alone, `ideas/013-cope/evidence.md`). Meanwhile 015
and 016 tied at exactly 6.3906 in the same bracket, and the 016 evidence
Notes explicitly hypothesize a shared "small-model headroom" — i.e. possible
redundancy. 001 answers "which levers fire alone?"; this campaign answers
the question its own evidence raised: **what is the interaction structure of
the WIN set** — additive, redundant, or destructive?

## Research question

**Do the campaign-001 WIN levers (FIRE, cautious-Lion, Moonlight-RMS,
QK-Norm, plus any new WINs in flight) compose additively at `tiny1m3m`
(seed 42), and when they do not, is the interaction redundancy or
destructive interference?**

## Paper claim

Single-lever WINs at 1M scale do not predict stacked performance: the
pairwise composition grid of the carry-forward set classifies each
interaction as additive, redundant, or destructive, and the best stack is
not the full stack.

## Mineability seed list

≥10 distinct directions, one source each (mostly the repo's own evidence —
the levers are already implemented, so each idea is a flag-combination
config plus an interaction analysis, trivially <200 LoC):

1. **FIRE + QK-Norm** — `ideas/009-fire-pe/evidence.md` ×
   `ideas/016-qk-norm/evidence.md`; both attention-score-path, different ops.
2. **FIRE + Moonlight-RMS** — `ideas/015-moonlight-muon-rms/evidence.md`;
   architecture × optimizer, the cleanest additivity bet.
3. **FIRE + cautious-Lion** — `ideas/011-cautious-lion/evidence.md`;
   architecture × optimizer-update-rule.
4. **QK-Norm + Moonlight-RMS redundancy probe** — the exact-tie at 6.3906
   and the shared-headroom hypothesis in `ideas/016-qk-norm/evidence.md`
   Notes; sub-additivity here confirms redundancy.
5. **Moonlight-RMS + cautious-Lion** — two update-rule edits on different
   param groups (`ideas/015.../evidence.md`, `ideas/011.../evidence.md`);
   tests optimizer-side composability.
6. **QK-Norm + cautious-Lion** — stability lever × optimizer
   (`ideas/016.../evidence.md`, `ideas/011.../evidence.md`).
7. **Triple stack FIRE + QK-Norm + Moonlight-RMS** — best-pair winner plus
   third lever; tests whether interaction terms are pairwise-explainable.
8. **Full carry-forward stack (all WINs)** — directly tests 001's "the
   surviving levers form a concrete carry-forward set" claim
   (`briefs/001-cheap-mechanism-screening/brief.md`).
9. **Gated CoPE-on-FIRE repair** — zero-init learnable gate on the CoPE
   bias to test the 013 destructive-interference hypothesis ("combined
   per-position bias too large", `ideas/013-cope/evidence.md`); if the gate
   learns ≈0, interference is confirmed mechanistically.
10. **Second additive-bias interference probe** — FIRE + ALiBi
    (arXiv:2108.12409) under the same zero-init gate; tests whether 013's
    failure generalizes to *any* second positional bias or was CoPE-specific.
11. **SSMax-on-FIRE** — already specced as follow-up #1 in
    `ideas/025-scalable-softmax/idea.md`; folds into this grid if 025 lands
    a WIN under 001.
12. **Interaction-sign meta-analysis** — classify every measured pair as
    super-/sub-additive/destructive against predicted Δ-sums across the
    001+003 tables (analysis idea; sources: all evidence.md files).

In-flight WINs from 020–025 extend the grid combinatorially — the seed list
grows as 001 retires, so the miner cannot starve.

## Scope & constraints

- **Tier:** `tiny1m3m` only (0.94M params · 3M tokens). No screen20m, no ladder.
- **Seed:** 42 always. One seed, no sweeps.
- **Changes:** mechanisms / structural edits only — no LR, schedule, or init HP sweeps.
- **Code budget:** implementable in < 200 LoC; step-0 ≈ baseline (identity/zero-init) unless noted.
- **Dedup:** check `autoresearch/closed.md` before filing; reviewer appends on reject.
- Campaign-specific narrowing: every idea must combine ≥2 levers with
  *measured single-lever evidence* in this repo (or add a zero-init gate to
  a measured-destructive pair). The ctrl for a stack-on-X idea is the
  X-equipped config (per the precedent set in
  `ideas/021-value-residual/idea.md`: never re-litigate the single-lever
  question). No new single mechanisms — those belong to 001/002.

## Success criteria

- **WIN:** stacked treatment beats *both* in-session ctrls (the
  best-single-lever config, bracketed) by more than the ctrl–ctrl2 gap.
- **NULL:** inside variance — logged as "redundant" if the single-lever Δs
  predicted a win, with the interaction sign recorded in `evidence.md`.
- **Drift (worse than ctrls):** logged as destructive interference — for
  this campaign that is a first-class *result*, not a failure (013 is the
  template).
- **Pipeline health:** ≥3 ideas at `needs-run` / `running`.
- Campaign-level: a complete pairwise interaction matrix over the WIN set,
  every cell labeled additive / redundant / destructive.

## Venue case

`tmlr`. This is the follow-on campaign that 001's own venue case names as
its explicit TMLR path: merging this interaction grid with 001's ~20-row
single-lever table yields 30+ mechanisms/configs under one identical
protocol — a cross-campaign meta-analysis with tightly scoped claims
("at this tier, under this bracket, these interactions have these signs"),
which is what TMLR reviews (correctness of scoped claims, not impact). Per
the `paper-writing` skill's gates (scaled by its Adaptation Notes): claims
are formal and hedged, tables are directly comparable (shared ctrl-bracket
error bar), and the peer-review pre-mortem survives because the paper never
claims transfer beyond `tiny1m3m`/seed-42. What it cannot pass: any
multi-seed ±std bar — the ctrl-bracket is the only error estimate, which is
why `tmlr` (breadth + scoped correctness) is the ceiling and main-conference
is unreachable. If the breadth merge with 001 falls through (e.g. 001's
table is judged too noisy after the 2026-06-09 drift incident), the
fallback is `workshop` on the composition grid alone — that risk is why the
one ceiling-preserving change is to re-bracket the 001 WINs inside this
campaign's own sessions (each stack run re-measures its single-lever ctrl,
rebuilding the merged table on clean within-session brackets).
