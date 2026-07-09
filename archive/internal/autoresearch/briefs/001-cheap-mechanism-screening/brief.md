---
id: 001-cheap-mechanism-screening
status: active
round: 1
updated: 2026-06-09T23:28:55Z
exit: "20 done ideas OR 2026-06-17"
venue_ceiling: workshop
---

# Research brief — 001 cheap mechanism screening

> Campaign 001, retrofitted from the original hand-written
> `autoresearch/brief.md` (which remains the live copy the dashboard and
> miner read). 13 ideas already `done` under this brief.

## Topic

Cheap mechanism screening for LLM pretraining: which **structural levers**
(optimizers, attention variants, positional encodings, loss functions) lower
validation loss on a fixed tiny model?

## Research question

**Which transferable, identity-init mechanisms improve val loss at `tiny1m3m`
(seed 42) without hyperparameter tuning?**

A null result is informative when the mechanism is sound on paper but does not
fire at this scale; a WIN is a lever worth carrying forward.

## Paper claim

A single-seed, single-tier ctrl-bracket protocol is enough to cheaply separate
pretraining mechanisms into WIN / NULL at 1M scale, and the surviving levers
form a concrete "carry-forward" set.

## Mineability seed list

Retrofit note: campaign already proven mineable — 19 ideas filed, 13 done.
Remaining directions live in `autoresearch/queue.md` and the miner's normal
external search.

## Scope & constraints

- **Tier:** `tiny1m3m` only (0.94M params · 3M tokens). No screen20m, no ladder.
- **Seed:** 42 always. One seed, no sweeps.
- **Changes:** mechanisms / structural edits only — no LR, schedule, or init HP sweeps.
- **Code budget:** implementable in < 200 LoC; step-0 ≈ baseline unless noted.
- **Dedup:** check `autoresearch/closed.md` before filing; reviewer appends on reject.

## Venue case

`workshop` at 20 ideas: one protocol, one coherent WIN/NULL table, clean
story — but under the ~30-mechanism breadth bar for `tmlr`. The one change
that raises the ceiling: extend the same protocol to ≳30 mechanisms (a
follow-on campaign can merge with this one's table — cross-campaign
meta-analysis is the explicit TMLR path).

## Success criteria

- **WIN:** treatment val loss beats *both* in-session ctrls by more than the
  ctrl–ctrl2 gap (noise floor).
- **NULL:** inside variance or wrong sign — still logged in `evidence.md`.
- **Pipeline health:** ≥3 ideas at `needs-run` / `running` so the GPU never idles.
