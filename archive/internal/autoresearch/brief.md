# Research brief

> This is the **active campaign's** brief — the paper's opening page. The
> dashboard and miner read it live from `autoresearch/brief.md`. Do not edit
> it ad hoc: topics are chosen via the brief pipeline
> ([`briefs/PIPELINE.md`](briefs/PIPELINE.md)) — proposer files candidates,
> brief-reviewer gates them, the human blesses one, and its body is copied
> here. Current campaign: `briefs/001-cheap-mechanism-screening/`.

## Topic

Cheap mechanism screening for LLM pretraining: which **structural levers**
(optimizers, attention variants, positional encodings, loss functions) lower
validation loss on a fixed tiny model?

## Research question

**Which transferable, identity-init mechanisms improve val loss at `tiny1m3m`
(seed 42) without hyperparameter tuning?**

A null result is informative when the mechanism is sound on paper but does not
fire at this scale; a WIN is a lever worth carrying forward.

## Goal

Run an autonomous loop — mine → taste → define → implement → GPU — that keeps
the remote box busy and accumulates proven mechanisms. Each idea is one ablation
with evidence, not a one-off hack.

## Scope & constraints

- **Tier:** `tiny1m3m` only (0.94M params · 3M tokens). No screen20m, no ladder.
- **Seed:** 42 always. One seed, no sweeps.
- **Changes:** mechanisms / structural edits only — no LR, schedule, or init HP sweeps.
- **Code budget:** implementable in < 200 LoC; step-0 ≈ baseline unless noted.
- **Dedup:** check `autoresearch/closed.md` before filing; reviewer appends on reject.

## Success criteria

- **WIN:** treatment val loss beats *both* in-session ctrls by more than the
  ctrl–ctrl2 gap (noise floor).
- **NULL:** inside variance or wrong sign — still logged in `evidence.md`.
- **Pipeline health:** ≥3 ideas at `needs-run` / `running` so the GPU never idles.
