---
id: 004-measurement-noise-floor
status: needs-scope
round: 1
updated: 2026-06-10T00:16:21Z
exit: "10 done ideas OR 2 WINs OR 2026-07-01"
venue_ceiling: arxiv
---

# Research brief — 004 anatomy of the single-seed noise floor

> Candidate filed by brief-proposer, 2026-06-10. Kind: **cross-cutting
> protocol question** — the campaign is about the measurement instrument
> itself, not any one mechanism family.

## Topic

Every verdict this lab produces rests on the single-seed ctrl-bracket, and
the 2026-06-09 evidence shows the bracket's noise floor is neither small nor
stable: ctrl–ctrl2 gaps ranged 30× in one day (0.0006 in the 003 batch to
0.0175 in the 009 batch), same-seed ctrl spread hit 0.045 ("flash-kernel
nondeterminism", `ideas/001-cautious-muon/evidence.md`), and one session's
ctrl drifted +0.19 cross-day — suspected wholesale-file-sync baseline
pollution flagged "needs follow-up" in `ideas/006-schedule-free-adamw/
evidence.md`. Effects smaller than the floor (010-polyloss at −0.0053) are
unresolvable; shrinking the floor directly raises screening sensitivity.
This campaign treats the harness as the object of study: which structural
interventions in the run/eval pipeline shrink fixed-seed run-to-run spread
and session drift, and what is the calibrated minimum detectable effect?

## Research question

**Which harness-level mechanisms (deterministic kernels, eval-protocol
changes, run-scheduling changes) shrink the fixed-seed ctrl–ctrl gap and
cross-session drift at `tiny1m3m` (seed 42) without shifting the ctrl mean,
and what minimum detectable effect does the improved bracket support?**

## Paper claim

At fixed seed, kernel nondeterminism and session-level harness drift — not
seed variance — set the noise floor of cheap mechanism screening, and a
small set of deterministic-execution and evaluation-averaging interventions
shrinks the minimum detectable effect severalfold at 1M scale.

## Mineability seed list

≥10 distinct directions, one source each (repo evidence where available;
literature ids from memory, unverified ones marked):

1. **Deterministic SDPA backend** — force math/mem-efficient attention +
   `torch.use_deterministic_algorithms(True)`; the flash-kernel
   nondeterminism named in `ideas/001-cautious-muon/evidence.md` (0.045
   same-seed spread) — PyTorch reproducibility docs.
2. **cuBLAS workspace pinning** — `CUBLAS_WORKSPACE_CONFIG=:4096:8` —
   PyTorch reproducibility docs.
3. **TF32 off / fp32 accumulation in attention and matmul** — torch docs;
   isolates reduced-precision accumulation as a noise source.
4. **LAWA-style latest-weight averaging before eval** — average last K
   checkpoints, eval once — Kaddour, "Stop Wasting My Time!" / LAWA
   (arXiv:2209.14981).
5. **Val-loss tail averaging** — report mean of last N eval points instead
   of the final point; speedrun-community practice (modded-nanogpt repo).
6. **Bigger/full-pass validation slice** — eval-set size vs verdict
   stability; harness mechanism (`training/trainer.py` eval loop).
7. **Paired-batch Δ evaluation (common random numbers)** — score trt and
   ctrl on identical val batches and report per-batch paired Δ; classic CRN
   variance reduction (simulation literature; no single arxiv id —
   unverified).
8. **Targeted-patch deploy vs wholesale file sync** — the suspected root
   cause of the +0.19 drift incident
   (`ideas/006-schedule-free-adamw/evidence.md`); mechanism: a
   manifest-pinned deploy step, A/B'd by re-running the 006 bracket.
9. **Ctrl-scheduling geometry** — sandwich (ctrl-trt-ctrl2) vs batch-edge
   ctrls vs interleaved; the 015/016/017 batch shared one bracket across
   three treatments (`ideas/015-moonlight-muon-rms/evidence.md`) — does
   bracket sharing inflate false NULLs?
10. **Bitwise re-run box-health gate** — pre-batch probe that re-runs 50
    steps and requires bit-identical loss before any A/B (extends the
    `boxval` smoke in `ideas/002-cautious-adamw/evidence.md`).
11. **Dataloader order audit/pinning** — verify and pin shard/batch order
    across deploys; drift suspect adjacent to (8); harness mechanism.
12. **Empirical noise-floor meta-analysis** — pool all logged 2026-06 ctrl
    pairs (8+ across `remote-results/2026-06-09-vast-tiny1m3m/` and
    evidence files) into a fixed-seed noise distribution; re-grade every
    001 verdict against the calibrated floor (analysis idea, zero GPU
    cost).

## Scope & constraints

- **Tier:** `tiny1m3m` only (0.94M params · 3M tokens). No screen20m, no ladder.
- **Seed:** 42 always. One seed, no sweeps — this campaign measures and
  shrinks *fixed-seed* noise; multi-seed variance is explicitly out of
  scope (per the one-seed-only project rule, variance = the two-ctrl
  bracket, never a seed sweep).
- **Changes:** mechanisms / structural edits only — no LR, schedule, or init HP sweeps.
- **Code budget:** implementable in < 200 LoC; step-0 ≈ baseline (identity/zero-init) unless noted — here "identity" means the intervention must not change the ctrl *mean* beyond the bracket, only its spread.
- **Dedup:** check `autoresearch/closed.md` before filing; reviewer appends on reject.
- Campaign-specific narrowing: treatments are harness/eval/deploy
  mechanisms, not model mechanisms. Model-side levers belong to 001/002.

## Success criteria

Each idea runs the standard bracket, replicated within-session
(ctrl, trt, ctrl2, trt2 — trt = ctrl + harness mechanism; same GPU-hours
shape as a normal A/B):

- **WIN:** the mechanism's same-session repeat gap |trt − trt2| is ≤ half
  the same-session ctrl–ctrl2 gap, **and** the trt mean sits inside the
  ctrl bracket (mean-preserving). Judged by the ctrl-bracket protocol —
  the bracket is both the baseline and the yardstick.
- **NULL:** repeat gap not reduced (≥ ctrl–ctrl2 gap) or mean shifted
  outside the bracket — logged in `evidence.md`, appended to `closed.md`.
- **Pipeline health:** ≥3 ideas at `needs-run` / `running`.
- Campaign-level: a calibrated minimum-detectable-effect number for the
  default harness and for the best intervention stack, plus a re-graded
  001 verdict table.

## Venue case

`arxiv`. An honest exploratory methods note: "how small an effect can a
single-seed, single-tier, rented-GPU screening rig actually resolve, and
how to cheapen the floor." It passes the `paper-writing` skill's
scoped-claims and figure/table gates (per its Adaptation Notes: every claim
carries its own measured floor; the Δ tables are the paper) but the
evidence is one box, one model size, one seed — too narrow for the
workshop bar of "one coherent campaign with a clean mechanism story", since
several seeds of noise (pun intended) are box-specific. It structurally
cannot claim main-conference (no ±std across trials — the very thing under
study). **The one change that raises the ceiling a tier (→ workshop):**
replicate the top-2 noise interventions on a second box/GPU type (one cheap
rental day) so the floor-shrinking claim is not single-box — cross-box
replication of the *protocol*, not multi-seed of the *model*, keeps the
one-seed rule intact while making the result portable.
