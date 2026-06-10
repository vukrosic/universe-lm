---
id: 002-ffn-family
status: scoping
round: 1
updated: 2026-06-10T00:43:12Z
exit: "12 done ideas OR 3 WINs OR 2026-07-01"
venue_ceiling: workshop
---

# Research brief — 002 FFN family deep-dive

> Candidate filed by brief-proposer, 2026-06-10. Kind: **mechanism-family
> deep-dive** — one sublayer, mined to exhaustion under the 001 protocol.

## Topic

Campaign 001 screened optimizers, attention variants, positional encodings,
and loss functions — but never touched the **FFN/MLP sublayer**, which holds
the majority of non-embedding transformer parameters. The closed list
(`autoresearch/closed.md`) covers attention exhaustively (SWA, RoPE, NoPE,
MLA, GQA, norm zoo, sinks, NSA/diff-attn) yet contains zero FFN levers; the
only FFN items ever considered (Product-key FFN, Soft MoE) sit untouched in
`autoresearch/queue.md` PENDING. This campaign mines the FFN block as a
family: activation/gating variants, memory-augmented FFNs, branch gates, and
FFN-side local mixing — every lever a structural drop-in on the same
`tiny1m3m` baseline.

## Research question

**Which structural edits to the FFN sublayer (activation/gating form,
memory augmentation, branch gating, FFN-side token mixing) lower val loss at
`tiny1m3m` (seed 42) under the ctrl-bracket protocol?**

## Paper claim

At 1M-parameter scale the FFN sublayer is not a solved block: a single-seed
ctrl-bracket screen over N FFN mechanisms separates them into a small
carry-forward WIN set and a majority NULL set, completing the per-sublayer
map started by the attention/optimizer screening campaign.

## Mineability seed list

≥10 distinct directions, one source each (ids from literature memory where
the repo has no digest; unverified ids marked):

1. **GLU variant swap** — GeGLU / ReGLU / Bilinear in place of SwiGLU —
   Shazeer, "GLU Variants Improve Transformer" (arXiv:2002.05202).
2. **Squared ReLU** — Primer's `relu(x)^2` activation
   (arXiv:2109.08668).
3. **ReLU² as used in speedruns** — distinct wiring (no gate branch,
   widened hidden) — modded-nanogpt repo (kellerjordan0).
4. **xIELU activation** — expanded-gating integral of ELU
   (arXiv:2411.13010 — id unverified, EPFL 2024).
5. **Persistent-memory FFN vectors** — learned key/value vectors appended
   to the FFN as memory slots — Sukhbaatar et al. (arXiv:1907.01470).
6. **Product-key memory layer (downscaled)** — replace one FFN with a
   small PKM — Lample et al. (arXiv:1907.05242); already in
   `autoresearch/queue.md` PENDING, never foldered.
7. **Soft-MoE-lite** — 2 experts, soft routing, one layer —
   Puigcerver et al. (arXiv:2308.00951); queue.md PENDING.
8. **Top-k hidden-activation sparsity** — ReLUfication / "ReLU Strikes
   Back" sparsity in the FFN hidden layer (arXiv:2310.04564).
9. **LayerScale on the FFN branch** — per-channel zero-init scalar gate on
   the FFN residual contribution — CaiT (arXiv:2103.17239). Distinct from
   the closed norm zoo (those replaced norms; this gates the branch).
10. **nGPT-style hypersphere constraint on FFN weights/activations** —
    Loshchilov et al. (arXiv:2410.01131), FFN-side only.
11. **Canon-conv on the FFN side** — depthwise causal conv before the FFN
    pre-LN; explicitly carved out of idea 023's scope ("no FFN-side conv",
    `autoresearch/ideas/023-canon-conv/idea.md`) — Griffin
    (arXiv:2402.19427).
12. **Gated FFN output (per-token scalar sigmoid)** — the 024 head-gate
    mechanism transposed to the FFN output site — Qiu et al.
    (arXiv:2505.06708) by analogy; tests whether gating wins are
    site-specific.

Dedup check: none of these appear in `autoresearch/closed.md` (its
architecture entries are all attention/norm/embedding-side; "parallel
block" is closed and is excluded from this list).

## Scope & constraints

- **Tier:** `tiny1m3m` only (0.94M params · 3M tokens). No screen20m, no ladder.
- **Seed:** 42 always. One seed, no sweeps.
- **Changes:** mechanisms / structural edits only — no LR, schedule, or init HP sweeps.
- **Code budget:** implementable in < 200 LoC; step-0 ≈ baseline (identity/zero-init) unless noted.
- **Dedup:** check `autoresearch/closed.md` before filing; reviewer appends on reject.
- Campaign-specific narrowing: every idea must touch the FFN sublayer (or
  its branch/gate); attention-side, optimizer-side, and loss-side levers are
  out of scope here (they belong to 001's ground). Hidden-width changes
  alone are HP sweeps, not mechanisms — excluded.

## Success criteria

- **WIN:** treatment val loss beats *both* in-session ctrls by more than the
  ctrl–ctrl2 gap (noise floor).
- **NULL:** inside variance or wrong sign — still logged in `evidence.md`
  and appended to `closed.md`.
- **Pipeline health:** ≥3 ideas at `needs-run` / `running` so the GPU never
  idles.
- Campaign-level: a per-mechanism WIN/NULL table covering ≥10 FFN levers,
  each row reproducible from one config flag.

## Venue case

`workshop`. Passes the scoped-claims gates of
`autoresearch/skills/paper-writing/SKILL.md` (per its Adaptation Notes:
formal claims with hedge language, comparable Δ tables with the ctrl-bracket
as the error bar, abstract–conclusion alignment around one sublayer). One
coherent campaign with 10–15 ablations and a clean story ("the FFN map at 1M
scale") is exactly the workshop bar in `briefs/PIPELINE.md`. It cannot pass
a main-track experimental bar (seed-42-only — Gate "add trials / ±std" is
structurally out of scope) and at 10–15 mechanisms it is under the ≳30
breadth bar for `tmlr`. **The one change that raises the ceiling a tier:**
merge this campaign's table with 001's (~20 mechanisms) into a
cross-campaign meta-analysis under the identical protocol — together they
clear the ~30-mechanism TMLR breadth bar that 001's own venue case names as
its explicit TMLR path.
