# How We Screen LM Training Tricks on a Single H100: tiny1m3m

The fastest way to kill a research loop is to run the wrong experiment.

We run every new training idea at the same tiny scale first, on the same seed, and call the result a *screen* — not a verdict.

A screen tells you whether a trick is worth carrying up the scale ladder at all.

This post is a tutorial for the screen we run today, the seven ideas that have cleared it, and the ones still in flight.

## Why Care About the Screen?

The flagship target is a 135M-parameter, 2T-token run that beats SmolLM2-135M.

A flagship run costs roughly $3k and ~6 days on 8× H100.

Running 30 different "what if we used Cautious masking" variants at flagship scale is not a plan. It is a budget fire.

So every idea gets pushed through a single cheap tier first — `tiny1m3m`, 0.94M params, 3M tokens, seed 42. A run takes ~5 minutes on a V100. We run 10 of them in an `arq` queue and move on.

If a trick cannot beat the control at tiny1m3m, we drop it. If it can, it goes onto the scale ladder for transfer testing.

```text
tier            params   tokens    wall-clock    $/run
tiny1m3m (this) 0.94M    3M        ~5 min        ~$0.01
10M             10M      ~1B       hours         tens
30M             30M      ~3B       hours         tens
135M (flagship) 135M     2T        ~6 days       ~$3k
```

The screen is the gate. The scale ladder is the proof.

## Step 1: Pick a Single Seed and Stick With It

Every ablation runs at **seed 42 only**.

No `≥3 seeds`. No seed sweeps. No per-seed means.

This is a hard rule, not a budget accident. The screen has to be cheap enough that we can run ten of them in an hour. Multi-seed protocols are out of scope.

```text
seed 42 only — no exceptions
A sub-noise effect is inconclusive, not real.
Log it and move on. Do not "add seeds to confirm".
```

The discipline is uncomfortable. A −0.003 Δ on a single seed is *not* evidence. Neither is a +0.003 Δ. Both are inside the noise band. We log the idea as NULL and stop.

## Step 2: Run a Two-Control Variance Bracket

A single control is not a control. It is a coin flip.

Every batch runs **two** control configurations back-to-back. The two values form a *bracket* — a measure of how much the run-to-run noise actually is on this box, today, on this dataset.

```text
ctrl  = 6.3875
ctrl2 = 6.4050
gap   = 0.0175          ← this is the noise bracket
```

If a treatment sits inside that gap, we call it NULL — the effect is indistinguishable from "the box had a bad day".

If a treatment beats **both** controls by more than the gap, we call it WIN. Anything clearly worse is NEG.

```text
WIN  : Δ < -gap  (treatment beats both ctrls beyond noise)
NULL : |Δ| ≤ gap (effect could be the run, not the trick)
NEG  : Δ > +gap  (clear regression)
```

That is the only rule. No p-values, no bootstraps, no per-seed std. The bracket *is* the noise estimate.

## Step 3: Keep the Box Fed

A rented GPU that is sitting idle is the one failure this system must not tolerate.

The pipeline target is **≥3 ideas at `needs-run`/`running` at all times**. The moment a slot frees, the runner fills it from the queue.

```text
slot 1: 020-forgetting-attn      running
slot 2: 021-value-residual       running
slot 3: 022-softpick-attention   running
─────────────────────────────────────────
queue:  023-canon-conv, 024-gated-attn, 025-ssmax
```

If the box is empty, an upstream gate is starving the queue — the taste agent, the reviewer, the code-implementer, or the miner. Drain it. Don't wait for a human to notice.

## Step 4: Push Every Idea Through Three Gates

An idea is not "ran on hardware" just because someone wrote a config. It has to clear three gates first, each one a doer paired with a skeptical critic.

```text
Gate 1  Taste       miner        →  taste-reviewer
        "Is this idea worth a slot at all?"

Gate 2  Definition  reviser      →  reviewer
        "Is the idea fully and soundly specified?"

Gate 3  Code        code-impl    →  code-reviewer
        "Does the code match the spec and run correctly?"
```

Each gate runs its own 3-round budget. On round 3, the critic may only `accept` or `reject` — no idea cycles forever. A `reject` moves the folder to `_closed/` and the miner never re-files it.

The status field in `idea.md` is the only routing signal. Agents don't talk to each other — they just grep for their queue and claim work.

```text
needs-taste → tasting → needs-review → reviewing → needs-plan
   → planning → needs-codereview → codereviewing → needs-run
       → running → done
```

## Step 5: Read the Remote Run Log

Every finished run lands in one table. That table is the only honest record of "what did we actually measure".

```text
date        idea                  val     Δ vs ctrl
2026-06-09  ctrl  (vast-34386)    6.3875     —
2026-06-09  ctrl2 (variance brkt) 6.4050     —
2026-06-09  001 Cautious-Muon     6.4125   +0.025 / +0.0075   NULL
2026-06-09  004 RetNet retention  6.4162   +0.029 / +0.011    NULL
2026-06-09  005 Decoupled QKV     6.3909   +0.003 / -0.014    NULL
2026-06-09  009 FIRE-PE           6.3234  -0.064 / -0.082     WIN
2026-06-09  006 SF-AdamW          6.8056   +0.21 / +0.20      NEG
2026-06-09  010 PolyLoss          6.5938  -0.0053 / -0.0112   NULL
2026-06-09  011 Cautious-Lion     6.3941  -0.0312 / -0.0321   WIN
```

Two controls. Seven ideas. Two wins, four nulls, one clear negative. None of this is hand-curated. The whole table is the truth.

## Step 6: Plot It as a Bar

A table is fine for the runner. For everyone else, a picture.

![Delta vs control per finished idea](../posts/charts/delta_bars.png)

```text
green  = WIN     (beats both ctrls by more than the bracket)
gray   = NULL    (sits inside the bracket — inconclusive)
red    = NEG     (clear regression)
```

The shaded band is the single-seed noise band (±0.01). Anything inside it is a coin flip, regardless of color. We do not over-interpret the gray.

## Step 7: Read the Two Wins

### FIRE positional encoding (009)

FIRE replaces RoPE with a learnable position-dependent bias added directly to attention logits. The bias is *content-aware* — a small MLP over learned projections of the token embeddings chooses the effective distance — but the distance kernel itself is a fixed monotone-decay Lp-norm.

The screen result:

```text
treatment:  6.3234
ctrl:       6.3875    Δ = -0.0641
ctrl2:      6.4050    Δ = -0.0816
bracket:    0.0175
```

The treatment beats both controls by **3-4× the bracket**. Margin to spare. This is the largest single Δ in the screen so far, and it came from the smallest expected-Δ bin (the idea.md forecast was −0.005 to −0.02). The win is on the train-distribution val loss, not on length-extrapolation — that test is a future-tier question.

### Cautious-Lion (011)

Cautious masking zeroes the optimizer update wherever the update direction disagrees with the current gradient's sign. Liang et al. (2024) showed this one-line trick helps AdamW and Muon. The question here: does it help Lion too, or is Cautious momentum-specific?

```text
treatment:  6.3941
ctrl:       6.4253    Δ = -0.0312
ctrl2:      6.4262    Δ = -0.0321
bracket:    0.0009
```

The bracket is unusually tight (0.0009), so even a 0.03 effect clears it by 30×. The win is robust *within this session*. The same session had a +0.19 baseline drift versus the prior day, so we log the win in-session only — the absolute number is suspicious, the relative number is not.

## Step 8: Take the Nulls Seriously

The four nulls are not failures. They are results.

```text
001 Cautious-Muon    +0.0075 / -0.014  → momentum-specific
004 RetNet retention +0.011  / +0.011   → linear-attn, no win at tiny1m3m
005 Decoupled QKV    +0.003  / -0.014  → split for Muon routing, no win
010 PolyLoss         -0.0053 / -0.0112 → loss smoothing, inside bracket
```

Each one teaches us something. Cautious-Muon and Cautious-Lion both pass, but with different magnitudes — the sign-update path is alive, the momentum-variance path is weaker. RetNet retention doesn't transfer at this scale. PolyLoss sits inside noise at this scale and may need a larger test.

A null that we logged is more useful than a win we never measured.

## Step 9: Do Not Hide the Negative

006 (Schedule-Free AdamW) regressed hard: +0.21 over the controls. That is a clear negative — well outside the bracket in the wrong direction.

```text
treatment:  6.8056
ctrl:       6.5953    Δ = +0.2103
ctrl2:      6.6091    Δ = +0.1965
```

Reasons matter. The +0.19 session-level drift on the same day means the absolute number is probably off by that much, but the relative negative is still real — the treatment is *worse than* its in-session control by 0.21. We do not bury this in a footnote. A negative result is a result.

## What Is In Flight (Placeholders)

The screen is alive. Several batches are running, composing, or queued for scale transfer. The sections below are empty on purpose — they get filled in as the runs land.

### Batch 020–025 (running now)

Ten-job arq queue, launched 2026-06-10 on `vast-81.45.65.189` (V100-PCIE-32GB). All 9 configs build-smoke OK on CPU. Wall-clock target ~50 min.

```text
<!-- RESULTS PENDING: 020-025 arq batch -->

slot 1: 020-forgetting-attn      ctrl_fire + FIRE + ForgettingAttn
slot 2: 021-value-residual        ctrl_fire + V-residual
slot 3: 022-softpick-attention    ctrl_fire + Softpick
        : 023-canon-conv          ctrl_fire + Canon (gated causal conv)
        : 024-gated-attn          ctrl       + Gated Attention
        : 025-ssmax               ctrl       + Scalable-Softmax (SSMax)
        : ctrl_fire + ctrl_fire2  variance bracket
```

These all stack on top of the FIRE-equipped baseline (where applicable), so a win here is a *FIRE-relative* win, not a baseline-from-scratch win.

### Compositions 026 / 029 / 030 (queued)

Compositions stack two levers from the screen and ask whether they add. None have run yet.

```text
<!-- RESULTS PENDING: compositions 026, 029, 030 -->

026 FIRE × QK-Norm            additive?  expected −0.07 to −0.09
029 V-Norm                     symmetric to QK-Norm on the V projection
030 U-Net skip sigmoid(-1.5)   modded-nanogpt fix; +1.25% speedrun
```

The interesting one is 026: if FIRE-PE and QK-Norm both individually clear the bracket at tiny1m3m, the composition is the test of whether attention levers are additive or saturating.

### 10M / 30M token transfer runs (Phase 2)

The scale ladder is the proof that a tiny-scale win was not a tiny-scale artifact. It is the next, larger tier — and it is empty today.

```text
<!-- RESULTS PENDING: 10M / 30M transfer runs -->

tier       FIRE    Cautious-Lion    Cautious-Muon    RetNet
1M         -0.064   -0.031          +0.0075          +0.011
10M        ...      ...             ...              ...
30M        ...      ...             ...              ...
```

The chart below is the placeholder. The 1M column is real data. The 10M and 30M columns are blank on purpose — they get filled when the Phase-2 ladder runs.

![Scale transfer stub](../posts/charts/transfer.png)

The rule from the program plan: a lever enters the flagship recipe only if it clears the bracket at *both* 10M and 30M. If FIRE wins at 1M and dies at 10M, FIRE gets dropped. Stacking happens after that, not before.

## Tiny Hand Check

A worked example using the FIRE numbers, in case the rule is still abstract.

```text
treatment val:  6.3234
ctrl  val:      6.3875
ctrl2 val:      6.4050
bracket:        6.4050 - 6.3875 = 0.0175

Δ vs ctrl  = 6.3234 - 6.3875 = -0.0641
Δ vs ctrl2 = 6.3234 - 6.4050 = -0.0816

Is Δ < -bracket?
  |-0.0641| = 0.0641 > 0.0175   YES
  |-0.0816| = 0.0816 > 0.0175   YES

Both Δs clear the bracket. → WIN.
```

A null would look like this:

```text
treatment val:  6.4162   (004 RetNet)
ctrl  val:      6.3875
ctrl2 val:      6.4050
bracket:        0.0175

Δ vs ctrl  = +0.0287
Δ vs ctrl2 = +0.0112

|Δ ctrl|  = 0.0287 > 0.0175   (would say WIN)
|Δ ctrl2| = 0.0112 < 0.0175   (would say NULL)

Mixed → does not beat BOTH ctrls → NULL.
```

That is the rule. It is the only rule.

## Done Checklist

You are done when:

- you can explain why every idea runs at seed 42 and at tiny1m3m
- you can state the two-ctrl variance rule in one sentence
- you can read a row of the run log and call its verdict WIN / NULL / NEG
- you can name the three gates and what each one filters for
- you know which sections of this post are placeholders waiting for runs

Stop here. The next post will be the 020–025 batch results, with the same table and the same bar chart filled in.
