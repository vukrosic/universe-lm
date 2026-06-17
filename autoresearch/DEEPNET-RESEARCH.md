# DeepNet-α on the release ladder — does it bend the scaling exponent, and why?

**Status:** active study (autoresearch loop). **Owner:** Claude (autonomous).
**Last updated:** 2026-06-17.

## Why DeepNet is the right thing to study now

DeepNet-α is the tiny champion's only **release-safe** structural carry: it's an
init/stability mechanism, not a positional one, so it does **not** touch attention
range — it's clean under `DECISIONS.jsonl` **D002** (unlike the alibi/poly-alibi
lever, which is cut). So if *anything* from the tiny tier earns a slot in the 135M
release, deepnet is the first candidate. The question is whether it actually helps
**at scale**, and *why*.

## The mechanism (exact, from the code)

- **α (forward, `use_deepnet_alpha`, idea 197):** every block's sublayer output
  (attention AND FFN) is scaled by a **fixed depth-conditional scalar**
  `α = (2·n_layers)^(−1/2)` before the residual add:
  `x = x + α · sublayer(x)`. It bounds the residual stream's magnitude growth to
  `O(1)` across depth instead of `≈√L`. **0 new params.** Per-rung α:

  | rung | n_layers | α = (2L)^(−1/2) |
  |---|---|---|
  | Ladder8M  | 8  | 0.250 |
  | Ladder13M | 8  | 0.250 |
  | Ladder23M | 15 | 0.183 |
  | Ladder52M | 21 | 0.154 |
  | **135M target** | **30** | **0.129** |

- **β (init, `use_deepnet_beta_init`, idea 288/289 — the UNTESTED half):** canonical
  DeepNet (Wang 2022, Thm 1) *pairs* the forward α with an **init down-scaling** β of
  the V/O/FFN projection weights so the model *update* is bounded at step 0.
  Canonical decoder gain `β = (8·n_layers)^(−1/4)`. The champion implements **only α**.

## Central question

Does DeepNet-α earn a slot in the 135M release — and is its benefit a **steeper
scaling exponent** (advantage that GROWS with scale/depth), or a **constant
intercept shift** (washes at scale, like most tiny-tier wins)?

This matters because the ladder's whole thesis (see `LADDER.md`) is that the prize
is a steeper α, not a lower loss at one small size. DeepNet is a *prior-rich*
candidate for a steeper exponent: its entire reason for existing is **depth
stability**, and our rungs increase in depth (8 → 30). If the benefit is depth-driven,
it should compound exactly where we're extrapolating to.

## Hypotheses (falsifiable)

- **H1 — depth-driven / steeper exponent (the bet).** DeepNet-α's benefit grows with
  n_layers because deeper stacks suffer more residual-variance growth that α bounds.
  → deepnet's fitted `L(N)` has a **steeper α-exponent** than baseline → it earns the
  135M run. *Prediction:* the baseline−deepnet gap widens monotonically across rungs
  (and specifically with **depth**, not just N).
- **H2 — intercept shift (the skeptic).** At our modest depths (≤30 layers, with
  pre-norm + RMSNorm already stabilizing), α is just a small init-conditioning tweak:
  a roughly **constant** benefit that does not bend the exponent and washes at the
  target. *Prediction:* parallel curves — same exponent, lower intercept — and the
  gap at 135M is within noise.
- **H0 — null.** No reliable benefit beyond the paired-noise band.

The ladder is the instrument that distinguishes these: fit `L = E + A·N^(−α)` for
baseline vs deepnet and **compare the exponents**, not just the endpoint losses.

## Experiments

### E1 — Primary: the ladder (RUNNING)
baseline vs deepnet at 8M/13M/23M (local) → 52M/135M (contributors). Fit and compare
exponents with `scaling_fit.py`. **First read** on H1 vs H2. *In flight now.*

### E2 — Depth-isolation (the clean mechanism test; run IF E1 is promising)
The main ladder confounds **depth, width, and N** (each rung changes all three).
DeepNet's claim is about **depth specifically**. So: hold **width fixed**
(`d_model=256`), vary **only depth** `L ∈ {8, 16, 24, 32}` at a matched token budget,
baseline vs deepnet. Plot **Δ(deepnet) vs L**. A monotone-increasing Δ confirms the
mechanism is depth-driven (H1) and not an N/width artifact.
- **Free partial check already in E1:** Ladder8M and Ladder13M **share L=8** (different
  width). If deepnet's benefit is depth-driven, their deepnet-Δ should be ~equal — a
  width-isolation sanity check for free.

### E3 — α vs α+β (completeness)
Test `deepnet` (α only) vs `deepnet_ab` (α + canonical β init) at rungs 1–2. Canonical
DeepNet pairs them; we only carry α. Does the init-side β stack, or is α sufficient?

### E4 — Specificity: is it DeepNet, or any residual damping?
Compare at the cheapest rung (8M): `deepnet` (fixed (2L)^(−1/2)) vs **ReZero**
(`use_re_zero`, learned per-block α init 0) vs **LayerScale** (per-channel γ init 1e-4)
vs a **fixed non-DeepNet constant** (e.g. α=0.5). If a plain constant matches DeepNet's
depth formula, the win is "damp the residual at init," not DeepNet's specific
`(2L)^(−1/2)` — a simpler, equally-good lever and a cleaner story.

### E5 — Understanding (instrumentation, the "why")
Log **per-layer residual-stream RMS** and **per-layer gradient norms** at init and over
training, baseline vs deepnet. Wang Thm 1 predicts deepnet **flattens the per-layer
update-magnitude profile** (bounded update). Confirming that profile change *is* the
mechanistic explanation, independent of the loss number.

**E5 result — step-0 residual-stream probe (`autoresearch/bin/deepnet_probe.py`, 2026-06-17).**
Fixed width (d=128), depth varied, random batch, no training. Per-block residual RMS
growth (last/first):

| L | α=(2L)^(−1/2) | baseline growth | deepnet growth |
|---|---|---|---|
| 4  | 0.354 | 1.03× | 1.00× |
| 8  | 0.250 | 1.08× | 1.01× |
| 16 | 0.177 | 1.16× | 1.01× |
| 30 (target) | 0.129 | **1.31×** | **1.01×** |

Three reads, including an honest one that reshapes the bet:
1. **Mechanism confirmed.** DeepNet-α holds the residual stream flat (~1.01×) at every
   depth — the bounded-O(1) regime Wang Thm 1 predicts.
2. **Depth-dependent (supports H1 direction).** Baseline growth *rises* with depth
   (1.03×→1.31×), so the thing deepnet fixes gets bigger as the stack deepens.
3. **…but mild, which tempers H1 toward H2.** Baseline grows only 1.31× at L=30 —
   nowhere near the loose √L≈5.5× — because **pre-norm + RMSNorm already tames residual
   growth**. So at our depths (≤30) the magnitude deepnet corrects is small; expect a
   correspondingly small loss benefit unless the *update*-side (β, idea 288) or training
   dynamics matter more than the step-0 forward magnitude. **The ladder loss-delta (E1)
   is the arbiter.** Caveat: this probe measures the forward magnitude only; the β
   init-downscale targets the gradient/update bound, which this does not capture — a
   reason E3 (α vs α+β) could still move even though α's forward effect looks small.

**E5 result — per-layer GRADIENT-norm uniformity at L=30 (the update-side, more
revealing than the forward probe).** Random CE loss, one backward; spread=max/min,
cv=std/mean of the per-block grad norms:

| arm | grad spread (max/min) | cv (std/mean) |
|---|---|---|
| baseline   | 1.60 | **0.141** |
| deepnet (α) | 1.04 | **0.011** |
| deepnet_ab (α+β) | 1.09 | 0.020 |

1. **DeepNet-α's primary effect is gradient uniformity, not forward bounding.** It
   cuts per-layer grad-norm variation ~13× (cv 0.141→0.011) — a much larger effect
   than on the forward residual (1.31×→1.01×, already mostly handled by RMSNorm).
   Uniform per-layer updates = all depths learn at compatible rates. *This* is the
   real mechanism at our scale.
2. **β adds no extra flattening.** `deepnet_ab` cv (0.020) ≈ `deepnet` (0.011); β
   only rescales the absolute grad magnitude ~10× (0.031→0.004), which an adaptive
   optimizer absorbs. **Preview of E3: expect β to add little on top of α.**
3. **The caveat that could sink the whole bet → Muon.** Our optimizer is Muon, which
   already orthogonalizes/normalizes gradients *per weight matrix* — plausibly
   supplying much of the same per-layer balancing deepnet provides. **If Muon +
   RMSNorm already do deepnet's job, deepnet-α is largely redundant → small/null loss
   delta (H2/H0).** This is now the leading prediction; E1's loss-delta arbitrates,
   and a sharp follow-up (E6) would be deepnet-α under AdamW vs Muon — if deepnet
   helps far more under AdamW, redundancy-with-Muon is confirmed.

## Decision rule

DeepNet earns a 135M slot **iff** its fitted curve sits below baseline at the target N
— ideally via a **steeper exponent** (H1), not just a lower intercept (H2) — **and** it
does not regress long context. (It's non-positional, so **D002-safe by construction** —
note this is a real advantage over the cut alibi lever.) If only H2 holds, it's a free
small win to keep, but **not** the scaling lever the release needs — keep searching the
`LONG-CONTEXT-IDEAS.md` shortlist for a true exponent-bender.

## Run order (GPU-aware; the box is one RTX 3060)
1. E1 finishes (baseline+deepnet × 8M/13M/23M) → first exponent comparison.
2. If E1 shows a widening gap → **E2 depth-isolation** (the headline result).
3. Cheap parallel screens at 8M when the GPU is free: **E3** (α+β), **E4** (specificity).
4. **E5** instrumentation on one baseline/deepnet pair (no extra training — hook the
   existing run).

Arms are wired in `autoresearch/bin/run_rung.py` (`deepnet`, `deepnet_ab`, `rezero`,
`layerscale`, `fixedres`). All are D002-safe (non-positional). Results land in
`autoresearch/ladder/results.jsonl`; `arch` distinguishes the arm.
