# Taste log — 061 ALiBi bias

## r1 — 2026-06-11 — verdict: revise

- **Bet is stale — references a closed baseline.** The "why it's worth a slot"
  line says *"if tiny1m3m prefers it over RoPE, that tells us the model wants
  explicit distance pressure more than phase rotation"*. RoPE is **closed** (the
  RoPE-base sweep settled on 500k, see `closed.md` axes line). The current best
  baseline is **009 FIRE-PE** (Δ −0.064 to −0.082 vs the V+q+SWA+HighRoPE ctrl
  at 6.3234). The bet has to be reframed against FIRE — *replace* it or *stack*
  on it — not against RoPE. As written, the null result would be "ALiBi vs
  nothing novel", which is uninformative.

- **Replace-or-stack choice is the crux and is unstated.** The mechanism note
  says "drop-in attention bias path" but does not say whether ALiBi is wired
  *instead of* FIRE (then the bet is "ALiBi-only beats FIRE-only" — a high bar
  given FIRE's content-aware kernel) or *in addition to* FIRE (then the bet is
  "the linear distance penalty adds signal on top of FIRE's decay kernel").
  013-CoPE stacked on FIRE and was *destructively* closed (closed.md: trt
  6.4659 vs ctrl ≈6.39, +0.069 — the worst stacking interaction of the
  020–025 cluster). The re-pitch must pick a side and defend it; *not* picking
  a side is a "mood, not a bet" failure.

- **Portfolio fit is poor — PE axis is over-crowded.** Per the 065-bilevel-pe
  r1 review the active needs-taste/repitch/review queue is already 061, 062
  (rejected), 063-YaRN, 064-XPOS, 065 (revise), 072-T5-RPE, 073-DeBERTa.
  Seven PE ideas in flight is a portfolio-fit failure even when each is
  individually fine. To survive triage 061 must clearly differentiate from
  the others — *what does ALiBi test that none of the other PE-queue ideas
  tests, in one sentence?* My candidate framing: "ALiBi is the *only* candidate
  with a fixed non-learnable per-head distance schedule and zero content
  coupling — the cleanest ablation against FIRE's content-aware MLP, so a
  clean null ('linear distance does nothing on top of FIRE's kernel') is itself
  informative." But this must be the miner's argument, not mine.

- **Slope schedule is unspecified — the lever is dominated by this choice.**
  Press et al. (2021) use a fixed geometric schedule of 8 slopes
  {1/2⁰, 1/2¹, …, 1/2⁷} assigned to heads in a round-robin. At tiny1m3m with
  6 heads the schedule reduces to 6 of those slopes; whether to use that
  schedule verbatim, double it for steeper recency, halve it for shallower,
  or learn the slopes is *the* implementation choice. The current mechanism
  note is "per-head slopes" without commitment. Pin one (Press et al.'s
  geometric is the natural default) and treat any deviation as a single
  follow-up — not a sweep, per the one-seed rule.

- **Identity/zero-init pathway must be stated explicitly.** The taste bar
  requires a step-0 identity so a null result is "the lever didn't fire" not
  "the init was bad". ALiBi's slopes are non-learnable constants → setting
  `m_h = 0` for all heads at step 0 makes the attention logits
  bit-identical to the FIRE-only / RoPE-only path; the schedule is then
  linearly ramped on over a small number of warmup steps. State this in the
  re-pitch so the implementer has no judgement to make.

- **Pass/fail bar must reference FIRE, not RoPE.** The plan needs `pass: Δ ≤
  −X` against the **FIRE-equipped control** at 6.3234 (current best baseline),
  not the V+q+SWA+HighRoPE reference at 6.4287. The bar should also be tight
  enough that an effect inside the run-to-run ctrl-gap (~±0.01 at this tier,
  see the 020-025 cluster) is logged as inconclusive, not "passing".

- **Why not a flat reject?** 062-PosInterp was rejected because PI is a
  *long-context extension* mechanism with no operating point at tiny1m3m
  (down-scaling indices inside the already-trained 0–2048 range can only
  hurt). ALiBi is different: it's a *recency bias* that operates at any
  sequence length. It can fire in principle at tiny1m3m, and a clean null
  against FIRE is itself informative ("explicit linear distance pressure is
  redundant with FIRE's content-aware kernel"). The current pitch is
  *salvageable* if the bet is reframed — that is exactly what `revise` is
  for.
