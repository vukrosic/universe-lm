# Taste log — 033 sophia

## r2 — 2026-06-11 — verdict: accept

**All four r1 findings closed; the pitch is now a real, falsifiable bet with a genuine discriminator from the 003-SOAP NULL.**

- **Routing committed (option a) and the SOAP-bypass concern is dissolved by the mechanism difference, not papered over.** The r1 critic flagged option (a) as "same SOAP routing, same SOAP problem" — but the miner correctly answers that the *problem* with SOAP wasn't the routing slot, it was that the `(d_in, d_in)` = `(49152, 49152)` eigendecomp on `token_embedding` is intractable and the SOAP impl falls back to AdamW on that param specifically. Sophia's Hutchinson diagonal is *shape-equal to the gradient*, so it runs on the embedding without fallback. This is a real, checkable discriminator — not a rhetorical one — and it isolates a clean question the prior 7 optimizer probes did not ask: "does elementwise curvature info on the vocab-head slot move the needle where AdamW's `1/√v` and SOAP's-via-fallback-AdamW could not?"

- **The bet is now sharp and directional.** "WIN of −0.01 to −0.03 vs AdamW because Sophia damps per-coordinate steps on rare-token rows where AdamW under-conditions, with clip ρ as the safety net." That's the "expect X because Y" the r1 critic demanded. The lower edge (−0.01) is right at the plan-bar / ctrl-gap floor (~0.0047 recent ctrls), so a marginal win is borderline measurable — but the bet *is* committed, falsifiable, and routed to a recognizable closure (WIN → curvature on the vocab head matters; NULL → "even the cheapest elementwise curvature estimator with a clip safety net can't beat AdamW at 732 steps," which is genuinely new info past the 003-SOAP NULL because Sophia didn't bypass the slot).

- **EMA-firing math lifted; reflex-reject path closed.** β1≈0.965 → ~38 momentum half-lives, β2≈0.99 → ~10 Hessian half-lives, k=10 → ~73 Hutchinson refreshes across 732 steps. The 018-ademamix failure mode (β3=0.9999, ~0.1 half-lives) does not apply. Anyone re-reading the idea won't fire-and-forget reject it on that axis.

- **Portfolio: 8th optimizer probe, but the slot is paid for.** The angle — *elementwise* curvature on the *vocab-head slot* — has been touched by zero prior optimizer ideas (SOAP couldn't reach it; 001/002 cautious-variants, 005 decoupled-Muon, 006 schedule-free are orthogonal; 011 Lion / 015 Moonlight-Muon don't touch the AdamW path). Not crowded on the axis Sophia actually tests.

- **Niche fit / transfer fine.** transfer-risk: low is defensible at 125M-1.5B paper validation; tiny1m3m at vocab=49152 (~91% of params on the AdamW slot) is exactly where the bet *can* fire if it's real; mechanism, not HP; identity-recoverable (replace optimizer = revert).

- **Residual risk noted, not blocking.** The "AdamW underconditions rare-token rows because `v` is small" story is the weakest link — both AdamW's `v` and Sophia's `h` are squared-gradient surrogates and degenerate similarly on rarely-fired rows. The actual discriminator is more likely the *Hutchinson averaging over batch positions* (smooths the per-row estimate) plus the *clip ρ* (caps the blowup on near-zero denominators), not the diagonal Hessian per se. This is a quibble on the *why* in the bet sentence, not on whether the A/B is worth running — the run will tell us regardless of which sub-mechanism dominates. Bouncing to a 3rd round just to refine "we expect X because Y₁ vs Y₂" would burn a round on a sub-discrimination the data settles for free.

**Verdict: accept → needs-review (round reset to 1 for the definition gate).**

## r1 — 2026-06-11 — verdict: revise

**Mechanism is real and the EMAs *do* fire at tiny1m3m (unlike 018-ademamix) — but the bet is vague, the routing is destructive to the load-bearing Muon path, and a null teaches us nothing past the 003-SOAP NULL already on record.**

- **Sanity check vs 018-ademamix precedent.** AdEMAMix was taste-rejected because β3=0.9999 (half-life ~7k steps) is essentially flat across the 732-step tiny1m3m run. Sophia's constants are very different: β1≈0.965 (half-life ~19 steps) and β2≈0.99 for the Hessian EMA (half-life ~69 steps). At 732 optimizer steps (batch 2 · seq 2048 · 3M tok, grad_accum=1) you get ~10 half-lives on the Hessian EMA and ~38 on momentum — both fully fired. Hessian refresh every k=10 steps = ~73 refreshes. The "EMA-doesn't-fire" failure mode of 018 does *not* apply. State this explicitly in `idea.md` so the next reader doesn't dismiss it on reflex.

- **The bet is a vibe, not a prediction.** "A good probe for whether tiny1m3m is limited by missing curvature information rather than missing regularization" is the prompt's textbook example of a `revise`. Pick a side, commit a mechanism: *e.g.* "we expect a WIN of −0.01 to −0.03 because diagonal Hessian damps the per-coordinate update on high-curvature directions in the vocab tail, which AdamW's `1/√v` underestimates at sparse tokens." Or the inverse: "we expect a NULL because 732 steps is too short for the Hutchinson estimator to converge below noise, so the clip falls back to AdamW-like behavior." One sentence of "expect X because Y" — without it, both outcomes read as inconclusive.

- **Routing is destructive to Muon.** `idea.md` says "Sophia on the 2D matrix weights, with 1D scalars and norms left on the existing path." Read literally, that strips Muon out of the 2D-hidden path — and Muon (with Moonlight RMS, 015 WIN) is the load-bearing optimizer in this stack. A run that removes Muon is comparing "no-Muon + Sophia" vs "Muon + AdamW", not "AdamW vs Sophia". Three options the miner must pick from and write into `idea.md`:
  (a) Sophia replaces *only* the AdamW path (vocab embed, lm_head, emb_proj, 1D scalars stay on AdamW or move under Sophia for 2D) — same SOAP routing, same SOAP problem (see below).
  (b) Sophia replaces Muon on 2D hidden, AdamW unchanged elsewhere — a destructive A/B, useless as a curvature probe because the loss of Muon orthogonalization dominates the delta.
  (c) Sophia on 2D hidden, *plus* keep Muon's update direction (Sophia rescales a Muon-orthogonalized step) — a real composition but ~3× the LoC and a different paper than the source. Don't claim this unless you spec it.
  Whichever you pick, also spell out what happens to `token_embedding` / `lm_head` / `emb_proj` (the SOAP plan made these explicit at `003-soap/review.md` r2).

- **Information value vs the 003-SOAP NULL.** SOAP at tiny1m3m was a NULL (closed.md: "vocab params on AdamW fallback so SOAP mostly bypassed at tiny1m3m"). If Sophia takes routing (a) above, it inherits the same fallback geometry and a NULL repeats the SOAP story — uninformative. The miner needs one paragraph naming the *specific mechanism difference* that would make Sophia win where SOAP didn't (not "EMA of grad / Hessian" — that's the description, not the discriminator). Candidates: (i) elementwise diagonal Hutchinson trace vs SOAP's full eigenbasis is more stable in 732 steps, (ii) clip ρ caps step size where SOAP's preconditioner blew up at our scale, (iii) per-step Hessian estimate vs SOAP's K-step basis is more responsive at short horizons. Pick one, argue it. A null *with* this argument teaches "even the cheapest curvature estimator can't beat AdamW at 732 steps"; without it, the null reads "another curvature optimizer didn't fire, like SOAP."

- **Portfolio.** Optimizer NULLs already in closed.md: 001 cautious-muon, 002 cautious-adamw, 003 SOAP, 005 decoupled-qkv-muon, 006 schedule-free-adamw. Optimizer WINs: 011 cautious-lion, 015 Moonlight-Muon-RMS. 8th optimizer at tiny1m3m — not auto-reject, but the bar is "what does THIS one teach that the prior 7 didn't?" Answer it on the same line as the bet.

- **Transfer / niche fit is fine.** transfer-risk: low is defensible — Sophia has 125M-1.5B LM-pretraining evidence at and above the target 135M scale. Not the problem here; the problem is the bet at tiny1m3m specifically.

**To pass r2: (1) commit the routing (one of the three options, named), (2) one sentence of crisp directional bet with mechanism, (3) one paragraph differentiating from the SOAP NULL. EMA-firing math (already done above) can be lifted into `idea.md` verbatim.**
