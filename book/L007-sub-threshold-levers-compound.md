# L007 — Sub-threshold levers can compound past the floor

**Statement.** Levers that individually miss the screen band can compound past it when
stacked: DeepNet-α (≈ −0.017 alone) + poly-ALiBi (≈ −0.011 alone, a null) stack into a
confirmed champion, and successive sub-threshold refinements continued to compound down the
lineage.

**Status.** L — strong for the *existence* of compounding (each lineage rung is a paired
confirm). The strongest single super-additivity claim (323) is **L? (tentative)** — its 3-seed
CI straddles 0.

**Scope.** tiny1m3m, ALiBi base. Compounding is **not universal** — anti-stacking happens too
(see Falsifier), so this is "can compound," not "always compounds."

## Evidence (champion lineage, each rung a paired 3-seed confirm)
- 253-deepnet-α: 6.2367 (6.2309/6.2434/6.2359), Δ −0.0172 vs ALiBi (2.73σ) — sub-band alone.
- poly-ALiBi alone: 230 Δ −0.0111 — a null alone.
- 267 stack (DeepNet-α + poly-ALiBi): 6.2209 (6.2125/6.2306/6.2197), Δ −0.0330 vs ALiBi —
  the two sub-band levers clear the floor together.
- 296 combo (slope+curvature warm-start): 6.1998 (6.1947/6.2131/6.1916), Δ −0.0218 vs 267.
- 323 (+ optimizer levers): 6.1720 (6.1762/6.1669/6.1728), per-seed paired Δ
  −0.0185/−0.0462/−0.0188 (mean −0.0278). **Point estimate clears 0.02, but with n=3 the
  t-based 95% CI ≈ [−0.067, +0.012] includes 0** — super-additivity here is tentative.

## Falsifier
If stacking sub-band levers *systematically* gave a combined Δ ≤ the best single part
(anti-stacking as the rule). One in-scope counterexample already exists — 322 (momentum + bs=1)
anti-stacked — which is why the claim is "can," and why every stack must be measured, not
assumed.

## Why it matters
The core search strategy of the lab: a refined base is improved by *stacking independently
right-sign sub-band levers*, not by hunting one big winner. Tempered by [[L006]] (not everything
stacks) and the 323 CI caveat (verify magnitude, don't trust a point estimate at n=3).

Links: [[L004]], [[L003]], [[L006]], [[L001]].
