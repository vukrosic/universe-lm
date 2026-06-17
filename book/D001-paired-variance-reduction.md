# D001 — Paired seeds cancel common-mode noise

**Statement.** Differencing two runs that share their seed (paired Δ = val_treatment −
val_control at the *same* seed) cancels the seed-common component of the variance, so the
standard error of the measured effect is far below the run-to-run val-loss SD.

**Status.** D — derivation. The deductive step is exact; its load-bearing *assumption*
(that same-seed runs are positively correlated) is empirically reconfirmed each batch.

**Scope.** Any A/B where both arms share the initialization + data-order seed and differ
only in the lever under test. Numerically demonstrated at tiny1m3m; the algebra is
scale-free.

## Derivation
For treatment T and control C, Var(T − C) = Var(T) + Var(C) − 2·Cov(T, C). Seed fixes the
weight init and the data-order stream, which are large shared sources of run-to-run
variation. Sharing the seed makes Cov(T, C) large and positive, so Var(Δ) ≪ Var(T) + Var(C).
The leftover variance is only the part of the noise the lever actually interacts with.
Consequence: a paired Δ resolves effects an *unpaired* comparison at the same seed count
cannot. Every screen and confirm in this lab differences same-seed arms for this reason.

## Falsifier
A regime where same-seed T and C are uncorrelated (Cov(T, C) ≈ 0) — then pairing gives no
variance reduction and the gate must fall back to many unpaired seeds. (Would show up as
paired-Δ SD ≈ unpaired SD across a seed sweep.)

## Why it matters
This is the *measurement gate* the whole ledger stands on: it is why a 0.02 screen band is
meaningful at all (see [[L001]]). Mis-pairing (different construction between arms) reintroduces
a non-cancelling term — the failure mode catalogued separately as the construction-artifact rule.

Links: PIPELINE.md (measurement gate), [[L001]].
