# L002 — Concave distance penalties underperform linear

**Statement.** A concave (log-distance) attention penalty — KERPLE-log — is *worse* than
linear ALiBi by a large, paired margin (≈ +0.056 val loss), well outside the noise floor.

**Status.** L — strong. The sign and size are conclusive in-scope.

**Scope.** tiny1m3m. A claim about penalty *shape* given the same "growing with distance"
family ([[L004]] establishes the family is load-bearing).

## Evidence
- ALiBi deep-dive: paired KERPLE-log vs linear ALiBi Δ ≈ +0.056 (concave loses), several ×
  the L001 floor.
- Box-era corroboration: 231-kerple-log-alibi Δ +0.0449 (wrong-sign vs alibi);
  269-deepnet-kerple-log Δ −0.0089 (null even under the stabilized champion). No regime
  found where the concave form helps.

## Falsifier
Any in-scope paired comparison where a concave distance penalty beats linear past the 0.02
band with CI excluding 0.

## Why it matters
Bounds the positional-penalty design space: among growing penalties, *don't* bend the curve
concave. Pairs with [[L003]] (curvature is otherwise irrelevant) and [[L004]] (the growth
itself is what matters).

Links: [[L003]], [[L004]].
