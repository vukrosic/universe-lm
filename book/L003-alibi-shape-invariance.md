# L003 — ALiBi shape invariance (curvature is otherwise irrelevant)

**Statement.** Among monotonically *growing* distance penalties, the exact curvature is
statistically indistinguishable on the bare model — linear ALiBi and polynomial ALiBi tie at
this scale; what matters is that the penalty grows ([[L004]]), not its shape — **except** that
once the residual stream is stabilized (DeepNet-α), the extra curvature degree of freedom
becomes weakly usable.

**Status.** L — strong for the bare-path invariance; the scope-extension (poly becomes weakly
load-bearing under DeepNet-α) is the documented boundary, not a contradiction.

**Scope.** tiny1m3m. Concave shapes are excluded — those *lose* ([[L002]]).

## Evidence
- Bare path: 230-poly-alibi Δ −0.0111 vs linear ALiBi — inside the L001 floor (a tie).
- Scope boundary: under DeepNet-α the poly degree of freedom contributed to the 267 champion
  stack (poly-ALiBi + DeepNet-α reached 6.2209; see [[L007]]). So curvature is inert on the
  bare penalty but a small usable DOF once the deep stack is stabilized.

## Falsifier
A curvature variant that beats linear past the 0.02 band on the **bare** model (CI ≠ 0) —
would break the bare-path invariance rather than just extend scope.

## Why it matters
Tells you not to spend search budget tuning penalty curvature in isolation — but flags that
DOFs which are inert alone can reopen once a stabilizer is added (the compounding theme of
[[L007]]).

Links: [[L002]], [[L004]], [[L007]].
