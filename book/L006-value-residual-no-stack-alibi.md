# L006 — Value-residual does not stack on a distance penalty

**Statement.** Value-residual — a stand-alone win on the bare model — gives no measurable
gain once ALiBi (and the DeepNet-α champion) is present.

**Status.** L? — tentative null. Right-sign but sub-band; not yet a confirmed null.

**Scope.** tiny1m3m, on the ALiBi / DeepNet-α champion path.

## Evidence
- 021-value-residual won as a stand-alone lever on the bare model.
- On the recipe it vanishes: 208-value-residual-alibi null; 275-deepnet-value-residual
  Δ −0.0208 — right-sign but inside the band on the deepnet champion (no clean clearance).
  The stand-alone effect does not survive into the recipe.

## Falsifier
A 3-seed paired confirm atop the current champion showing Δ ≤ −0.02 with CI excluding 0 —
would flip this from "doesn't stack" to a stacking win.

## Why it matters
The canonical *non-stacking* case, and the counterweight to [[L007]]: a real stand-alone win
is not guaranteed to compound. Whether a lever stacks must be measured *on the recipe*, never
assumed from its bare-model result.

Links: [[L004]], [[L007]].
