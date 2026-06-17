# DRAFT — A paired difference measures the lever only if both arms are built by the identical path

**Hypothesis (derivation).** For a paired comparison Δ = f(candidate) − f(base) to measure
*only* the intended lever, both models must be instantiated by the **identical construction
path differing solely in that lever**. Any difference in *how* the two configs are built adds
a construction term to Δ:

> Δ_measured = [lever effect] + [construction-path delta].

If the construction paths are identical except for the lever, the second term is exactly 0
**by construction**, and Δ_measured is the lever effect. If they differ (different dataclass
machinery, attr-vs-field defaults, import order, fused-vs-reference code branch), the second
term is generally nonzero and *contaminates every comparison equally* — looking like a real,
uniform effect.

**Believed because.** This is the same logic as [[D001]] (paired seeds cancel *common-mode*
noise): differencing only cancels what is genuinely common. If construction is not common,
it does not cancel. Empirically confirmed by the **0.1070 artifact**: probing all `use_*`
flags with base built via `@dataclass`-field-subclass but candidates via `make_dataclass`
with class-attr defaults gave a **uniform step-0 logit diff of 0.1070 for every flag** —
including flags the model never reads. Rebuilding *both* arms via the identical
`make_dataclass(...)` path dropped a **nonsense-flag control to exactly 0.0**, isolating the
real per-flag effects.

**Test.** The control *is* the test: instantiate base and base+flag by the identical path;
a flag the model never consumes must give step-0 |Δ| = 0.0 (deterministic, CPU, no training).
Any nonzero value means the construction is not actually identical (or there is
nondeterminism) — not a real mechanism.

**Predicted.** Under identical construction, a no-op flag → step-0 |Δ| = 0.0 to float
precision; a wired flag → |Δ| > 1e-6.

**Promotes to.** **D** (a derivation; companion to D001). The numbers (0.1070 artifact,
0.0 control) confirm the *assumption* that the two construction paths differed — they do not
"prove" the claim, which is deductive.

**Falsifier (of the assumption, not the logic).** A nonsense flag giving nonzero step-0 |Δ|
**under provably identical construction and fixed seed** → either construction is not
identical or there is hidden nondeterminism; either way, re-audit before trusting any Δ.

**Why it matters.** This is the gate *before* the noise gate. Mis-construction produces
fake, uniform "effects" (the 0.1070 family) that survive paired differencing because they
are not common-mode — they would otherwise be admitted as spurious laws. Every architecture
probe must pass the 0.0-nonsense-flag control first.

**Evidence so far.** strong (deterministic, reproducible). CPU probe transcript this session;
see [[DRAFT-flag-inertness-on-champion]] which depends on this discipline.

**Blocked on.** Nothing — admit as D once a numbered slot is assigned.
