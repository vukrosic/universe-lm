# §01 — Scaling Laws: how to spend compute

The master section. Before data, before architecture, before a single hyperparameter, you
decide **how to split a fixed compute budget `C` between model size `N` and training tokens
`D`.** Get this wrong and no later choice recovers the wasted FLOPs.

Working approximations used throughout: **compute `C ≈ 6·N·D`** (FLOPs for a dense
transformer forward+backward), and loss modeled as `L(N,D) = E + A/N^α + B/D^β` (irreducible
loss `E` plus power-law terms in size and data).

## Rules in this section

| ID | Rule | Confidence |
|---|---|---|
| [FM-01.1](FM-01.1-chinchilla-compute-optimal.md) | Compute-optimal training scales N and D together, ≈20 tokens/param | **[E]** |
| [FM-01.2](FM-01.2-data-constrained-repetition.md) | Under a data ceiling, repeating data up to ~4 epochs is ~free | **[E]** |
| [FM-01.3](FM-01.3-vocabulary-scaling.md) | Larger models deserve larger vocabularies | **[C]** |
| [FM-01.4](FM-01.4-emergent-abilities-mirage.md) | "Emergent abilities" are largely a metric artifact, not a phase change | **[C]** |

## The one-paragraph version

For a fresh budget with abundant data, **scale parameters and tokens in lockstep** (FM-01.1):
doubling compute means roughly doubling *both*, landing near 20 tokens per parameter. If you
are **data-limited** (you have less than ~20 tokens/param of clean text), it is nearly free
to repeat what you have up to ~4 passes, and you should prefer a *smaller model trained for
more epochs* over a bigger one (FM-01.2). Size the **vocabulary** to the model — bigger
models want bigger vocabularies, and most shipped vocabularies are too small (FM-01.3).
Finally, do not over-interpret sudden capability jumps on benchmarks: most "emergence"
dissolves under a continuous metric (FM-01.4).

## Caveat that colors the whole section

These laws were fit at **10⁸–10¹⁰ parameters** and assume a roughly fixed, clean data
distribution and a standard transformer recipe. They are the best-replicated quantitative
laws in the field — *and* they are routinely misquoted outside their scope (inference-aware
budgets, reasoning vs memorization, distilled/curated data all shift the optimum). Treat the
numbers as anchors, not constants.
