# §02 — Data: what to train on

Compute allocation (§01) decides *how many* tokens; this section decides *which* tokens. The
field's consensus, hard-won and now near-universal: **at equal compute, a clean, deduplicated,
well-mixed corpus beats a larger dirty one.** Data quality is a *multiplier* on your compute,
not an additive bonus.

## Rules in this section

| ID | Rule | Confidence |
|---|---|---|
| [FM-02.1](FM-02.1-quality-over-quantity.md) | Quality filtering beats raw scale at equal compute | **[E]** |
| [FM-02.2](FM-02.2-deduplication.md) | Deduplicate first; near-duplicates waste compute and hurt | **[E]** |
| FM-02.3 *(in this README)* | Set the domain mixture deliberately; oversample code & math | **[C]** |

## FM-02.3 — Domain mixture (the short version)

The strategic shift across model generations is visible: from "more internet" (GPT-3 ≈ 60%
Common Crawl) to "a precise recipe" (Llama-3 ≈ 42% math+code). Deliberately **oversampling
code and math** relative to their web frequency improves reasoning and structured output, and
is now standard. The exact proportions are a **[C]** open art — they depend on target use,
and the web-only (RefinedWeb) vs curated-multi-source (The Pile) debate is unsettled — but
"set the mixture on purpose, and oversample code/math" is established direction.

Ordering of operations (consensus): **dedup → quality filter → domain mix/sample.** Dedup
first so quality scores and mixing aren't distorted by duplicate mass.

## Relation to our ledger

Out of the ledger's structural scope (we hold data fixed to isolate architecture). Relevant as
the **fixed background** of our runs, and as a reminder of what we are *not* measuring: every
structural Δ we find is conditional on our particular tiny corpus. A different data regime
could move structural effect sizes — an unmeasured confounder worth stating in scope lines.
