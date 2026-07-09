> Source: https://victorchen96.github.io/auto_research/skill/paper-writing.html — scraped 2026-06-10; original source file docs/scientific_paper_writing_skill.md is not public — reconstructed from the HTML description page.

# 02 — Paper Structure & Logic

**IN:** `references.bib` + experiment findings  
**OUT:** `sections/*.tex` (full manuscript)

---

Chapter architecture, paragraph logic chains, taxonomy design, formal claims, hedge language, abstract-conclusion alignment.

## Chapter Architecture (Survey Standard)

| Section | Content |
|---|---|
| §1 Introduction | Hook → Gap → Contributions → Roadmap |
| §2 Background | Formal definitions, taxonomy overview |
| §3–6 Core | One method family per chapter, with critical assessment |
| §7 Benchmarks + Experiments | — |
| §8 Future | Specific open problems (Barrier + Attack vector) |
| §9 Conclusion | Numbered key findings (not repeat of abstract) |

## Paragraph Logic Patterns

| Pattern | Structure | Use Case |
|---|---|---|
| Claim-Evidence-Implication | Assert → Data → So what | Main body |
| Compare-Contrast | A → B → Difference → Trade-off | Method comparison |
| Concession-Rebuttal | Admit strength → But limitation | Critical analysis |
| Funnel | Broad → Narrow → This paper | Introduction |

## Taxonomy Design

- Multi-axis matrix (not flat list)
- MECE: mutually exclusive, collectively exhaustive
- Must have empty cells → gap analysis material
- Spanning methods show taxonomy tension (good)

## Formal Claims

- Default: Conjecture + Remark (not Theorem)
- Hedge ladder: **demonstrates** > **suggests** > **may** > **hypothesize**
- Rule: claim strength ≤ evidence strength

## Related Work Differentiation

- Mandatory comparison table with existing surveys
- "We're more recent" is **NOT** sufficient differentiation
- Need structural novelty: new taxonomy, new angle, new experiment
