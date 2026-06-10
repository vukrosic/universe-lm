> Source: https://victorchen96.github.io/auto_research/skill/paper-writing.html — scraped 2026-06-10; original source file docs/scientific_paper_writing_skill.md is not public — reconstructed from the HTML description page.

# 01 — Literature Survey

**IN:** topic + taxonomy keywords  
**OUT:** `references.bib` + `citation_plan.jsonl`

---

4-stage pipeline: Recall → Score (LQS) → Classify (A/B/C/D) → Upgrade (arXiv→accepted).

## Stage 1: High-Recall Retrieval

- 20–30 keyword queries via `search.py -o "site:arxiv.org ..."`
- Each taxonomy cell: 3+ query variants (core terms, synonyms, method names)
- Snowball: seed paper citation networks
- Target: 200–500 raw candidates

## Stage 2: LQS Multi-Dimensional Scoring

| Dimension | Weight | Scoring |
|---|---|---|
| Recency | 30% | 6mo=10, 1yr=8, 2yr=5, 3yr=3 |
| Citation Impact | 25% | cites/mo: ≥50=10, ≥10=8, ≥3=6 |
| Venue | 20% | Top-tier=10, Strong=7, Workshop=4 |
| Institution | 10% | Top lab=10, Top uni=9 |
| Acceptance | 15% | Accepted=10, Under review=5, None=3 |

Thresholds: LQS ≥ 7.0 must-cite · 5.0–7.0 conditional · < 5.0 drop

## Stage 3: Citation Depth Classification

| Level | Depth | Description | Volume per Chapter |
|---|---|---|---|
| A | 1–3 paragraphs | Section protagonist | 3–5 |
| B | 2–5 sentences | Important insight | 5–10 |
| C | 1 sentence | Supporting evidence | — |
| D | — | Dropped, not cited | — |

## Stage 4: Venue Upgrade

- Cross-check DBLP + OpenReview for acceptance status
- arXiv with "Accepted at X" → `@inproceedings`
- Target: arXiv-only ratio ≤ 60%

## Verification Rules

- Every 20 citations: title match, author, year, venue check
- Target: verification rate ≥ 80%; hallucinated = 0
- Year distribution: within-1yr ≥ 40%, accepted ≥ 30%
