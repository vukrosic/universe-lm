---
name: scientific-paper-writing
description: Hierarchical skill group orchestrating five sub-skills to autonomously produce survey papers (~8.5/10 score). Defines division of labor, phase routing, quality gates, and iterative review loops.
source: https://victorchen96.github.io/auto_research/skill/paper-writing.html
scraped: 2026-06-10
---

> Source: https://victorchen96.github.io/auto_research/skill/paper-writing.html — scraped 2026-06-10; original source file docs/scientific_paper_writing_skill.md is not public — reconstructed from the HTML description page.

# Scientific Paper Writing — Skill Group

Hierarchical skill group that orchestrates five sub-skills to autonomously produce 8.5/10 survey papers. Defines division of labor, phase routing, quality gates, and iterative review loops.

| Stat | Value |
|---|---|
| Sub-skills | 5 |
| Quality Gates | 4 (+ 1 blocking final gate) |
| Papers Produced | 3 |
| Avg Score | 8.5/10 |

## Sub-skills

| # | Name | Summary |
|---|---|---|
| 01 | Literature Survey | 4-stage pipeline: Recall → Score (LQS) → Classify (A/B/C/D) → Upgrade (arXiv→accepted) |
| 02 | Paper Structure & Logic | Chapter architecture, paragraph logic chains, taxonomy design, formal claims, hedge language, abstract-conclusion alignment |
| 03 | Experiment Design | 4-stage loop: Design → Execute → Iterate → Report |
| 04 | Academic Figures & Tables | High information-density tables and vector figures |
| 05 | Peer Review Simulation | Multi-persona scoring that drives the iteration loop |

## Workflow & Phase Routing

### Phase 0: Topic Selection (before pipeline starts)

3-question test: Scope? Angle? Audience?

### Phase 1: Draft (Iter 1–6, target: 6.0/10)

| Iter | Focus |
|---|---|
| 1 | [Structure] skeleton + §1-2 + compile |
| 2 | [Literature] Stage 1-2: recall + LQS scoring |
| 3 | [Structure] §3-6 core \|\| [Figures] 2+ figures |
| 4 | [Literature] Stage 3-4 \|\| [Structure] §7-8 |
| 5 | verify citations → compile → [Review] first score |
| 6 | route fixes → compile |

### Phase 2: Deep Improvement (Iter 7–9, target: 7.5–8.0)

| Iter | Focus |
|---|---|
| 7 | [Experiment] design + execute |
| 8 | [Figures] present results + [Structure] integrate |
| 9 | compile → [Review] → route fixes |

### Phase 3: Sprint (Iter 10+, target: 8.5+)

Loop: [Review] → weakness routing → fix → compile → [Review]

Stop when: score ≥ 8.5 OR Δ ≤ 0.3 for 2 rounds OR iter > 12

## Weakness Routing Table

When peer review identifies a weakness, route to the responsible sub-skill:

| Reviewer Weakness | Route To | Action |
|---|---|---|
| "Citation coverage insufficient" | Literature | Stage 1-2 targeted search |
| "Too many arXiv-only refs" | Literature | Stage 4 upgrade via DBLP |
| "Missing recent papers" | Literature | 2025-2026 focused search |
| "Structure unclear" | Structure | Reorganize + add transitions |
| "Analysis lacks depth" | Structure | Add Critical Assessment |
| "Taxonomy not novel" | Structure | Redesign multi-axis |
| "Claims too strong" | Structure | Hedge language downgrade |
| "No experiments" | Experiment | Design pilot study |
| "Experiment not rigorous" | Experiment | Add trials / ablation |
| "Tables incomparable" | Figures | Regroup + add Δ column |
| "Missing visualizations" | Figures | Add figure |
| "No error bars" | Figures | Add ± std |

## Quality Gates

Gates 1 & 2 can run in parallel; Gate 5 is blocking.

### Gate 1: Literature

- Citations ≥ 80 (draft) / ≥ pages×3 (final)
- Within 1yr ≥ 40%
- Accepted ≥ 30%
- arXiv-only ≤ 60%
- Verification rate ≥ 80%
- Every taxonomy cell ≥ 2 A/B refs

### Gate 2: Experiment

- Clear hypothesis pre-registered
- Statistical test reported (p or CI)
- ≥ 3 trials with std
- No ceiling/floor effect
- Links to specific paper claim
- (Bonus) Surprise finding

### Gate 3: Structure

- Compiles with 0 errors & 0 undefined refs
- Every .tex file ≤ 300 lines
- Abstract-conclusion alignment
- Inter-section transitions present
- Critical assessment in core sections
- ≥ 1 formal claim (conjecture/observation)
- Terminology consistent throughout

### Gate 4: Figures & Tables

- Tables ≥ 10, Figures ≥ 6 (full survey)
- booktabs format, no vertical lines
- Each carries a non-trivial insight
- Captions contain conclusion, not just description
- Every figure/table referenced in text
- Experimental data has mean ± std

### Gate 5: Final Review (Blocking)

- All Gates 1-4 passed
- PDF compiles cleanly
- Peer review score ≥ target (6.0/7.0/8.0/8.5 by phase)
- No regression: previously fixed weaknesses remain fixed
- Version bumped and snapshot saved

## Score Progression

| Score | Requirements Beyond Previous | Typical Additions |
|---|---|---|
| 6.0 | Complete draft, 80+ refs, compiles | Full 8 sections + basic tables |
| 7.0 | + logical transitions, quantitative data, gap analysis | Formal conjecture + grouped tables |
| 8.0 | + original experiment, critical assessment, 150+ refs | Multi-model pilot study + vector figures |
| 8.5 | + cross-validation, meta-analysis, key takeaways, proof sketch | Cross-benchmark table + deeper theory |

## Production Statistics

| Sub-skill | % of Time | Score Contribution | Key Output |
|---|---|---|---|
| Literature Survey | 20% | Foundation (without: ≤6.0) | 941 total citations across 3 papers |
| Structure & Logic | 35% | Main driver (6.0→7.5) | 190 pages of manuscript |
| Experiment Design | 20% | +1.0~1.5 points | 3,300+ API calls, 9 models evaluated |
| Figures & Tables | 10% | +0.5~1.0 points | 59+ tables, 26+ figures |
| Review + Integration | 15% | Drives iteration | 14 review rounds total |

## Adaptation Notes

This repo writes **mini-papers** from `autoresearch/` ablations (tiny1m3m, seed 42; see `autoresearch/briefs/PIPELINE.md`). Volume targets from the original skill do not transfer directly:

- **References:** 80+/150+ → scale to 15–30 for a mini-paper
- **Pages:** 50+ → scale to 4–8 pages (NeurIPS short / workshop format)
- **Tables/figures:** ≥10 tables / ≥6 figures → 2–4 tables, 1–2 figures
- **Iterations:** 10+ iters → 3–5 iters is realistic for a single campaign

What **does** transfer: the routing logic, the quality gates (scaled thresholds), the weakness-routing table, the anti-HARKing rules, the score calibration, and the reviewer personas.
