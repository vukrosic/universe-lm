> Source: https://victorchen96.github.io/auto_research/skill/paper-writing.html — scraped 2026-06-10; original source file docs/scientific_paper_writing_skill.md is not public — reconstructed from the HTML description page.

# 04 — Academic Figures & Tables

**IN:** `results.json` + section placeholders  
**OUT:** `figures/*.pdf` + `tables/*.tex`

---

High information-density tables and vector figures. Presentation layer for all data in the paper.

## Table Types

| Type | Use | Info Density |
|---|---|---|
| Comparison Matrix | Methods × features | Very high |
| Benchmark Table | Models × metrics | High |
| Ablation Table | Conditions × results | High |
| Taxonomy Table | Classification visualization | Medium |
| Meta-analysis | Aggregated cross-paper data | Very high |

## Table Rules

- No vertical lines — booktabs three-line style only
- Alternating row color: `\rowcolor{gray!6}`
- Bold best results in each column
- All experimental data: mean ± std
- Caption must contain key finding, not just description

## Figure Types & Tool Priority

| Type | Tool | Format |
|---|---|---|
| Data-driven (curves, bars, heatmaps) | matplotlib | PDF |
| Architecture / flow diagrams | TikZ or SVG→PDF | PDF |
| Simple schematics | PIL | PNG (acceptable per reviewer feedback) |

Priority: **TikZ > matplotlib PDF > SVG→PDF > PIL PNG**

## Quality Checklist

- Vector format (PDF) preferred; PNG ≥ 300 DPI
- Font size ≥ 10pt after scaling
- Academic palette: blue `#2196F3`, red `#F44336`, green `#4CAF50`, orange `#FF9800`
- All axes labeled; all lines have legend
- Light grid (alpha=0.3) for readability
- Self-contained: understandable without reading main text

## Quantity Targets

| Paper Length | Tables | Figures |
|---|---|---|
| Full survey (50+ pages) | ≥ 10 | ≥ 6 |
| Short survey (30 pages) | ≥ 5 | ≥ 3 |
