> Source: https://victorchen96.github.io/auto_research/skill/paper-writing.html — scraped 2026-06-10; original source file docs/scientific_paper_writing_skill.md is not public — reconstructed from the HTML description page.

# 03 — Experiment Design

**IN:** conjecture or gap  
**OUT:** `results.json` + `experiment_summary.md`

---

4-stage loop: Design (hypothesis) → Execute (API/GPU) → Iterate (adjust) → Report (structured JSON).

## Stage 1: Design (Most Important)

Must answer: "which paper claim does this support?"

Experiment spec must define:
- Hypothesis
- Independent / dependent variables
- Control variables
- Expected results

Principles: **falsifiable · minimal first · pre-registered · has control**

Statistical plan decided **BEFORE** running (no HARKing).

## Stage 2: Execute

| Path | Scale | Use Case |
|---|---|---|
| Path A: API | Hours, lightweight | Multi-model comparison, prompt ablation |
| Path B: GPU | Days, heavyweight | Agent training, reward shaping |

- API: 3–5 frontier models × 2–3 conditions × 15–25 tasks × 3 trials
- GPU: via cluster job submission + auto-monitoring loop

## Stage 3: Iterate

| Observation | Action |
|---|---|
| Ceiling effect | Increase difficulty |
| Floor effect | Decrease difficulty or check for bugs |
| Not significant | Increase trials or change hypothesis |
| Surprise finding | Design follow-up |

Max 5 iterations, then accept best result.

## Stage 4: Report (Data Only)

- Output: `results.json` — schema: `config` + `results` + `statistics` + `findings`
- Output: `experiment_summary.md` — purpose, results, limitations
- Does **NOT** produce LaTeX tables or figures — that is the Figures skill's job
