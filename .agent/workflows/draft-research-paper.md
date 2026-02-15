---
description: A workflow to automatically draft a professional research paper in markdown based on experimental evidence at 1M token scale.
---

# Research Paper Drafting Workflow (FULLY AUTONOMOUS)

**CRITICAL: NEVER STOP.** Execute from start to finish without pausing for user input.

**PREREQUISITE**: This workflow requires completed experiments with multi-seed variance data at 1M tokens. If experiments haven't been run, use `/autonomous-research` first.

## Steps

1.  **Evidence Check**:
    - Verify that `docs/research/baseline_variance_1M.md` exists.
    - Verify that experiment results exist with ≥3 seeds.
    - Verify that Cohen's d ≥ 0.5 for the claimed improvement.
    - **If any of these are missing**: STOP and inform the user that experiments must be completed first.

2.  **Idea Contextualization**:
    - Read the original research proposal and the final experiment report from `docs/research/`.
    - Identify the core contribution, the statistical evidence, and the limitations.

3.  **Drafting**:
    - Use `paper-drafter` skill to write the paper with ALL experimental data.
    - Include: error bars, seed counts, effect sizes, wall-clock comparisons.
    - Include a Related Work section (mandatory).
    - Authors: **Vuk Rosić and Gemini**.

4.  **Paper Review**:
    - Use `paper-reviewer` skill to critique the draft with focus on statistical rigor.

5.  **Abstract Review**:
    - Use `abstract-reviewer` skill to check for overclaiming and jargon.

6.  **Revision**:
    - Use `paper-revisor` skill to address ALL feedback from both reviewers.
    - Ensure claims match evidence (no overclaiming).

7.  **File Creation**:
    - Save the final paper to `docs/papers/<topic>.md`.

8.  **LaTeX Generation**:
    - Use `latex-paper-generator` skill to create `docs/papers/paper.tex` and compile to PDF.
